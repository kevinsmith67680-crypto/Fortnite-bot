"""Phase 8 static scan of the repository (evidence for the component inventory).

Every finding carries a ``file:line`` reference. The scan is mechanical; its output lists
*candidates* that were then reviewed by hand (the review verdicts live in the audit report,
not here):

* markers - ``TODO``, ``FIXME``, ``XXX``, ``HACK`` in Python and C++ sources;
* stubs - functions whose body is only ``pass`` / ``...`` / ``raise NotImplementedError``
  (abstract-method placeholders included, they are classified by hand);
* constant returns - functions that do nothing but ``return <literal>``;
* exception handlers - bare ``except:``, ``except Exception`` and ``except BaseException``,
  classified as *re-raised*, *recorded* (the handler logs, counts, appends or returns an
  error value) or *silent* (``pass`` / ``continue`` only);
* disconnected modules - package modules no other module, CLI command or test imports;
* growth candidates - instance attributes created as empty lists or dicts and appended to or
  assigned into, without any visible bound (``maxlen``, ``del``, ``clear``, ``pop``, slicing).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MARKER = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")


@dataclass(slots=True)
class Finding:
    kind: str
    where: str
    detail: str


@dataclass(slots=True)
class ScanResult:
    files: int = 0
    lines: int = 0
    findings: list[Finding] = field(default_factory=list)
    modules: dict[str, dict[str, Any]] = field(default_factory=dict)

    def of(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        return {"files": self.files, "lines": self.lines, "counts": counts,
                "findings": [asdict(f) for f in self.findings], "modules": self.modules}


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _is_stub(body: list[ast.stmt]) -> bool:
    stmts = [s for s in body if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant)
                                     and isinstance(s.value.value, str))]   # docstrings
    if len(stmts) != 1:
        return False
    s = stmts[0]
    if isinstance(s, ast.Pass):
        return True
    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is ...:
        return True
    if isinstance(s, ast.Raise) and s.exc is not None:
        exc = s.exc.func if isinstance(s.exc, ast.Call) else s.exc
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"
    return False


def _constant_return(body: list[ast.stmt]) -> object:
    stmts = [s for s in body if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant))]
    if len(stmts) == 1 and isinstance(stmts[0], ast.Return) \
            and isinstance(stmts[0].value, ast.Constant):
        return stmts[0].value.value
    return _SENTINEL


_SENTINEL = object()


def _handler_kind(h: ast.ExceptHandler) -> str | None:
    t = h.type
    names: list[str] = []
    if t is None:
        names = ["<bare>"]
    elif isinstance(t, ast.Name):
        names = [t.id]
    elif isinstance(t, ast.Tuple):
        names = [e.id for e in t.elts if isinstance(e, ast.Name)]
    broad = [n for n in names if n in ("<bare>", "Exception", "BaseException")]
    return broad[0] if broad else None


def _handler_behaviour(h: ast.ExceptHandler) -> str:
    if any(isinstance(n, ast.Raise) for n in ast.walk(h)):
        return "re-raised"
    if all(isinstance(s, (ast.Pass, ast.Continue)) for s in h.body):
        return "silent"
    return "recorded"


def _python_files(root: Path, dirs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        out += sorted(p for p in (root / d).rglob("*.py") if "__pycache__" not in p.parts)
    return out


def scan(root: str | Path) -> ScanResult:
    root = Path(root)
    res = ScanResult()
    py = _python_files(root, ("src", "tests", "scripts"))
    cpp = sorted(p for ext in ("*.cpp", "*.h", "*.cs") for p in (root / "unreal").rglob(ext))
    imported: set[str] = set()
    package_modules: dict[str, Path] = {}
    for path in py + cpp:
        text = path.read_text(encoding="utf-8", errors="replace")
        res.files += 1
        res.lines += text.count("\n")
        for n, line in enumerate(text.splitlines(), 1):
            m = MARKER.search(line)
            if m and "MARKER = re.compile" not in line:
                res.findings.append(Finding("marker", f"{_rel(path, root)}:{n}",
                                            line.strip()[:120]))
    for path in py:
        rel = _rel(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        if rel.startswith("src/chest_hunter/"):
            mod = rel[4:-3].replace("/", ".").removesuffix(".__init__")
            package_modules[mod] = path
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                for a in node.names:
                    imported.add(f"{node.module}.{a.name}")
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                where = f"{rel}:{node.lineno}"
                if _is_stub(node.body):
                    res.findings.append(Finding("stub", where, node.name))
                value = _constant_return(node.body)
                if value is not _SENTINEL and not node.name.startswith("__"):
                    res.findings.append(Finding("constant_return", where,
                                                f"{node.name} -> {value!r}"))
            elif isinstance(node, ast.ExceptHandler):
                kind = _handler_kind(node)
                if kind is not None:
                    res.findings.append(Finding(f"broad_except:{_handler_behaviour(node)}",
                                                f"{rel}:{node.lineno}", kind))
        if rel.startswith("src/"):
            res.findings += _growth_candidates(tree, rel)
    for mod, path in sorted(package_modules.items()):
        used = any(i == mod or i.startswith(mod + ".") for i in imported)
        res.modules[mod] = {"file": _rel(path, root), "imported": used,
                            "lines": path.read_text(encoding="utf-8").count("\n")}
        if not used and not mod.endswith("__main__") and mod.count(".") > 0:
            res.findings.append(Finding("disconnected_module", _rel(path, root), mod))
    return res


def _growth_candidates(tree: ast.Module, rel: str) -> list[Finding]:
    out = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        created: dict[str, int] = {}
        grown: set[str] = set()
        bounded: set[str] = set()
        for node in ast.walk(cls):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                for t in targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                            and t.value.id == "self":
                        if isinstance(value, (ast.List, ast.Dict)) and not getattr(
                                value, "elts", getattr(value, "keys", [1])):
                            created.setdefault(t.attr, node.lineno)
                        elif value is not None and isinstance(value, ast.Call) and \
                                "maxlen" in ast.dump(value):
                            bounded.add(t.attr)
                        elif t.attr in created and isinstance(value, (ast.List, ast.Dict,
                                                                      ast.Subscript,
                                                                      ast.ListComp,
                                                                      ast.DictComp)):
                            bounded.add(t.attr)               # re-created / sliced
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) \
                            and isinstance(t.value.value, ast.Name) \
                            and t.value.value.id == "self":
                        grown.add(t.value.attr)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) \
                        and owner.value.id == "self":
                    if node.func.attr in ("append", "extend", "add", "setdefault",
                                          "update", "insert"):
                        grown.add(owner.attr)
                    if node.func.attr in ("pop", "popleft", "clear", "remove", "discard",
                                          "popitem"):
                        bounded.add(owner.attr)
            elif isinstance(node, ast.Delete):
                for t in node.targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute):
                        bounded.add(t.value.attr)
        for attr, line in created.items():
            if attr in grown and attr not in bounded:
                out.append(Finding("growth_candidate", f"{rel}:{line}",
                                   f"{cls.name}.{attr}"))
    return out
