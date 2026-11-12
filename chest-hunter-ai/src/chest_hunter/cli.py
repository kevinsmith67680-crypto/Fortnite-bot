"""Command-line interface: ``python -m chest_hunter <command> [options]``.

Every command operates on the real SQLite database named in the
configuration (or ``--db``). Ordinary user errors are reported as a single
``error: ...`` line on stderr with a non-zero exit code, never a traceback.

Exit codes: 0 success, 1 operation failed (validation, missing record,
conflict, database or configuration problem), 2 invalid command-line usage,
130 interrupted.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from chest_hunter import __version__
from chest_hunter.chest_repository import (
    DEFAULT_MIN_SPAWN_SAMPLES,
    SORT_ORDERS,
    ChestRepository,
)
from chest_hunter.config import Settings, load_config
from chest_hunter.database import SCHEMA_VERSION, DatabaseManager
from chest_hunter.demo_data import DEMO_CHESTS, seed_demo_chests
from chest_hunter.exceptions import ChestHunterError, DatabaseError, RecordNotFoundError
from chest_hunter.logging_config import VALID_LEVELS, configure_logging
from chest_hunter.models import STRATEGY_NAMES, ChestLocation, SessionStatus, SimulationSession
from chest_hunter.run_repository import RunRepository
from chest_hunter.session_repository import SessionRepository

if TYPE_CHECKING:  # pragma: no cover
    from chest_hunter.agent.runner import SimulationRunner

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

PROG = "python -m chest_hunter"


# ---------------------------------------------------------------------------
# Output formatting (standard library only)
# ---------------------------------------------------------------------------


def format_number(value: float, places: int = 3) -> str:
    """Format a float with at most ``places`` decimals and no trailing zeros."""
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    right_align: Sequence[int] = (),
) -> str:
    """Render rows as a plain-text table with a header underline.

    :param right_align: indexes of columns to right-align (typically numbers).
    """
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in cells:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(values: Sequence[str]) -> str:
        parts = [
            value.rjust(widths[i]) if i in right_align else value.ljust(widths[i])
            for i, value in enumerate(values)
        ]
        return "  ".join(parts).rstrip()

    lines = [render(list(headers)), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in cells)
    return "\n".join(lines)


def format_key_values(pairs: Sequence[tuple[str, str]], *, indent: int = 2) -> str:
    """Render ``label: value`` pairs with aligned values."""
    width = max((len(label) for label, _ in pairs), default=0) + 1
    pad = " " * indent
    return "\n".join(f"{pad}{label + ':':<{width}}  {value}" for label, value in pairs)


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _chest_table(chests: Sequence[ChestLocation]) -> str:
    rows = [
        (
            chest.id,
            chest.map_id,
            chest.location_name,
            format_number(chest.x),
            format_number(chest.y),
            format_number(chest.z),
            f"{chest.spawn_probability:.2f}",
            f"{chest.estimated_open_seconds:.2f}",
            _yes_no(chest.enabled),
        )
        for chest in chests
    ]
    headers = ("ID", "Map", "Location", "X", "Y", "Z", "Spawn P", "Open s", "Enabled")
    return format_table(headers, rows, right_align=(0, 3, 4, 5, 6, 7))


def _chest_details(chest: ChestLocation) -> str:
    return format_key_values(
        [
            ("ID", str(chest.id)),
            ("Map", chest.map_id),
            ("Location", chest.location_name),
            (
                "Position (x, y, z)",
                f"({format_number(chest.x)}, {format_number(chest.y)}, {format_number(chest.z)})",
            ),
            ("Spawn probability", f"{chest.spawn_probability:.2f}"),
            ("Est. open time", f"{chest.estimated_open_seconds:.2f} s"),
            ("Enabled", _yes_no(chest.enabled)),
            ("Created", chest.created_at or "-"),
            ("Updated", chest.updated_at or "-"),
        ]
    )


def _session_table(sessions: Sequence[SimulationSession]) -> str:
    rows = [
        (
            session.id,
            session.map_id,
            session.status.value,
            session.started_at,
            session.ended_at or "-",
            session.notes or "",
        )
        for session in sessions
    ]
    headers = ("ID", "Map", "Status", "Started (UTC)", "Ended (UTC)", "Notes")
    return format_table(headers, rows, right_align=(0,))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return value


def _mode_list(text: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(part.strip().lower().replace("-", "_")
                                for part in text.split(",") if part.strip()))
    allowed = ("oracle", "chest_vision", "visual")
    if not names or any(name not in allowed for name in names):
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list from oracle,chest-vision,visual, got {text!r}")
    return names


def _strategy_list(text: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(part.strip().lower() for part in text.split(",") if part.strip()))
    unknown = [name for name in names if name not in STRATEGY_NAMES]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list from {','.join(STRATEGY_NAMES)}, got {text!r}"
        )
    return names


def _log_level(text: str) -> str:
    level = text.strip().upper()
    if level not in VALID_LEVELS:
        raise argparse.ArgumentTypeError(
            f"expected one of {', '.join(VALID_LEVELS)}, got {text!r}"
        )
    return level


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors mention ``--help``."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\nTry '{self.prog} --help'.\n")


EPILOG = f"""\
examples:
  {PROG} init-db
  {PROG} seed-demo
  {PROG} list-chests --map research_arena_01
  {PROG} add-chest --map research_arena_01 --name "Warehouse Chest" \\
      --x 120 --y 340 --z 0 --probability 0.85
  {PROG} start-session --map research_arena_01
  {PROG} record-observation --session 1 --chest 3 --present --opened --seconds 1.8
  {PROG} complete-session --session 1
  {PROG} session-stats --session 1
  {PROG} map-stats --map research_arena_01
  {PROG} simulate --map research_arena_01 --strategy weighted --seed 42 --headless
  {PROG} simulate --render
  {PROG} benchmark --episodes 100 --strategies nearest,weighted

Run '{PROG} COMMAND --help' for the options of a command.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed for documentation and tests)."""
    # Global options are accepted before *or* after the command name.
    common = _Parser(add_help=False)
    common.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="configuration file (default: config/settings.json in the project root)",
    )
    common.add_argument(
        "--db",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="database file, overriding database.path from the configuration",
    )
    common.add_argument(
        "--log-level",
        type=_log_level,
        metavar="LEVEL",
        default=argparse.SUPPRESS,
        help=f"logging level ({', '.join(VALID_LEVELS)}); overrides the configuration",
    )

    parser = _Parser(
        prog=PROG,
        description=(
            "Manage the chest-location research database (Phase 1: data foundation only; "
            "no navigation, vision or game input)."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"chest-hunter-ai {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def add(name: str, help_text: str, handler: Callable[..., int]) -> argparse.ArgumentParser:
        sub = commands.add_parser(
            name,
            help=help_text,
            description=help_text,
            parents=[common],
        )
        sub.set_defaults(handler=handler)
        return sub

    add("init-db", "Create the database or migrate it to the current schema.", _cmd_init_db)

    sub = add("seed-demo", "Insert the demo research-arena chest locations (idempotent).",
              _cmd_seed_demo)
    sub.add_argument("--map", metavar="MAP_ID",
                     help="map to seed (default: simulation.default_map, or the scenario's map)")
    sub.add_argument("--scenario", choices=("research_arena", "visual_arena"),
                     default="research_arena",
                     help="research_arena: 35 demo locations (default); visual_arena: the 20-"
                          "location Phase 3 acceptance arena (map visual_arena_01)")

    sub = add("list-chests", "List known chest locations.", _cmd_list_chests)
    sub.add_argument("--map", metavar="MAP_ID", help="only this map (default: all maps)")
    state = sub.add_mutually_exclusive_group()
    state.add_argument("--enabled-only", action="store_true", help="only enabled locations")
    state.add_argument("--disabled-only", action="store_true", help="only disabled locations")
    sub.add_argument("--sort", choices=tuple(SORT_ORDERS), default="id",
                     help="sort order (default: id)")

    sub = add("show-chest", "Show one chest location.", _cmd_show_chest)
    sub.add_argument("--chest", type=_positive_int, required=True, metavar="ID")

    sub = add("add-chest", "Add a chest spawn location.", _cmd_add_chest)
    sub.add_argument("--map", metavar="MAP_ID",
                     help="map ID (default: simulation.default_map)")
    sub.add_argument("--name", required=True, help="human-readable location name")
    sub.add_argument("--x", type=float, required=True, help="X coordinate")
    sub.add_argument("--y", type=float, required=True, help="Y coordinate")
    sub.add_argument("--z", type=float, default=0.0, help="height coordinate (default: 0)")
    sub.add_argument("--probability", type=float, metavar="P",
                     help="spawn probability 0-1 (default: simulation.default_spawn_probability)")
    sub.add_argument("--seconds", type=float, metavar="S",
                     help="estimated open time (default: simulation.default_open_seconds)")
    sub.add_argument("--disabled", action="store_true", help="create the location disabled")

    sub = add("update-chest", "Update fields of a chest location.", _cmd_update_chest)
    sub.add_argument("--chest", type=_positive_int, required=True, metavar="ID")
    sub.add_argument("--map", metavar="MAP_ID")
    sub.add_argument("--name")
    sub.add_argument("--x", type=float)
    sub.add_argument("--y", type=float)
    sub.add_argument("--z", type=float)
    sub.add_argument("--probability", type=float, metavar="P")
    sub.add_argument("--seconds", type=float, metavar="S")
    toggle = sub.add_mutually_exclusive_group()
    toggle.add_argument("--enable", dest="enabled", action="store_const", const=True)
    toggle.add_argument("--disable", dest="enabled", action="store_const", const=False)

    sub = add("delete-chest",
              "Delete a chest location that has no observations (disable it otherwise).",
              _cmd_delete_chest)
    sub.add_argument("--chest", type=_positive_int, required=True, metavar="ID")

    sub = add("start-session", "Start a simulation session and print its ID.",
              _cmd_start_session)
    sub.add_argument("--map", metavar="MAP_ID",
                     help="map ID (default: simulation.default_map)")
    sub.add_argument("--notes", help="free-text notes")
    sub.add_argument("--quiet", action="store_true", help="print only the session ID")

    sub = add("complete-session", "Mark a running session as completed.",
              _cmd_complete_session)
    sub.add_argument("--session", type=_positive_int, required=True, metavar="ID")

    sub = add("abort-session", "Mark a running session as aborted.", _cmd_abort_session)
    sub.add_argument("--session", type=_positive_int, required=True, metavar="ID")

    sub = add("list-sessions", "List simulation sessions.", _cmd_list_sessions)
    sub.add_argument("--map", metavar="MAP_ID")
    sub.add_argument("--status", choices=[status.value for status in SessionStatus])
    sub.add_argument("--limit", type=_positive_int, metavar="N")

    sub = add("record-observation", "Record the outcome of visiting a chest location.",
              _cmd_record_observation)
    sub.add_argument("--session", type=_positive_int, required=True, metavar="ID")
    sub.add_argument("--chest", type=_positive_int, required=True, metavar="ID")
    presence = sub.add_mutually_exclusive_group(required=True)
    presence.add_argument("--present", dest="present", action="store_const", const=True,
                          help="a chest was at the location")
    presence.add_argument("--absent", dest="present", action="store_const", const=False,
                          help="no chest was at the location")
    sub.add_argument("--opened", action="store_true",
                     help="the chest was opened successfully (requires --present)")
    sub.add_argument("--seconds", type=float, default=0.0, metavar="S",
                     help="interaction time in seconds (default: 0)")

    sub = add("unvisited-chests",
              "List enabled locations not yet observed in a session.", _cmd_unvisited)
    sub.add_argument("--session", type=_positive_int, required=True, metavar="ID")

    sub = add("session-stats", "Show statistics for a session.", _cmd_session_stats)
    sub.add_argument("--session", type=_positive_int, required=True, metavar="ID")

    sub = add("map-stats", "Show chest statistics for a map.", _cmd_map_stats)
    sub.add_argument("--map", metavar="MAP_ID",
                     help="map ID (default: simulation.default_map)")
    sub.add_argument("--min-samples", type=_positive_int, default=DEFAULT_MIN_SPAWN_SAMPLES,
                     metavar="N",
                     help="samples required before reporting observed spawn frequency "
                          f"(default: {DEFAULT_MIN_SPAWN_SAMPLES})")

    sub = add("simulate", "Run the autonomous agent (oracle or vision-based) in the 2D simulator.",
              _cmd_simulate)
    sub.add_argument("--map", metavar="MAP_ID",
                     help="map whose chest locations are loaded (default: simulation.default_map)")
    sub.add_argument("--strategy", choices=STRATEGY_NAMES,
                     help="route-planning strategy (default: navigation.default_strategy)")
    sub.add_argument("--seed", type=_non_negative_int, metavar="N",
                     help="seed for map layout, chest spawns and rewards "
                          "(default: simulation.default_seed)")
    sub.add_argument("--episode", type=_non_negative_int, default=0, metavar="N",
                     help="episode number: different chest spawns on the same map (default: 0)")
    sub.add_argument("--max-ticks", type=_positive_int, metavar="N",
                     help="stop after N fixed ticks (default: simulation.max_ticks)")
    sub.add_argument("--speed", type=int, choices=(1, 2, 5, 10), default=1,
                     help="initial GUI playback speed; headless runs are never throttled")
    sub.add_argument("--agent", choices=("oracle", "visual", "chest-vision"), default="oracle",
                     help="oracle: Phase 2 agent with perfect map knowledge (default); visual: "
                          "Phase 3 vision-based agent; chest-vision: visual chests with the "
                          "true obstacle map")
    mode = sub.add_mutually_exclusive_group()
    mode.add_argument("--headless", dest="render", action="store_false",
                      help="run without a window and print statistics (default)")
    mode.add_argument("--render", dest="render", action="store_true",
                      help="open the Tkinter window (needs a display)")
    sub.set_defaults(render=False)

    sub = add("benchmark", "Compare route-planning strategies on paired seeded episodes.",
              _cmd_benchmark)
    sub.add_argument("--map", metavar="MAP_ID",
                     help="map to benchmark (default: simulation.default_map)")
    sub.add_argument("--episodes", type=_positive_int, default=100, metavar="N",
                     help="paired episodes per strategy (default: 100)")
    sub.add_argument("--strategies", type=_strategy_list, default=("nearest", "weighted"),
                     metavar="LIST",
                     help=f"comma-separated, from {','.join(STRATEGY_NAMES)} "
                          "(default: nearest,weighted)")
    sub.add_argument("--seed", type=_non_negative_int, metavar="N",
                     help="base seed (default: simulation.default_seed)")
    sub.add_argument("--max-ticks", type=_positive_int, metavar="N",
                     help="tick budget per episode (default: simulation.max_ticks)")
    sub.add_argument("--csv", metavar="PATH",
                     help="per-episode CSV output (default: data/benchmark_results.csv)")
    sub.add_argument("--vary-obstacles", action="store_true",
                     help="use a different obstacle layout for every episode index")
    sub.add_argument("--persist", action="store_true",
                     help="store benchmark sessions in the database instead of in memory")
    sub.add_argument("--quiet", action="store_true", help="no progress output")

    sub = add("benchmark-vision",
              "Compare oracle, chest-vision and full visual agents on paired episodes.",
              _cmd_benchmark_vision)
    sub.add_argument("--map", metavar="MAP_ID",
                     help="map to benchmark (default: simulation.default_map)")
    sub.add_argument("--episodes", type=_positive_int, default=100, metavar="N",
                     help="paired episodes per mode (default: 100)")
    sub.add_argument("--seed", type=_non_negative_int, metavar="N",
                     help="base seed (default: simulation.default_seed)")
    sub.add_argument("--max-ticks", type=_positive_int, metavar="N",
                     help="tick budget per episode (default: simulation.max_ticks)")
    sub.add_argument("--modes", type=_mode_list, default=("oracle", "chest_vision", "visual"),
                     metavar="LIST", help="comma-separated from oracle,chest-vision,visual")
    sub.add_argument("--output-dir", metavar="PATH",
                     help="directory for the CSV/JSON results (default: benchmark_results)")
    sub.add_argument("--detector-frames", type=_non_negative_int, default=60, metavar="N",
                     help="labelled frames for the detector-method comparison (0 = skip)")
    sub.add_argument("--quiet", action="store_true", help="no progress output")

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    settings: Settings
    db: DatabaseManager

    @property
    def chests(self) -> ChestRepository:
        return ChestRepository(self.db)

    @property
    def sessions(self) -> SessionRepository:
        return SessionRepository(self.db)


def _cmd_init_db(args: argparse.Namespace, ctx: CommandContext) -> int:
    db = ctx.db
    existed = db.exists()
    before = db.schema_version() if existed else 0
    version = db.initialize()
    if not existed:
        print(f"Initialised new database at {db.location} (schema version {version}).")
    elif before == version:
        print(
            f"Database at {db.location} is already initialised "
            f"(schema version {version}); existing data was preserved."
        )
    else:
        print(
            f"Migrated database at {db.location} from schema version {before} "
            f"to {version}; existing data was preserved."
        )
    return EXIT_OK


def _cmd_seed_demo(args: argparse.Namespace, ctx: CommandContext) -> int:
    if args.scenario == "visual_arena":
        from chest_hunter.scenarios import VISUAL_ARENA_CHESTS, VISUAL_ARENA_ID

        result = seed_demo_chests(ctx.db, args.map or VISUAL_ARENA_ID,
                                  chests=VISUAL_ARENA_CHESTS)
        total = result.total
    else:
        map_id = args.map or ctx.settings.simulation.default_map
        result = seed_demo_chests(ctx.db, map_id)
        total = len(DEMO_CHESTS)
    print(
        f"Seeded map {result.map_id}: {result.inserted} inserted, "
        f"{result.skipped} already present ({total} demo locations)."
    )
    return EXIT_OK


def _cmd_list_chests(args: argparse.Namespace, ctx: CommandContext) -> int:
    enabled = True if args.enabled_only else False if args.disabled_only else None
    chests = ctx.chests.get_all_chests(map_id=args.map, enabled=enabled, order_by=args.sort)
    if not chests:
        print("No chest locations found.")
        return EXIT_OK
    print(_chest_table(chests))
    print(f"\n{len(chests)} chest location(s).")
    return EXIT_OK


def _cmd_show_chest(args: argparse.Namespace, ctx: CommandContext) -> int:
    chest = ctx.chests.require_chest(args.chest)
    print(f"Chest location {chest.id}")
    print(_chest_details(chest))
    return EXIT_OK


def _cmd_add_chest(args: argparse.Namespace, ctx: CommandContext) -> int:
    sim = ctx.settings.simulation
    chest = ctx.chests.add_chest(
        map_id=args.map or sim.default_map,
        location_name=args.name,
        x=args.x,
        y=args.y,
        z=args.z,
        spawn_probability=(
            args.probability if args.probability is not None else sim.default_spawn_probability
        ),
        estimated_open_seconds=(
            args.seconds if args.seconds is not None else sim.default_open_seconds
        ),
        enabled=not args.disabled,
    )
    print(f"Added chest location {chest.id}.")
    print(_chest_details(chest))
    return EXIT_OK


def _cmd_update_chest(args: argparse.Namespace, ctx: CommandContext) -> int:
    chest = ctx.chests.update_chest(
        args.chest,
        map_id=args.map,
        location_name=args.name,
        x=args.x,
        y=args.y,
        z=args.z,
        spawn_probability=args.probability,
        estimated_open_seconds=args.seconds,
        enabled=args.enabled,
    )
    print(f"Updated chest location {chest.id}.")
    print(_chest_details(chest))
    return EXIT_OK


def _cmd_delete_chest(args: argparse.Namespace, ctx: CommandContext) -> int:
    if not ctx.chests.delete_chest(args.chest):
        raise RecordNotFoundError(f"Chest location {args.chest} does not exist.")
    print(f"Deleted chest location {args.chest}.")
    return EXIT_OK


def _cmd_start_session(args: argparse.Namespace, ctx: CommandContext) -> int:
    session = ctx.sessions.start_session(
        args.map or ctx.settings.simulation.default_map, notes=args.notes
    )
    if args.quiet:
        print(session.id)
    else:
        print(f"Started session {session.id} on map {session.map_id} at {session.started_at}.")
    return EXIT_OK


def _print_finished(session: SimulationSession) -> None:
    duration = session.duration_seconds
    duration_text = "-" if duration is None else f"{duration:.3f} s"
    print(
        f"Session {session.id} {session.status.value} at {session.ended_at} "
        f"(duration {duration_text})."
    )


def _cmd_complete_session(args: argparse.Namespace, ctx: CommandContext) -> int:
    _print_finished(ctx.sessions.complete_session(args.session))
    return EXIT_OK


def _cmd_abort_session(args: argparse.Namespace, ctx: CommandContext) -> int:
    _print_finished(ctx.sessions.abort_session(args.session))
    return EXIT_OK


def _cmd_list_sessions(args: argparse.Namespace, ctx: CommandContext) -> int:
    sessions = ctx.sessions.list_sessions(map_id=args.map, status=args.status, limit=args.limit)
    if not sessions:
        print("No sessions found.")
        return EXIT_OK
    print(_session_table(sessions))
    print(f"\n{len(sessions)} session(s).")
    return EXIT_OK


def _cmd_record_observation(args: argparse.Namespace, ctx: CommandContext) -> int:
    observation = ctx.sessions.record_observation(
        args.session,
        args.chest,
        chest_present=args.present,
        chest_opened=args.opened,
        interaction_seconds=args.seconds,
    )
    if observation.chest_opened:
        outcome = "present, opened"
    elif observation.chest_present:
        outcome = "present, NOT opened (failed interaction)"
    else:
        outcome = "absent"
    print(
        f"Recorded observation {observation.id} in session {observation.session_id}: "
        f"chest location {observation.chest_location_id} {outcome} "
        f"({observation.interaction_seconds:.2f} s)."
    )
    return EXIT_OK


def _cmd_unvisited(args: argparse.Namespace, ctx: CommandContext) -> int:
    chests = ctx.chests.get_unvisited_chests(args.session)
    if not chests:
        print(f"Every enabled chest location has been visited in session {args.session}.")
        return EXIT_OK
    print(_chest_table(chests))
    print(f"\n{len(chests)} unvisited location(s) in session {args.session}.")
    return EXIT_OK


def _cmd_session_stats(args: argparse.Namespace, ctx: CommandContext) -> int:
    session = ctx.sessions.require_session(args.session)
    stats = ctx.sessions.get_session_statistics(args.session)
    if stats.duration_seconds is None:
        duration = "unavailable (session still running)"
    else:
        duration = f"{stats.duration_seconds:.3f} s"
    if stats.chests_opened_per_minute is not None:
        rate = f"{stats.chests_opened_per_minute:.2f}"
    elif stats.duration_seconds is None:
        rate = "unavailable (session still running)"
    else:
        rate = "unavailable (zero duration)"
    print(f"Session {stats.session_id} on map {stats.map_id} [{stats.status.value}]")
    print(
        format_key_values(
            [
                ("Started (UTC)", session.started_at),
                ("Ended (UTC)", session.ended_at or "-"),
                ("Duration", duration),
                ("Locations visited", str(stats.locations_visited)),
                ("Observations", str(stats.total_observations)),
                ("Chests found", str(stats.chests_found)),
                ("Chests opened", str(stats.chests_opened)),
                ("Failed interactions", str(stats.failed_interactions)),
                ("Interaction time", f"{stats.total_interaction_seconds:.2f} s"),
                ("Chests opened/minute", rate),
            ]
        )
    )
    run = RunRepository(ctx.db).get_run(args.session)
    if run is not None:
        print("Autonomous agent run")
        print(
            format_key_values(
                [
                    ("Strategy", run.strategy),
                    ("Seed / episode", f"{run.seed} / {run.episode}"),
                    ("Outcome", run.outcome or "running"),
                    ("Simulated time", "-" if run.simulated_seconds is None
                     else f"{run.simulated_seconds:.3f} s ({run.ticks} ticks)"),
                    ("Travel distance", "-" if run.travel_distance is None
                     else f"{run.travel_distance:.1f} cells"),
                    ("Recoveries", "-" if run.recoveries is None else str(run.recoveries)),
                    ("Unreachable / abandoned", "-" if run.unreachable_targets is None
                     else f"{run.unreachable_targets} / {run.abandoned_targets}"),
                ]
                + ([("Failure reason", run.failure_reason)] if run.failure_reason else [])
            )
        )
    return EXIT_OK


def _print_run_report(ctx: CommandContext, runner: SimulationRunner) -> None:
    m = runner.metrics
    controller = runner.controller
    sim = ctx.settings.simulation
    rate = m.chests_per_minute
    tps = m.ticks_per_cpu_second
    units = ctx.settings.simulation.world_units_per_cell
    print(f"Autonomous run finished: session {runner.session_id}, strategy {runner.strategy}, "
          f"seed {runner.seed}, episode {runner.episode}")
    print(
        format_key_values(
            [
                ("Outcome", f"{m.outcome} ({controller.end_reason if controller else '-'})"),
                ("Simulated time", f"{m.simulated_seconds:.3f} s ({m.ticks} ticks at "
                                   f"{sim.tick_rate} Hz)"),
                ("Chest locations", f"{m.candidate_locations} candidates, "
                                    f"{m.locations_visited} visited, "
                                    f"{m.unreachable_targets} unreachable, "
                                    f"{m.abandoned_targets} abandoned"),
                ("Chests", f"{m.chests_spawned} spawned, {m.chests_found} found, "
                           f"{m.chests_opened} opened, {m.failed_interactions} failed "
                           "interaction(s)"),
                ("Chests opened/minute", "-" if rate is None
                 else f"{rate:.2f} (simulated time)"),
                ("Travel distance", f"{m.travel_distance:.1f} cells "
                                    f"({m.travel_distance * units:.0f} world units)"),
                ("Recoveries", str(m.recoveries)),
                ("Tokens (fictional)", str(m.rewards)),
                ("A* searches", f"{m.astar_searches} ({m.astar_cache_hits} cache hits) for "
                                f"{controller.routes_planned if controller else 0} route(s)"),
                ("Observations stored", f"{m.observations_recorded} in {m.db_flushes} "
                                        "transaction(s)"),
                ("CPU time", f"{m.cpu_seconds:.3f} s"
                             + ("" if tps is None else f" ({tps:,.0f} ticks per CPU second)")),
            ]
        )
    )
    evaluator = runner.vision_evaluator
    if evaluator is not None:
        v = evaluator.summary()
        controller = runner.visual_controller

        def fmt(x: object, digits: int = 3) -> str:
            return "-" if x is None else f"{x:.{digits}f}"

        print("Visual perception (evaluated against ground truth; the agent never saw it)")
        print(format_key_values([
            ("Frames analysed", f"{v['frames_analysed']} ({v['frames_failed']} failed captures)"),
            ("Detection precision", fmt(v["precision"])),
            ("Detection recall", fmt(v["recall"])),
            ("False positives", str(v["false_positives"])),
            ("Open/closed confusions", str(v["class_errors"])),
            ("Localisation success", fmt(v["localisation_success_rate"])),
            ("Pose error mean / max", f"{fmt(v['estimate_error_cells']['mean'], 4)} / "
                                      f"{fmt(v['estimate_error_cells']['max'], 3)} cells"),
            ("Perception time", f"{fmt(v['inference_ms']['mean'], 2)} ms mean, "
                                f"{fmt(v['inference_ms']['peak'], 2)} ms peak per frame"),
            ("Opened per simulator", f"{sum(1 for c in runner.env.chests.values() if c.opened)}"
                                     f" (agent recorded {runner.metrics.chests_opened})"),
            ("Targets selected / reached", f"{controller.targets_selected} / "
                                           f"{controller.targets_reached}"),
            ("Replans after new walls", str(controller.replans)),
        ]))
    if runner.session_id is not None:
        print()
        _cmd_session_stats(argparse.Namespace(session=runner.session_id), ctx)


def _cmd_simulate(args: argparse.Namespace, ctx: CommandContext) -> int:
    from chest_hunter.agent.runner import SimulationRunner, load_world, make_oracle

    settings = ctx.settings
    map_id = args.map or settings.simulation.default_map
    seed = settings.simulation.default_seed if args.seed is None else args.seed
    strategy = args.strategy or settings.navigation.default_strategy
    world = load_world(ctx.db, settings, map_id=map_id, map_seed=seed)
    oracle = make_oracle(settings)

    agent = args.agent.replace("-", "_")

    def factory(strategy_name: str) -> SimulationRunner:
        return SimulationRunner(ctx.db, settings, map_id=map_id, strategy=strategy_name,
                                seed=seed, episode=args.episode, max_ticks=args.max_ticks,
                                world=world, oracle=oracle, agent=agent,
                                evaluate=agent != "oracle")

    if args.render:
        finished: list[SimulationRunner] = []
        try:
            import tkinter

            from chest_hunter.simulation.renderer import run_gui
        except ImportError as exc:
            raise ChestHunterError(
                f"The GUI needs Tkinter ({exc}). Install it (e.g. 'sudo apt install "
                "python3-tk') or use --headless."
            ) from exc
        try:
            run_gui(factory, strategies=STRATEGY_NAMES, strategy=strategy,
                    cell_size=settings.simulation.cell_size,
                    tick_rate=settings.simulation.tick_rate,
                    render_rate=settings.simulation.render_rate,
                    speed=args.speed, on_finish=finished.append,
                    debug=settings.visual_debug if agent != "oracle" else None)
        except tkinter.TclError as exc:
            raise ChestHunterError(
                f"Could not open the GUI window ({exc}). Use --headless, or run with a display."
            ) from exc
        if not finished:
            print("GUI closed before an episode finished; any running session was aborted.")
            return EXIT_OK
        _print_run_report(ctx, finished[-1])
        return EXIT_OK if finished[-1].metrics.outcome in ("completed", "tick_limit") \
            else EXIT_FAILURE

    runner = factory(strategy)
    print(f"Simulating map {map_id} headless: {args.agent} agent, strategy {runner.strategy}, "
          f"seed {seed}, episode {args.episode}, max {runner.max_ticks} ticks ...")
    result = runner.run()
    _print_run_report(ctx, runner)
    return EXIT_OK if result.outcome in ("completed", "tick_limit") else EXIT_FAILURE


def _cmd_benchmark_vision(args: argparse.Namespace, ctx: CommandContext) -> int:
    from chest_hunter.analytics.visual_benchmark import format_summary, run_visual_benchmark

    settings = ctx.settings
    map_id = args.map or settings.simulation.default_map
    seed = settings.simulation.default_seed if args.seed is None else args.seed
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        settings.base_dir / "benchmark_results")
    step = max(1, (args.episodes * len(args.modes)) // 20)

    def progress(done: int, total: int) -> None:
        if not args.quiet and (done % step == 0 or done == total):
            print(f"  ... {done}/{total} episode runs", file=sys.stderr, flush=True)

    print(f"Visual benchmark on map {map_id}: {args.episodes} paired episode(s) x modes "
          f"{', '.join(args.modes)}, seed {seed}.")
    summary = run_visual_benchmark(ctx.db, settings, map_id=map_id, modes=args.modes,
                                   episodes=args.episodes, seed=seed, max_ticks=args.max_ticks,
                                   output_dir=output, detector_frames=args.detector_frames,
                                   progress=progress)
    print()
    print(format_summary(summary))
    print("Chests/min uses SIMULATED time; CPU and wall seconds are real time on this machine.")
    print(f"Results written to {output}/navigation_benchmark.csv, vision_benchmark.csv, "
          "summary.json")
    return EXIT_OK


def _cmd_benchmark(args: argparse.Namespace, ctx: CommandContext) -> int:
    from chest_hunter.analytics.benchmark import run_benchmark

    settings = ctx.settings
    map_id = args.map or settings.simulation.default_map
    seed = settings.simulation.default_seed if args.seed is None else args.seed
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else (
        settings.base_dir / "data" / "benchmark_results.csv")
    step = max(1, (args.episodes * len(args.strategies)) // 10)

    def progress(done: int, total: int) -> None:
        if not args.quiet and (done % step == 0 or done == total):
            print(f"  ... {done}/{total} episodes", file=sys.stderr, flush=True)

    budget = args.max_ticks or settings.simulation.max_ticks
    print(f"Benchmark on map {map_id}: {args.episodes} paired episode(s) x "
          f"{len(args.strategies)} strategies, seed {seed}, budget {budget} ticks "
          f"({budget / settings.simulation.tick_rate:.0f} simulated s), obstacles "
          f"{'vary per episode' if args.vary_obstacles else 'fixed'}.")
    report = run_benchmark(
        ctx.db, settings, map_id=map_id, strategies=args.strategies,
        episodes=args.episodes, seed=seed, max_ticks=args.max_ticks,
        vary_obstacles=args.vary_obstacles, csv_path=csv_path,
        persist_db=ctx.db if args.persist else None, progress=progress,
    )
    print()
    print(report.format_table())
    print()
    if report.comparisons:
        print(report.format_comparisons())
        print()
    print("Chests/min is measured in SIMULATED time (fixed 1/"
          f"{settings.simulation.tick_rate} s timestep); CPU s/episode is real process CPU "
          "time on this machine.")
    print(f"Setup (map + distance fields): {report.setup_seconds:.2f} s; "
          f"total wall time {report.total_wall_seconds:.1f} s.")
    print(f"Per-episode results written to {csv_path}")
    return EXIT_OK


def _cmd_map_stats(args: argparse.Namespace, ctx: CommandContext) -> int:
    map_id = args.map or ctx.settings.simulation.default_map
    stats = ctx.chests.get_chest_statistics(map_id, min_spawn_samples=args.min_samples)
    avg = (
        "unavailable (no enabled locations)"
        if stats.average_spawn_probability is None
        else f"{stats.average_spawn_probability:.3f} (enabled locations)"
    )
    if stats.observed_spawn_frequency is None:
        observed = (
            f"unavailable (needs >= {stats.min_spawn_samples} samples, "
            f"have {stats.spawn_samples})"
        )
    else:
        observed = f"{stats.observed_spawn_frequency:.3f} (from {stats.spawn_samples} samples)"
    print(f"Chest statistics for map {stats.map_id}")
    print(
        format_key_values(
            [
                ("Total chest locations", str(stats.total_locations)),
                ("Enabled chest locations", str(stats.enabled_locations)),
                ("Avg. spawn probability", avg),
                ("Total visits", str(stats.total_visits)),
                ("Chests found", str(stats.chests_found)),
                ("Successful openings", str(stats.successful_openings)),
                ("Observed spawn frequency", observed),
            ]
        )
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return EXIT_OK
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return EXIT_FAILURE


def _load_settings(args: argparse.Namespace) -> Settings:
    settings = load_config(getattr(args, "config", None))
    db_override = getattr(args, "db", None)
    if db_override:
        settings = settings.with_database_path(db_override)
    return settings


def _require_database(db: DatabaseManager) -> None:
    """Refuse to create a database implicitly; bring an existing one up to date."""
    if not db.exists():
        raise DatabaseError(
            f"Database not found at {db.location}. Run '{PROG} init-db' first."
        )
    if db.schema_version() != SCHEMA_VERSION:
        db.initialize()  # applies pending migrations, or refuses a newer schema


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code (never raises SystemExit)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and usage errors
        return _exit_code(exc)

    try:
        settings = _load_settings(args)
        configure_logging(getattr(args, "log_level", None) or settings.logging.level)
        db = DatabaseManager(settings.database.path)
        try:
            if args.command != "init-db":
                _require_database(db)
            return int(args.handler(args, CommandContext(settings=settings, db=db)))
        finally:
            db.close()
    except ChestHunterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except sqlite3.Error as exc:  # pragma: no cover - repositories translate these
        print(f"error: database error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
