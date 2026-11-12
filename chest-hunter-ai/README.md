# chest-hunter-ai: Phase 1 data foundation

A lightweight research project working toward an autonomous agent that
navigates a Fortnite-style **simulated** environment, finds chests, opens them,
and maximises chests opened per minute.

> **Phase 1 is data infrastructure only.** It stores possible chest spawn
> locations, simulation sessions and chest observations in SQLite, and gives
> you a command-line tool to manage them. It contains **no** autonomous
> gameplay agent: no movement, pathfinding, computer vision or game input.
> It never talks to a real game client, game service or network.

---

## 1. Purpose and scope

| In scope (Phase 1) | Out of scope (later phases or never) |
|---|---|
| Persistent SQLite database of possible chest spawn points | Movement, navigation, A* pathfinding (Phase 2) |
| Configurable spawn probabilities and opening-time estimates | Computer vision, screen capture |
| Simulation sessions with a strict lifecycle | Game input / controller emulation |
| Per-session chest observations (present / opened / duration) | Any interaction with the live Fortnite client, accounts or matchmaking |
| Session and map statistics (e.g. chests opened per minute) | Memory reading, process injection, anti-cheat evasion (never) |
| CLI, tests and documentation | Web servers, cloud services, GPUs, ML frameworks |

The demo dataset describes a **fictional** 1000 x 1000 research arena. Its
coordinates are invented and are not Fortnite chest locations.

## 2. System architecture

```text
            +----------------------------------------------+
  user ---> | cli.py   (argparse, table output, exit codes) |
            +-----------------------+----------------------+
                                    |
         +--------------------------+--------------------------+
         |                          |                          |
+--------v---------+   +------------v-----------+   +----------v---------+
| config.py        |   | chest_repository.py    |   | session_repository |
| JSON settings,   |   | CRUD, filters, unvisited|  | lifecycle, record_ |
| path resolution  |   | chests, map statistics |   | observation, stats |
+------------------+   +------------+-----------+   +----------+---------+
                                    |                          |
                       +------------v--------------------------v---------+
                       | database.py  DatabaseManager                    |
                       | connection, PRAGMA foreign_keys, transactions,  |
                       | savepoints, migrations (PRAGMA user_version)    |
                       +------------------------+------------------------+
                                                |
                                     data/chest_locations.db (SQLite)

models.py      dataclasses (ChestLocation, SimulationSession, ChestObservation,
               ChestStatistics, SessionStatistics), validation, UTC timestamps
exceptions.py  ChestHunterError hierarchy (ValidationError, RecordNotFoundError, ...)
demo_data.py   deterministic 35-location demo arena + idempotent seeding
logging_config.py  stderr logging setup
```

Design principles:

* **Standard library only.** No runtime dependencies (`dependencies = []`).
* **Defence in depth.** Every rule is validated in Python (clear messages)
  *and* enforced by SQLite (CHECK constraints, foreign keys, a partial unique
  index and triggers), so a buggy future module cannot corrupt the data.
* **Typed models out, parameterised SQL in.** Repositories return dataclasses,
  never raw tuples. All values are bound with `?` placeholders.
* **Explicit transactions.** Multi-step writes run inside
  `BEGIN IMMEDIATE ... COMMIT`; nested blocks become SAVEPOINTs.
* **Injectable clock.** Repositories accept a `clock` callable, so a future
  simulator can run on simulated time and tests get exact durations.

### Source layout

```text
chest-hunter-ai/
├── pyproject.toml          # packaging (setuptools, src layout, no dependencies)
├── README.md
├── config/settings.json    # default configuration
├── data/                   # SQLite database lives here (git-ignored)
├── src/chest_hunter/
│   ├── __init__.py         # public API re-exports
│   ├── __main__.py         # python -m chest_hunter
│   ├── cli.py              # argparse CLI
│   ├── config.py           # configuration loader / validator
│   ├── database.py         # DatabaseManager, schema, migrations
│   ├── models.py           # dataclasses, validation, timestamps
│   ├── chest_repository.py
│   ├── session_repository.py
│   ├── demo_data.py        # demo arena dataset + seeding (used by CLI and script)
│   ├── exceptions.py       # error hierarchy
│   ├── logging_config.py
│   └── py.typed            # ships type information (PEP 561)
├── scripts/seed_demo_chests.py
└── tests/                  # unittest suite (temporary databases only)
```

`exceptions.py`, `demo_data.py` and `tests/test_config.py` were added beyond the
requested layout: the error types are shared by every module, and the demo data
must live inside the package so the installed CLI's `seed-demo` command can use it.

## 3. Hardware requirements

Any ordinary computer that can run Python 3.11 is enough.

* **CPU:** any x86-64 or ARM64 processor. No GPU is used.
* **Memory:** about 16 MB peak per CLI command, measured on Linux.
* **Disk:** a few MB for the package. The demo database is about 60 KB.
* **Speed:** each CLI command takes under 100 ms, including interpreter
  startup. Queries are indexed and touch at most a few thousand rows.
* No background processes, servers or network access.

## 4. Supported operating systems

Pure Python with no native extensions beyond the `sqlite3` module that ships
with CPython:

* **Linux:** tested with Python 3.11 and 3.13 (SQLite 3.53).
* **macOS, Windows 10/11:** expected to work with Python 3.11+ from python.org
  or a package manager, but not tested yet. The code uses `pathlib`
  throughout, avoids POSIX-only APIs and prints ASCII-only output.

The schema uses partial indexes and triggers, which need SQLite 3.8 or newer.
The SQLite that comes with any supported Python is much newer than that. To
check yours, run `python -c "import sqlite3; print(sqlite3.sqlite_version)"`.

## 5. Installation

```bash
git clone <your-repository-url>
cd <repository>/chest-hunter-ai
```

Install in **editable mode** (recommended). The default configuration and
database paths are located relative to this project directory:

```bash
python -m pip install -e .
```

This installs the `chest_hunter` package and a `chest-hunter` console command.
`python -m chest_hunter ...` and `chest-hunter ...` are equivalent.

Running without installing also works from the project directory:
`PYTHONPATH=src python -m chest_hunter --help` (on Windows PowerShell:
`$env:PYTHONPATH="src"; python -m chest_hunter --help`).

## 6. Virtual environment setup

```bash
# Linux / macOS
python3.11 -m venv .venv          # any Python >= 3.11
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Deactivate with `deactivate`. Delete `.venv/` to remove everything.

## 7. Configuration

`config/settings.json`:

```json
{
  "database":   { "path": "data/chest_locations.db" },
  "simulation": {
    "default_map": "research_arena_01",
    "default_spawn_probability": 0.8,
    "default_open_seconds": 2.0
  },
  "logging":    { "level": "INFO" }
}
```

* Relative paths are resolved against the **project root** (the directory that
  holds `pyproject.toml`), never the current working directory.
* Every field is validated: probability in [0, 1], non-negative finite open
  time, a valid map ID, and a level of `CRITICAL`, `ERROR`, `WARNING`, `INFO`
  or `DEBUG`. Unknown keys inside known sections are rejected to catch typos.
  Unknown *top-level* sections are allowed, so later phases can add their own.
* Overrides:
  * `--config PATH` or the environment variable `CHEST_HUNTER_CONFIG` selects
    another configuration file.
  * `--db PATH` uses another database file for a single command.
  * `--log-level LEVEL` overrides `logging.level`.
  * `CHEST_HUNTER_HOME` changes the project root, which is useful for a
    non-editable install.

The `simulation.*` defaults are used when a CLI command omits `--map`,
`--probability` or `--seconds`.

## 8. Database structure

File: `data/chest_locations.db`. Schema version is stored in
`PRAGMA user_version` (currently **1**). Foreign keys are enabled on every
connection. All timestamps are UTC ISO 8601 strings with microseconds, e.g.
`2026-11-12T13:17:26.629499+00:00`.

### `chest_locations`: possible spawn points

| Column | Type | Rules |
|---|---|---|
| `id` | INTEGER PK | |
| `map_id` | TEXT | 1-64 chars: letters, digits, `_ . -` |
| `location_name` | TEXT | non-empty, <= 200 chars |
| `x`, `y` | REAL | finite (NaN / +-inf rejected) |
| `z` | REAL | finite, default 0 (height; negative = below ground) |
| `spawn_probability` | REAL | 0 <= p <= 1 (an *estimate*, not a guarantee) |
| `estimated_open_seconds` | REAL | finite, >= 0 |
| `enabled` | INTEGER | 0 / 1 |
| `created_at`, `updated_at` | TEXT | UTC ISO 8601 |

Indexes: `UNIQUE (map_id, x, y, z)`, so one spawn point per exact position per
map (this also serves map lookups), plus `(map_id, enabled)`.

There is intentionally **no `visited` column**. Whether a location was visited
or opened is derived from `chest_observations` *for a given session*, so the
same spawn point can be visited again in every new session.

### `simulation_sessions`: research runs

| Column | Type | Rules |
|---|---|---|
| `id` | INTEGER PK | |
| `map_id` | TEXT | map the session runs on |
| `started_at` | TEXT | UTC ISO 8601 |
| `ended_at` | TEXT | NULL while running; >= `started_at` |
| `status` | TEXT | `running`, `completed` or `aborted` |
| `notes` | TEXT | optional |

Lifecycle: `running -> completed` or `running -> aborted`. Both end states
are final. A trigger blocks any status or `ended_at` change on a finished
session, and a CHECK ensures `status = 'running'` exactly when `ended_at IS NULL`.

### `chest_observations`: what happened at a location in a session

| Column | Type | Rules |
|---|---|---|
| `id` | INTEGER PK | |
| `session_id` | INTEGER FK -> `simulation_sessions.id` | `ON DELETE RESTRICT` |
| `chest_location_id` | INTEGER FK -> `chest_locations.id` | `ON DELETE RESTRICT` |
| `observed_at` | TEXT | UTC ISO 8601 |
| `chest_present` | INTEGER | 0 / 1 |
| `chest_opened` | INTEGER | 0 / 1; `opened = 1` requires `present = 1` |
| `interaction_seconds` | REAL | finite, >= 0 |

Enforced rules (in Python **and** SQLite):

* Observations can only be added to a **running** session (trigger).
* The location must be on the **same map** as the session (trigger).
* At most **one successful opening** per location per session (partial
  unique index `WHERE chest_opened = 1`). Failed attempts and later revisits
  are still allowed.
* A location with observations **cannot be deleted** (FK `RESTRICT`) or moved
  to another map (trigger). Disable it instead.
* Additionally (Python only): observations of **disabled** locations are
  rejected.

## 9. CLI usage

Global options (`--config`, `--db`, `--log-level`) can be placed before or
after the command name. `python -m chest_hunter COMMAND --help` documents
every option.

| Command | Purpose |
|---|---|
| `init-db` | Create the database or apply pending migrations (idempotent, never erases data) |
| `seed-demo [--map MAP]` | Insert the 35 demo arena locations (idempotent) |
| `list-chests [--map MAP] [--enabled-only \| --disabled-only] [--sort id\|map\|name\|probability\|position]` | List locations |
| `show-chest --chest ID` | Show one location |
| `add-chest --name N --x X --y Y [--z Z] [--map MAP] [--probability P] [--seconds S] [--disabled]` | Add a location |
| `update-chest --chest ID [--name/--x/--y/--z/--map/--probability/--seconds] [--enable \| --disable]` | Edit a location |
| `delete-chest --chest ID` | Delete a never-observed location |
| `start-session [--map MAP] [--notes TEXT] [--quiet]` | Start a session and print its ID (`--quiet` prints only the ID) |
| `record-observation --session ID --chest ID (--present \| --absent) [--opened] [--seconds S]` | Record a visit outcome |
| `complete-session --session ID` / `abort-session --session ID` | End a running session |
| `list-sessions [--map MAP] [--status STATUS] [--limit N]` | List sessions |
| `unvisited-chests --session ID` | Enabled locations not yet observed in that session |
| `session-stats --session ID` | Session statistics |
| `map-stats [--map MAP] [--min-samples N]` | Map statistics |

Every command except `init-db` refuses to run until the database exists, so
a typo in `--db` cannot silently create an empty database.

**Exit codes:** `0` success; `1` the operation failed (validation error,
missing record, rule conflict, configuration or database problem); `2`
invalid command-line usage; `130` interrupted. Errors are printed as one
`error: ...` line on stderr, without a Python traceback.

### Example workflow

```bash
python -m chest_hunter init-db
python -m chest_hunter seed-demo
python -m chest_hunter list-chests --map research_arena_01
python -m chest_hunter add-chest --map research_arena_01 --name "Warehouse Chest" \
    --x 120 --y 340 --z 0 --probability 0.85
python -m chest_hunter start-session --map research_arena_01
python -m chest_hunter record-observation --session 1 --chest 3 --present --opened --seconds 1.8
python -m chest_hunter record-observation --session 1 --chest 4 --absent
python -m chest_hunter complete-session --session 1
python -m chest_hunter session-stats --session 1
python -m chest_hunter map-stats --map research_arena_01
```

Scripting tip: `SESSION=$(python -m chest_hunter start-session --quiet)`.

### Statistics definitions

**Session statistics** (`session-stats`, `SessionRepository.get_session_statistics`):

| Field | Definition |
|---|---|
| Locations visited | Distinct locations with at least one observation in the session |
| Observations | Every recorded observation, including revisits |
| Chests found | Distinct locations where a chest was observed **present** |
| Chests opened | Successful openings (at most one per location per session) |
| **Failed interactions** | Observations where a chest **was present but was not opened**. Each failed attempt counts, even if a later retry succeeded. Visiting an empty spawn point (`--absent`) is **not** a failed interaction, because there was nothing to open. |
| Duration | `ended_at - started_at`. *Unavailable* while the session is running |
| Chests opened / minute | `chests_opened / (duration / 60)`. *Unavailable* while running or when the duration is 0 (no division by zero). `0.00` for a finished session with no openings |

**Map statistics** (`map-stats`, `ChestRepository.get_chest_statistics`):

| Field | Definition |
|---|---|
| Total / enabled chest locations | Row counts for the map |
| Avg. spawn probability | Mean configured `spawn_probability` of **enabled** locations |
| Total visits | All observations of the map's locations, across all sessions |
| Chests found / Successful openings | Observations with a chest present / opened |
| Observed spawn frequency | Share of distinct *(session, location)* samples in which a chest was ever present. Revisiting an already-opened (now empty) chest in the same session does not count as a missing spawn. Reported as **unavailable** until at least `--min-samples` samples exist (default 5). The tool never invents a probability. |

## 10. Running the tests

```bash
python -m unittest discover -s tests -v
```

The package must be importable: install it with `pip install -e .` or prefix
the command with `PYTHONPATH=src`. The suite (127 tests) covers the database
layer, both repositories, configuration, the CLI (in-process, via
`python -m chest_hunter` in a subprocess, and via the seed script) and the
demo dataset. Every test uses its own temporary directory and SQLite file.
One test also checks that the real `data/chest_locations.db` is not created
or modified.

## 11. Example output

These come from a real run with the default configuration. Timestamps will
differ on your machine.

```text
$ python -m chest_hunter init-db
2026-11-12 13:17:26,291 INFO     chest_hunter.database: Applied schema migration 1 to .../data/chest_locations.db
Initialised new database at .../data/chest_locations.db (schema version 1).

$ python -m chest_hunter init-db
Database at .../data/chest_locations.db is already initialised (schema version 1); existing data was preserved.

$ python -m chest_hunter seed-demo
Seeded map research_arena_01: 35 inserted, 0 already present (35 demo locations).

$ python -m chest_hunter seed-demo
Seeded map research_arena_01: 0 inserted, 35 already present (35 demo locations).

$ python -m chest_hunter list-chests
ID  Map                Location                                 X    Y   Z  Spawn P  Open s  Enabled
--  -----------------  -------------------------------------  ---  ---  --  -------  ------  -------
 1  research_arena_01  Central Warehouse - Loading Bay        470  455   0     0.90    2.00  yes
 2  research_arena_01  Central Warehouse - Office             525  540   4     0.75    2.00  yes
 3  research_arena_01  Central Warehouse - Mezzanine          505  500   8     0.85    2.00  yes
...
35 chest location(s).

$ python -m chest_hunter start-session --map research_arena_01
Started session 1 on map research_arena_01 at 2026-11-12T13:17:26.629499+00:00.

$ python -m chest_hunter record-observation --session 1 --chest 3 --present --opened --seconds 1.8
Recorded observation 1 in session 1: chest location 3 present, opened (1.80 s).

$ python -m chest_hunter complete-session --session 1
Session 1 completed at 2026-11-12T13:17:28.790703+00:00 (duration 2.161 s).

$ python -m chest_hunter session-stats --session 1
Session 1 on map research_arena_01 [completed]
  Started (UTC):         2026-11-12T13:17:26.629499+00:00
  Ended (UTC):           2026-11-12T13:17:28.790703+00:00
  Duration:              2.161 s
  Locations visited:     1
  Observations:          1
  Chests found:          1
  Chests opened:         1
  Failed interactions:   0
  Interaction time:      1.80 s
  Chests opened/minute:  27.76

$ python -m chest_hunter map-stats --map research_arena_01
Chest statistics for map research_arena_01
  Total chest locations:     35
  Enabled chest locations:   35
  Avg. spawn probability:    0.736 (enabled locations)
  Total visits:              1
  Chests found:              1
  Successful openings:       1
  Observed spawn frequency:  unavailable (needs >= 5 samples, have 1)

$ python -m chest_hunter add-chest --name Bad --x 1 --y 1 --probability 1.5
error: spawn_probability must be between 0 and 1, got 1.5.          # exit code 1

$ python -m chest_hunter record-observation --session 1 --chest 3 --present --opened
error: Session 1 is completed; observations can only be recorded for running sessions.
```

## 12. Programmatic use (for Phase 2 modules)

```python
from chest_hunter import ChestRepository, DatabaseManager, SessionRepository

with DatabaseManager("data/chest_locations.db") as db:
    db.initialize()                              # idempotent
    chests, sessions = ChestRepository(db), SessionRepository(db)

    session = sessions.start_session("research_arena_01")
    for chest in chests.get_unvisited_chests(session.id):
        x, y, z = chest.position
        # ...a Phase 2 planner/simulator decides where to go here...
        sessions.record_observation(session.id, chest.id, chest_present=True,
                                    chest_opened=True, interaction_seconds=1.9)
        break
    sessions.complete_session(session.id)
    print(sessions.get_session_statistics(session.id))
```

Errors derive from `chest_hunter.ChestHunterError`: `ValidationError`,
`RecordNotFoundError`, `DuplicateChestError`, `ChestInUseError`,
`SessionStateError`, `DuplicateOpeningError`, `DatabaseError`, `ConfigError`.
Use `db.transaction()` to group several repository calls atomically, since
nested calls become savepoints.

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named chest_hunter` | Activate your virtual environment and run `pip install -e .`, or prefix commands with `PYTHONPATH=src`. |
| `error: Database not found at ... Run 'python -m chest_hunter init-db' first.` | Run `init-db`, or check that `--db` / `database.path` points where you expect. |
| `error: Configuration file not found` | You probably did a non-editable install. Use `pip install -e .`, pass `--config PATH`, or set `CHEST_HUNTER_CONFIG` / `CHEST_HUNTER_HOME`. |
| `error: Invalid configuration in ...` | The message names the bad field. Compare with section 7. |
| `error: ... has schema version N, but this software only supports up to version M` | The database was written by a newer release. Upgrade this package; the database is left untouched. |
| `error: Failed to apply schema migration 1 ... already exists` | The file already holds unrelated tables. Use a different `--db` path. Existing data is never dropped. |
| `error: Could not start a transaction ... database is locked` | Another process holds a write lock. Close it and retry (the wait timeout is 5 s). |
| `error: Chest location N is referenced by ... observation(s)` | History is protected. Use `update-chest --chest N --disable` instead of deleting. |
| `error: ... was already opened in session N` | Each location can be opened once per session. Record a revisit with `--absent`, or start a new session. |
| Want a clean slate | Delete `data/chest_locations.db` (this is irreversible) and run `init-db` again. |
| More detail | Add `--log-level DEBUG`. |

## 14. Roadmap

* **Phase 1 (this release):** data foundation, CLI, tests.
* **Phase 2: route planning and navigation in a 2D simulator**
  * 2D simulation environment that renders `chest_locations` (x, y) directly;
    obstacles/terrain in a new table or map file keyed by `map_id`, added via
    a schema migration.
  * Simulated player position, and an injected simulation clock for
    `SessionRepository`.
  * A* obstacle-aware pathfinding between `ChestLocation.position`s.
  * Chest selection using `spawn_probability`, `estimated_open_seconds`,
    travel time and per-session `get_unvisited_chests()`.
  * Dynamic route recalculation and automatic (simulated) chest interaction,
    each outcome recorded with `record_observation()`.
  * Performance tracking from `get_session_statistics()` (chests opened per
    minute) and Bayesian updates of spawn probability from observed frequency.
* **Phase 3 (possible):** a custom Unreal Engine research environment that
  reuses the same database and repositories.

Never planned: live-game automation, matchmaking bots, memory reading,
process injection or anti-cheat evasion.

## Known limitations

* One SQLite connection per `DatabaseManager`, intended for single-process,
  single-thread use (SQLite's default thread check is kept). Concurrent
  processes are serialised by SQLite locking (`BEGIN IMMEDIATE`, 5 s timeout).
* Spawn-point identity is the exact `(map_id, x, y, z)`. Positions that differ
  by a tiny float amount count as different spawn points.
* Session duration uses the injected clock (wall clock by default). If the
  clock goes backwards before a session ends, the duration is recorded as 0
  and a warning is logged.
* There is no export/import or backup command yet. Copy the `.db` file while
  no command is running.
