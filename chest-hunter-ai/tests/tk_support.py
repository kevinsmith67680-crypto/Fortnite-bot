"""Free Tk objects on the main thread.

Tk widgets, variables and images sit in reference cycles, so Python's cyclic garbage collector
frees them, in whichever thread happens to trigger a collection. Some tests start threads (the
mock Unreal server). When a Tk object left over from a GUI test is freed in one of those
threads, its finaliser fails with "main thread is not in main loop", and freeing the Tcl
interpreter there aborts the whole test process ("Tcl_AsyncDelete: async handler deleted by the
wrong thread").

So modules that open Tk windows collect their garbage on the main thread when they finish
(``tearDownModule``), and a test that runs Tk while other threads run keeps automatic
collection off until it has collected on the main thread (:func:`collect_tk_garbage_after`).
"""

from __future__ import annotations

import gc
import unittest


def collect_on_main_thread() -> None:
    """Free unreachable Tk objects now, on the calling thread (the main thread in tests)."""
    gc.collect()


def collect_tk_garbage_after(test: unittest.TestCase) -> None:
    """For a test that runs Tk next to other threads: collect now, switch automatic collection
    off (in every thread) for the test, and after it collect on this thread and switch it back
    on. Register this before any other cleanup, so it runs last."""
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()

    def restore() -> None:
        gc.collect()
        if was_enabled:
            gc.enable()

    test.addCleanup(restore)
