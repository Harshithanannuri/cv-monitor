"""Run every test without needing pytest:   python tests/run_all.py"""
import importlib
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

failed = 0
for mod in ("test_units", "test_features"):
    m = importlib.import_module(mod)
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        t0 = time.time()
        try:
            getattr(m, name)()
            print(f"PASS {mod}.{name}  ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {mod}.{name}")
            traceback.print_exc()

import test_gui_api   # noqa: E402
print("\n--- GUI API")
failed += test_gui_api.main() != 0
print("\nALL TESTS PASSED" if not failed else f"\n{failed} FAILED")
sys.exit(1 if failed else 0)
