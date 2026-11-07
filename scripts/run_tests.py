"""Explicit unittest groups; default application tests need no training data or broker."""
import argparse
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUPS = {
    "application": ["tests.test_config", "tests.test_runtime", "tests.test_nats_requests",
                    "tests.test_rdkit_api", "tests.test_pcp", "tests.test_inference", "tests.test_model_package"],
    "model": ["tests.test_model_export"],
    "training": ["tests.test_training_pipeline", "tests.test_checkpoint_training"],
}


def cases(suite):
    for entry in suite:
        if isinstance(entry, unittest.TestSuite):
            yield from cases(entry)
        else:
            yield entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="?", choices=[*GROUPS, "all", "reference"], default="application")
    parser.add_argument("--reverse", action="store_true", help="Check order independence in the same selected group")
    args = parser.parse_args()
    if args.group == "reference":
        modules = ["tests.reference.model_reference"]
    else:
        modules = [module for group in GROUPS.values() for module in group] if args.group == "all" else GROUPS[args.group]
    print(f"unittest group: {args.group}; modules: {len(modules)}", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    if args.reverse:
        suite = unittest.TestSuite(reversed(list(cases(suite))))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
