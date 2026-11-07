"""Explicit unittest groups; default application tests need no training data or broker."""
import argparse
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUPS = {
    "application": ["tests.test_service_auth", "tests.test_container_evidence", "tests.test_build_identity", "tests.test_config", "tests.test_runtime", "tests.test_nats_requests",
                    "tests.test_rdkit_api", "tests.test_pcp", "tests.test_inference", "tests.test_model_package"],
    "model": ["tests.test_model_export"],
    "training": ["tests.test_training_pipeline", "tests.test_training_contracts", "tests.test_checkpoint_training"],
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
    parser.add_argument("--report-json", type=Path, help="Optional CI result, including failing test diagnostics")
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
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps({
            'group': args.group, 'tests': result.testsRun, 'success': result.wasSuccessful(),
            'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures],
            'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors],
            'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
        }, indent=2) + '\n', encoding='utf-8')
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
