"""Prepare known dependency manifests and enforce strict, complete Trivy source evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.container_evidence import evaluate_security_report

ROOT = Path(__file__).resolve().parents[1]
SCOPES = {'legacy': 'requirements.txt', 'runtime': 'requirements/runtime-linux-cpu.lock',
          'training': 'requirements/training-linux-cpu.lock', 'build': 'requirements/build-linux-cpu.lock'}


def prepare(destination, root=ROOT):
    destination.mkdir(parents=True, exist_ok=False)
    for scope, source in SCOPES.items():
        folder = destination / scope
        folder.mkdir()
        (folder / 'requirements.txt').write_bytes((root / source).read_bytes())


def evaluate(report, policy):
    result = evaluate_security_report(report, policy)
    targets = {entry['Target'].replace('\\', '/') for entry in report['Results']}
    for scope in SCOPES:
        expected = scope + '/requirements.txt'
        if not any(target == expected or target.endswith('/' + expected) for target in targets):
            raise ValueError('Incomplete dependency scan: ' + scope)
    return {**result, 'scopes': list(SCOPES)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.prepare is not None and args.report is None and args.output is None:
        prepare(args.prepare)
        print(json.dumps({'preparedScopes': list(SCOPES)}))
        return 0
    if args.prepare is not None or args.report is None or args.output is None:
        parser.error('Choose --prepare DIRECTORY or --report JSON --output JSON')
    policy = json.loads((ROOT / 'docs/container-policy.json').read_bytes())
    result = evaluate(json.loads(args.report.read_bytes()), policy)
    result.update(policySha256=hashlib.sha256((ROOT / 'docs/container-policy.json').read_bytes()).hexdigest(),
                  manifests={scope: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for scope, path in SCOPES.items()})
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'eligible': result['eligible'], 'counts': result['counts'], 'scopes': result['scopes']}))
    return 0 if result['eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
