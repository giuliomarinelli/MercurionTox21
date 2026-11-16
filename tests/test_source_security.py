import json
from pathlib import Path
import tempfile
import unittest
from scripts.check_source_security import SCOPES, evaluate, prepare
from tests.test_container_evidence import POLICY


def clean():
    return {'SchemaVersion': 2, 'Results': [{'Target': scope + '/requirements.txt'} for scope in SCOPES]}


class SourceSecurityTests(unittest.TestCase):
    def test_every_declared_scope_is_required_even_when_no_vulnerability_is_reported(self):
        self.assertTrue(evaluate(clean(), POLICY)['eligible'])
        for scope in SCOPES:
            report = clean()
            report['Results'] = [entry for entry in report['Results'] if not entry['Target'].startswith(scope + '/')]
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                evaluate(report, POLICY)

    def test_low_finding_in_training_or_build_blocks(self):
        for entry in clean()['Results']:
            report = clean()
            report['Results'].append({**entry, 'Vulnerabilities': [
                {'Severity': 'LOW', 'VulnerabilityID': 'CVE-fixture', 'PkgName': 'fixture'}]})
            self.assertFalse(evaluate(report, POLICY)['eligible'])

    def test_scanner_error_cannot_approve_partial_evidence(self):
        for key in ('Error', 'Errors'):
            with self.assertRaises(ValueError):
                evaluate({**clean(), key: ['scanner unavailable']}, POLICY)

    def test_prepare_copies_only_known_manifests_without_rewriting_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'source'
            for path in SCOPES.values():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'fixture==1.0 --hash=sha256:synthetic\n')
            destination = Path(temporary) / 'scan'
            prepare(destination, root)
            self.assertEqual({str(path.relative_to(destination)).replace('\\', '/')
                for path in destination.rglob('*') if path.is_file()},
                {scope + '/requirements.txt' for scope in SCOPES})
            for scope, path in SCOPES.items():
                self.assertEqual((destination / scope / 'requirements.txt').read_bytes(), (root / path).read_bytes())
            with self.assertRaises(FileExistsError):
                prepare(destination, root)
