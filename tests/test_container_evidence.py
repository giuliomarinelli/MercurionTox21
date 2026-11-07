import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from scripts.container_evidence import archive_config, security_summary

POLICY = {'schema_version': 1, 'blocking_severities': ['CRITICAL'], 'exceptions': []}
DIGEST = 'sha256:' + 'a' * 64


class ContainerEvidenceTests(unittest.TestCase):
    def test_critical_findings_block_without_hiding_other_severities(self):
        findings = [{'Severity': severity, 'VulnerabilityID': 'CVE-fixture', 'PkgName': 'fixture'}
                    for severity in ['CRITICAL', 'HIGH']]
        report = {'SchemaVersion': 2, 'Metadata': {'ImageID': DIGEST}, 'Results': [{'Vulnerabilities': findings}]}
        result = security_summary(report, DIGEST, POLICY)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['counts'], {'CRITICAL': 1, 'HIGH': 1})

    def test_empty_valid_scan_is_eligible(self):
        report = {'SchemaVersion': 2, 'Metadata': {'ImageID': DIGEST}, 'Results': [{'Target': 'fixture'}]}
        self.assertTrue(security_summary(report, DIGEST, POLICY)['eligible'])

    def test_missing_or_wrong_image_scan_and_policy_override_fail_closed(self):
        report = {'SchemaVersion': 2, 'Metadata': {'ImageID': DIGEST}, 'Results': [{'Target': 'fixture'}]}
        for altered, policy in [({}, POLICY), ({**report, 'Metadata': {}}, POLICY),
                                (report, {**POLICY, 'blocking_severities': []}),
                                (report, {**POLICY, 'exceptions': ['unapproved']})]:
            with self.assertRaises(ValueError):
                security_summary(altered, DIGEST, policy)

    def test_archive_config_is_hashed_without_extraction(self):
        self.check_archive([])

    def test_tagged_archive_is_rejected_before_import(self):
        with self.assertRaises(ValueError):
            self.check_archive(['user/existing:tag'])

    def check_archive(self, tags):
        raw = json.dumps({'config': {'Labels': {}}, 'rootfs': {'diff_ids': []}}).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.tar'
            with tarfile.open(path, 'w') as archive:
                for name, data in [('manifest.json', json.dumps([{'Config': 'config.json', 'RepoTags': tags}]).encode()), ('config.json', raw)]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            digest, _ = archive_config(path)
            self.assertEqual(digest, 'sha256:' + hashlib.sha256(raw).hexdigest())
            self.assertEqual({entry.name for entry in Path(directory).iterdir()}, {'image.tar'})
