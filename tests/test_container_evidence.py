import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from scripts.container_evidence import archive_config, security_summary

POLICY = {'schema_version': 2, 'blocking_severities': ['UNKNOWN', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
          'dependency_scopes': ['legacy', 'runtime', 'training', 'build'], 'exceptions': []}
DIGEST = 'sha256:' + 'a' * 64


def image_report(findings=None):
    return {'SchemaVersion': 2, 'Metadata': {'ImageID': DIGEST, 'OS': {'Family': 'wolfi'}},
            'Results': [{'Target': 'fixture', 'Class': 'os-pkgs', 'Type': 'wolfi',
                         'Vulnerabilities': findings or []}]}


class ContainerEvidenceTests(unittest.TestCase):
    def test_critical_findings_block_without_hiding_other_severities(self):
        findings = [{'Severity': severity, 'VulnerabilityID': 'CVE-fixture', 'PkgName': 'fixture'}
                    for severity in ['CRITICAL', 'HIGH']]
        report = image_report(findings)
        result = security_summary(report, DIGEST, POLICY)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['counts'], {'CRITICAL': 1, 'HIGH': 1})
        self.assertEqual(result['blocking'], findings)

    def test_each_severity_blocks_even_without_a_cve_identifier(self):
        for severity in POLICY['blocking_severities']:
            report = image_report([{'Severity': severity,
                                   'VulnerabilityID': 'GHSA-fixture', 'PkgName': 'fixture'}])
            with self.subTest(severity=severity):
                self.assertFalse(security_summary(report, DIGEST, POLICY)['eligible'])

    def test_empty_valid_scan_is_eligible(self):
        report = image_report()
        self.assertTrue(security_summary(report, DIGEST, POLICY)['eligible'])

    def test_missing_or_wrong_image_scan_and_policy_override_fail_closed(self):
        report = image_report()
        for altered, policy in [({}, POLICY), ({**report, 'Metadata': {}}, POLICY),
                                (report, {**POLICY, 'blocking_severities': []}),
                                (report, {**POLICY, 'blocking_severities': ['CRITICAL']}),
                                (report, {**POLICY, 'exceptions': ['unapproved']})]:
            with self.assertRaises(ValueError):
                security_summary(altered, DIGEST, policy)

    def test_missing_target_unknown_severity_and_partial_scans_fail_closed(self):
        for results in ([], [{}], [{'Target': 'fixture', 'Vulnerabilities': None}],
                        [{'Target': 'fixture', 'Vulnerabilities': [{'Severity': 'NEW',
                          'VulnerabilityID': 'CVE-fixture', 'PkgName': 'fixture'}]}]):
            results = [{**entry, 'Class': 'os-pkgs', 'Type': 'wolfi'} for entry in results]
            with self.assertRaises(ValueError):
                security_summary({**image_report(), 'Results': results}, DIGEST, POLICY)

    def test_unrecognized_os_or_language_only_scan_cannot_approve_image(self):
        report = image_report()
        for altered in [
            {**report, 'Metadata': {'ImageID': DIGEST}},
            {**report, 'Metadata': {'ImageID': DIGEST, 'OS': {'Family': 'unknown'}}},
            {**report, 'Results': [{'Target': 'Python', 'Class': 'lang-pkgs', 'Type': 'python-pkg'}]},
        ]:
            with self.assertRaises(ValueError):
                security_summary(altered, DIGEST, POLICY)

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
