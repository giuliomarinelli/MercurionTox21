import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from bootstrap_secrets import run_secret_file_bootstrap
from secret_bootstrap import (
    BOOTSTRAP_FILE_VARIABLE, MAXIMUM_BOOTSTRAP_BYTES, SecretBootstrapError,
    load_secret_file_environment, parse_secret_environment,
)

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'synthetic-$PATH-${HOME}-password'


class SecretBootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='tox21-secret-bootstrap-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def payload(self, variables=None, **changes):
        document = {'schemaVersion': 1, 'environment': 'staging',
                    'variables': {'NATS_USERNAME': 'synthetic-worker', 'NATS_PASSWORD': MARKER}
                    if variables is None else variables}
        document.update(changes)
        return json.dumps(document).encode()

    def write(self, data=None):
        path = self.root / 'credentials.json'
        path.write_bytes(self.payload() if data is None else data)
        path.chmod(0o600)
        return path

    def environment(self, path=None):
        return {'PY_ENV': 'staging', 'NATS_URL': 'nats://broker:4222',
                BOOTSTRAP_FILE_VARIABLE: str(self.write() if path is None else path)}

    def test_loads_literal_credentials_without_changing_metadata_or_deleting_file(self):
        environment = self.environment()
        load_secret_file_environment(environment)
        self.assertEqual(environment['NATS_PASSWORD'], MARKER)
        self.assertEqual(environment['NATS_USERNAME'], 'synthetic-worker')
        self.assertEqual(environment['PY_ENV'], 'staging')
        self.assertEqual(environment['NATS_URL'], 'nats://broker:4222')
        self.assertTrue(Path(environment[BOOTSTRAP_FILE_VARIABLE]).exists())

    def test_production_and_utf8_bom_are_supported(self):
        result = parse_secret_environment(b'\xef\xbb\xbf' + self.payload(environment='production'), {'PY_ENV': 'production'})
        self.assertEqual(result['NATS_PASSWORD'], MARKER)

    def test_environment_mismatch_and_implicit_development_are_rejected(self):
        for value in (None, '', 'test', 'development', 'production'):
            with self.subTest(environment=value), self.assertRaisesRegex(SecretBootstrapError, 'environment-mismatch'):
                parse_secret_environment(self.payload(), {} if value is None else {'PY_ENV': value})

    def test_only_supported_versions_and_exact_top_level_fields_are_accepted(self):
        for value in (True, '1', 1.0, 2, None):
            with self.subTest(version=value), self.assertRaisesRegex(SecretBootstrapError, 'invalid-document'):
                parse_secret_environment(self.payload(schemaVersion=value), {'PY_ENV': 'staging'})
        with self.assertRaisesRegex(SecretBootstrapError, 'invalid-document'):
            parse_secret_environment(self.payload(extra=MARKER), {'PY_ENV': 'staging'})

    def test_malformed_encoding_non_json_constants_and_duplicate_keys_fail(self):
        for data in (b'', b'\xff', b'[]', b'{bad', b'NaN', b'{"environment":1,"environment":2}',
                     self.payload().replace(b'"schemaVersion": 1', b'"schemaVersion": NaN'),
                     self.payload().replace(b'"NATS_PASSWORD":', b'"NATS_PASSWORD":"duplicate", "NATS_PASSWORD":'),
                     b'[' * 2000 + b']' * 2000):
            with self.subTest(size=len(data)), self.assertRaises(SecretBootstrapError):
                parse_secret_environment(data, {'PY_ENV': 'staging'})

    def test_unknown_and_control_variables_never_appear_in_errors(self):
        for name in ('PY_ENV', 'PYTHONPATH', 'PATH', 'AUTH_INFERENCE', 'NATS_URL', 'JWT_PUBLIC_KEY_PATH', MARKER):
            with self.subTest(name=name), self.assertRaisesRegex(SecretBootstrapError, 'unknown-variable') as caught:
                parse_secret_environment(self.payload({name: MARKER}), {'PY_ENV': 'staging'})
            self.assertNotIn(name, str(caught.exception))

    def test_invalid_values_and_incomplete_pairs_are_rejected(self):
        for value in ('', ' ', None, False, 1, 'with\0nul', '\ud800'):
            with self.subTest(kind=type(value).__name__), self.assertRaisesRegex(SecretBootstrapError, 'invalid-value'):
                parse_secret_environment(self.payload({'NATS_USERNAME': 'worker', 'NATS_PASSWORD': value}), {'PY_ENV': 'staging'})
        for values in ({}, {'NATS_USERNAME': 'worker'}, {'NATS_PASSWORD': MARKER}):
            with self.subTest(fields=list(values)), self.assertRaises(SecretBootstrapError):
                parse_secret_environment(self.payload(values), {'PY_ENV': 'staging'})

    def test_existing_values_are_rejected_even_if_identical(self):
        for name, value in (('NATS_USERNAME', 'synthetic-worker'), ('NATS_PASSWORD', MARKER)):
            with self.subTest(field=name), self.assertRaisesRegex(SecretBootstrapError, 'environment-conflict'):
                parse_secret_environment(self.payload(), {'PY_ENV': 'staging', name: value})

    def test_late_invalid_entry_does_not_inject_partial_credentials(self):
        path = self.write(self.payload({'NATS_USERNAME': 'worker', 'NATS_PASSWORD': MARKER, 'PYTHONPATH': MARKER}))
        environment = self.environment(path)
        before = dict(environment)
        with self.assertRaises(SecretBootstrapError):
            load_secret_file_environment(environment)
        self.assertEqual(environment, before)

    def test_missing_relative_non_regular_and_unreadable_paths_are_sanitized(self):
        for path in ('relative.json', str(self.root / MARKER), str(self.root), str(self.root / 'with\0nul')):
            with self.subTest(kind=path), self.assertRaises(SecretBootstrapError) as caught:
                load_secret_file_environment(self.environment(path))
            self.assertNotIn(MARKER, str(caught.exception))
            self.assertNotIn(str(self.root), str(caught.exception))
        with self.assertRaisesRegex(SecretBootstrapError, 'missing-file'):
            load_secret_file_environment({'PY_ENV': 'staging'})
        environment = self.environment()
        with patch('secret_bootstrap.os.open', side_effect=PermissionError(MARKER)), \
                self.assertRaisesRegex(SecretBootstrapError, 'file-unreadable') as caught:
            load_secret_file_environment(environment)
        self.assertNotIn(MARKER, str(caught.exception))
        self.assertNotIn('NATS_PASSWORD', environment)
        with patch('secret_bootstrap.os.read', side_effect=OSError(MARKER)), \
                patch('secret_bootstrap.os.close', wraps=os.close) as close, \
                self.assertRaisesRegex(SecretBootstrapError, 'file-unreadable'):
            load_secret_file_environment(environment)
        close.assert_called_once()

    def test_oversized_files_and_growth_after_stat_are_bounded(self):
        for data in (b'', b'x' * (MAXIMUM_BOOTSTRAP_BYTES + 1)):
            with self.subTest(size=len(data)), self.assertRaisesRegex(SecretBootstrapError, 'file-size'):
                load_secret_file_environment(self.environment(self.write(data)))
        environment = self.environment()
        # Simulate growth with real descriptor open/close, bounded fake reads.
        with patch('secret_bootstrap.os.read', side_effect=lambda _fd, count: b'x' * count) as read, \
                self.assertRaisesRegex(SecretBootstrapError, 'file-size'):
            load_secret_file_environment(environment)
        self.assertEqual(sum(call.args[1] for call in read.call_args_list), MAXIMUM_BOOTSTRAP_BYTES + 1)

    @unittest.skipIf(os.name == 'nt', 'POSIX modes and no-follow checked on Linux')
    def test_posix_modes_symlinks_and_actual_read_permissions(self):
        path = self.write()
        for mode in (0o600, 0o400, 0o640):
            path.chmod(mode)
            load_secret_file_environment(self.environment(path))
        for mode in (0o644, 0o660):
            path.chmod(mode)
            with self.assertRaisesRegex(SecretBootstrapError, 'file-permissions'):
                load_secret_file_environment(self.environment(path))
        path.chmod(0o600)
        link = self.root / 'symlink'
        link.symlink_to(path)
        with self.assertRaisesRegex(SecretBootstrapError, 'file-type'):
            load_secret_file_environment(self.environment(link))
        if os.geteuid() != 0:
            path.chmod(0)
            try:
                with self.assertRaisesRegex(SecretBootstrapError, 'file-unreadable'):
                    load_secret_file_environment(self.environment(path))
            finally:
                path.chmod(0o600)

    def test_launcher_injects_before_start_and_sanitizes_all_failures(self):
        environment = self.environment()
        report = Mock()
        start = Mock(side_effect=lambda: self.assertEqual(environment['NATS_PASSWORD'], MARKER))
        self.assertEqual(run_secret_file_bootstrap(start, environment, report), 0)
        start.assert_called_once()
        report.assert_not_called()
        start.reset_mock()
        self.assertEqual(run_secret_file_bootstrap(start, {'PY_ENV': 'staging'}, report), 1)
        start.assert_not_called()
        self.assertEqual(report.call_args.args[0], {'code': 'INVALID_SECRET_BOOTSTRAP', 'reason': 'missing-file'})
        self.assertEqual(run_secret_file_bootstrap(Mock(side_effect=RuntimeError(MARKER)), self.environment(), report), 1)
        self.assertEqual(report.call_args.args[0], {'code': 'APPLICATION_START_FAILED', 'reason': 'application-startup'})

    @unittest.skipUnless(hasattr(os, 'O_NOFOLLOW'), 'Leaf replacement race checked on Linux')
    def test_symlink_replacement_between_lstat_and_open_is_rejected(self):
        path = self.write()
        target = self.root / 'other.json'
        target.write_bytes(self.payload())
        target.chmod(0o600)
        original_open = os.open

        def replace_then_open(name, flags):
            path.unlink()
            path.symlink_to(target)
            return original_open(name, flags)

        with patch('secret_bootstrap.os.open', side_effect=replace_then_open), \
                self.assertRaisesRegex(SecretBootstrapError, 'file-unreadable'):
            load_secret_file_environment(self.environment(path))

    def test_real_cli_defers_import_and_keeps_credentials_out_of_diagnostics(self):
        for source in ('secret_bootstrap.py', 'bootstrap_secrets.py'):
            shutil.copyfile(ROOT / source, self.root / source)
        (self.root / 'main.py').write_text(
            "import os\nfrom pathlib import Path\nPath('main-imported').touch()\n"
            "async def main():\n assert os.environ['NATS_PASSWORD'] == " + repr(MARKER) + "\n print('FAKE_APP_STARTED')\n",
            encoding='utf-8')
        system = {'SystemRoot': os.environ['SystemRoot']} if 'SystemRoot' in os.environ else {}
        invalid = subprocess.run([sys.executable, '-B', '-m', 'bootstrap_secrets'], cwd=self.root,
                                 env={**system, 'PY_ENV': 'staging'}, capture_output=True, text=True, timeout=10)
        self.assertEqual(invalid.returncode, 1)
        self.assertIn('INVALID_SECRET_BOOTSTRAP', invalid.stderr)
        self.assertFalse((self.root / 'main-imported').exists())
        valid = subprocess.run([sys.executable, '-B', '-m', 'bootstrap_secrets'], cwd=self.root,
                               env={**system, **self.environment()}, capture_output=True, text=True, timeout=10)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertEqual(valid.stdout.strip(), 'FAKE_APP_STARTED')
        self.assertNotIn(MARKER, valid.stdout + valid.stderr)


if __name__ == '__main__':
    unittest.main()
