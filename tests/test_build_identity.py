import json
from pathlib import Path
import tempfile
import unittest
from runtime import read_build_identity
from config import ConfigurationError


class BuildIdentityTests(unittest.TestCase):
    def test_source_checkout_has_no_invented_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(read_build_identity(Path(directory) / 'absent.json'))

    def test_valid_image_identity_is_returned(self):
        value = {'version': 'qa', 'revision': 'a' * 40,
                 'source': 'https://github.com/giuliomarinelli/MercurionTox21'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.json'
            path.write_text(json.dumps(value))
            self.assertEqual(read_build_identity(path), value)

    def test_malformed_or_incomplete_identity_fails_closed(self):
        values = ['not-json', {}, [], {'version': 'qa', 'revision': 'short', 'source': 'wrong'},
                  {'version': 'qa', 'revision': None, 'source': 'wrong'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.json'
            for value in values:
                with self.subTest(value=value):
                    path.write_text(value if isinstance(value, str) else json.dumps(value))
                    with self.assertRaises(ConfigurationError):
                        read_build_identity(path)
