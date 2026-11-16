from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from config import ConfigurationError, get_config, nats_subject
from schemas.schemas import Configuration
from pydantic import ValidationError


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.values = {"NATS_URL": "nats://broker.invalid:4222", "VERSION": "test-version"}

    def config(self, values):
        return get_config(values, root=self.root)

    def test_all_environments_and_default_public_pem_paths(self):
        for env in ("development", "test", "staging", "production"):
            with self.subTest(env=env):
                result = self.config({**self.values, "PY_ENV": env})
                self.assertEqual(result.py_env, env)
                self.assertEqual(result.jwt_public_key_path,
                                 "public.pem" if env == "production" else f"public.{env}.pem")
        self.assertEqual(self.config(self.values).py_env, "development")

    def test_invalid_environment_never_reads_a_fallback_file(self):
        for env in ("", "Production", "../development", "unknown", " staging "):
            with self.subTest(env=env), patch("config.dotenv_values") as read:
                with self.assertRaises(ConfigurationError):
                    self.config({**self.values, "PY_ENV": env})
                read.assert_not_called()

    def test_local_files_and_process_precedence_do_not_mutate_environment(self):
        for env in ("development", "test"):
            (self.root / f".env.{env}").write_text(
                "NATS_URL=nats://file.invalid:4222\nVERSION=file-version\n", encoding="utf-8"
            )
            with self.subTest(env=env):
                before = dict(os.environ)
                result = self.config({"PY_ENV": env, "VERSION": "process-version"})
                self.assertEqual(result.nats_url, "nats://file.invalid:4222")
                self.assertEqual(result.version, "process-version")
                self.assertEqual(dict(os.environ), before)

    def test_staging_and_production_never_read_dotenv(self):
        for env in ("staging", "production"):
            with self.subTest(env=env), patch("config.dotenv_values", side_effect=AssertionError):
                self.assertEqual(self.config({**self.values, "PY_ENV": env}).py_env, env)
                with self.assertRaises(ConfigurationError):
                    self.config({"PY_ENV": env})

    def test_required_values_reject_missing_blank_and_non_string(self):
        for key in ("NATS_URL", "VERSION"):
            for value in (None, "", "   ", 123):
                with self.subTest(key=key, value=value), self.assertRaises(ConfigurationError):
                    self.config({**self.values, key: value})

    def test_explicit_pem_path_is_preserved_without_reading_it(self):
        for env in ("staging", "production"):
            result = self.config({**self.values, "PY_ENV": env, "JWT_PUBLIC_KEY_PATH": "/keys/access.pem"})
            self.assertEqual(result.jwt_public_key_path, "/keys/access.pem")

    def test_broker_url_validation_and_sanitized_diagnostics(self):
        for url in ("https://broker.invalid", "nats://", "nats://host:0", "nats://host:99999",
                    "nats://host/path", "nats://user:private-test-value@host:bad",
                    "nats://user:private-test-value@host:4222", "nats://user@host:4222", "nats://@host:4222"):
            with self.subTest(url=url):
                with self.assertRaises(ConfigurationError) as caught:
                    self.config({**self.values, "NATS_URL": url})
                self.assertNotIn(url, str(caught.exception))
                self.assertNotIn("private-test-value", str(caught.exception))
        self.assertEqual(self.config({**self.values, "NATS_URL": "tls://broker.invalid:4222"}).nats_url,
                         "tls://broker.invalid:4222")

    def test_nats_credentials_are_optional_and_literal_in_every_environment(self):
        for env in ("development", "test", "staging", "production"):
            with self.subTest(env=env):
                base = {**self.values, "PY_ENV": env}
                self.assertEqual(self.config(base).nats_connect_credentials(), {})
                result = self.config({**base, "NATS_USERNAME": "synthetic-tox21",
                                      "NATS_PASSWORD": "synthetic-@$:{HOME} password "})
                self.assertEqual(result.nats_connect_credentials(), {
                    "user": "synthetic-tox21", "password": "synthetic-@$:{HOME} password "})
                self.assertEqual(result.nats_url, self.values["NATS_URL"])
                self.assertNotIn("synthetic-@$", repr(result))
                self.assertNotIn("synthetic-@$", result.model_dump_json())

    def test_partial_blank_null_and_wrong_type_nats_credentials_fail_without_values(self):
        base = {**self.values, "PY_ENV": "staging"}
        for credentials in ({"NATS_USERNAME": "synthetic-private-user"},
                            {"NATS_PASSWORD": "synthetic-private-password"},
                            {"NATS_USERNAME": "", "NATS_PASSWORD": "synthetic-private-password"},
                            {"NATS_USERNAME": "synthetic-private-user", "NATS_PASSWORD": ""},
                            {"NATS_USERNAME": None}, {"NATS_PASSWORD": None},
                            {"NATS_USERNAME": 12, "NATS_PASSWORD": "synthetic-private-password"},
                            {"NATS_USERNAME": "synthetic-private-user", "NATS_PASSWORD": False}):
            with self.subTest(fields=sorted(credentials)), self.assertRaises(ConfigurationError) as caught:
                self.config({**base, **credentials})
            self.assertNotIn("synthetic-private", str(caught.exception))
            self.assertIn("nats_", str(caught.exception))

    def test_direct_pydantic_errors_hide_input_and_partial_mutation_never_becomes_anonymous(self):
        base = {"py_env": "test", "nats_url": self.values["NATS_URL"],
                "version": "qa", "jwt_public_key_path": "not-read.pem"}
        with self.assertRaises(ValidationError) as caught:
            Configuration(**base, nats_password="synthetic-private-password")
        self.assertNotIn("synthetic-private-password", str(caught.exception))
        settings = Configuration(**base)
        settings.nats_username = "synthetic-user"
        with self.assertRaises(ValueError):
            settings.nats_connect_credentials()

    def test_local_nats_credentials_follow_process_precedence_without_interpolation(self):
        (self.root / ".env.test").write_text(
            "NATS_URL=nats://broker.invalid:4222\nVERSION=qa\n"
            "NATS_USERNAME=synthetic-file-user\nNATS_PASSWORD=literal-${HOME}\n", encoding="utf-8")
        result = self.config({"PY_ENV": "test"})
        self.assertEqual(result.nats_connect_credentials(), {"user": "synthetic-file-user", "password": "literal-${HOME}"})
        result = self.config({"PY_ENV": "test", "NATS_PASSWORD": "synthetic-process-password"})
        self.assertEqual(result.nats_connect_credentials()["password"], "synthetic-process-password")
        (self.root / ".env.test").write_text("NATS_USERNAME\nNATS_PASSWORD\n", encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            self.config({**self.values, "PY_ENV": "test"})

    def test_subject_matrix_matches_web_contract(self):
        subjects = ("inference.tox21.smiles", "rdkit_api.get_molecule_properties",
                    "rdkit_api.to_canonical_smiles", "rdkit_api.are_same_structure",
                    "pcp_api.get_iupac_name_from_smiles")
        for env in ("development", "test", "staging", "production"):
            for base in subjects:
                with self.subTest(env=env, base=base):
                    self.assertEqual(nats_subject(base, env), base if env == "production" else f"{env}.{base}")
        with self.assertRaises(ConfigurationError):
            nats_subject(subjects[0], "unknown")

    def test_model_package_selection_is_explicit_and_nonblank(self):
        self.assertIsNone(self.config(self.values).model_package_dir)
        self.assertEqual(self.config({**self.values, "MODEL_PACKAGE_DIR": "/models/release"}).model_package_dir,
                         "/models/release")
        with self.assertRaises(ConfigurationError):
            self.config({**self.values, "MODEL_PACKAGE_DIR": " "})


if __name__ == "__main__":
    unittest.main()
