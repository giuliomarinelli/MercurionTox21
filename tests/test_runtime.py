import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

import main as application
from config import ConfigurationError
from runtime import load_public_key


ROOT = Path(__file__).resolve().parents[1]


class PemAndImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.pem = cls.key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

    def test_public_mode_does_not_read_any_key(self):
        with patch("pathlib.Path.read_text", side_effect=AssertionError("No key I/O")):
            self.assertIsNone(load_public_key("missing.pem", skip_auth=True))

    def test_rsa_public_pem_is_loaded_as_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public.pem"
            path.write_text(self.pem, encoding="utf-8")
            self.assertEqual(load_public_key(str(path), skip_auth=False), self.pem)

    def test_missing_invalid_private_non_rsa_and_non_pem_inputs_fail_closed(self):
        private = self.key.private_bytes(serialization.Encoding.PEM,
                                        serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        non_rsa = ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        with tempfile.TemporaryDirectory() as directory:
            for name, value in (("bad.pem", b"not-a-public-key"), ("private.pem", private),
                                ("ec.pem", non_rsa), ("base64.txt", self.pem.encode())):
                path = Path(directory) / name
                path.write_bytes(value)
                with self.subTest(name=name), self.assertRaises(ConfigurationError) as caught:
                    load_public_key(str(path), skip_auth=False)
                self.assertNotIn("not-a-public-key", str(caught.exception))
            with self.assertRaises(ConfigurationError):
                load_public_key(str(Path(directory) / "missing.pem"), skip_auth=False)

    def test_real_jwt_signature_expiry_audience_and_algorithm(self):
        now = datetime.now(timezone.utc)
        claims = {"sub": "synthetic-user", "aud": "mercurion-api", "exp": now + timedelta(minutes=2)}
        wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        tokens = (
            jwt.encode({**claims, "exp": now - timedelta(minutes=2)}, self.key, algorithm="RS256"),
            jwt.encode({**claims, "aud": "wrong-audience"}, self.key, algorithm="RS256"),
            jwt.encode(claims, wrong_key, algorithm="RS256"),
            jwt.encode(claims, "synthetic-hmac-secret-long-enough-for-tests-only", algorithm="HS256"),
        )
        with patch.object(application, "PUBLIC_KEY", self.pem):
            valid = jwt.encode(claims, self.key, algorithm="RS256")
            self.assertEqual(application.verify_jwt(valid)["sub"], "synthetic-user")
            for token in tokens:
                with self.subTest(token_kind=jwt.get_unverified_header(token)["alg"]):
                    self.assertIsNone(application.verify_jwt(token))

    def test_main_import_has_no_config_model_key_or_connection_side_effects(self):
        with patch("config.get_config", side_effect=AssertionError("No configuration at import")), \
             patch("runtime.load_inference_runtime", side_effect=AssertionError("No model at import")), \
             patch("runtime.load_public_key", side_effect=AssertionError("No key at import")), \
             patch("nats.aio.client.Client.connect", side_effect=AssertionError("No connection at import")):
            module = runpy.run_path(str(ROOT / "main.py"), run_name="import_test")
        self.assertIsNone(module["PUBLIC_KEY"])
        self.assertIs(module["GLOBAL_SKIP_AUTH_FLAG"], True)

    def test_inference_import_does_not_read_threshold_files(self):
        with patch.dict("sys.modules", {"torch": Mock()}), \
             patch("builtins.open", side_effect=AssertionError("No threshold files at import")):
            module = runpy.run_path(str(ROOT / "api" / "inference.py"), run_name="inference_import_test")
        self.assertTrue(callable(module["predict"]))


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stop = asyncio.Event()
        self.ready = asyncio.Event()
        self.callbacks = {}
        self.subjects = []
        self.tasks = []

        async def connect(url, **callbacks):
            self.callbacks.update(callbacks)

        async def subscribe(subject, cb):
            self.subjects.append(subject)

        self.client = SimpleNamespace(
            connect=AsyncMock(side_effect=connect), subscribe=AsyncMock(side_effect=subscribe),
            flush=AsyncMock(), drain=AsyncMock(), close=AsyncMock(), is_connected=True, is_closed=False,
        )
        self.factory = Mock(return_value=self.client)
        self.loader = Mock(return_value=(Mock(), "cpu", Mock(return_value={})))
        self.settings = SimpleNamespace(py_env="test", nats_url="nats://test.invalid:4222",
                                        version="synthetic", jwt_public_key_path="not-used.pem")
        patcher = patch.multiple(application, NATS=self.factory, get_config=Mock(return_value=self.settings),
                                 load_inference_runtime=self.loader, GLOBAL_SKIP_AUTH_FLAG=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        printer = patch("builtins.print")
        printer.start()
        self.addCleanup(printer.stop)

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def start(self):
        task = asyncio.create_task(application.run(stop_event=self.stop, ready_event=self.ready))
        self.tasks.append(task)
        await asyncio.wait_for(self.ready.wait(), timeout=1)
        return task

    async def test_stop_drains_closes_and_clears_readiness(self):
        task = await self.start()
        self.assertEqual(len(self.subjects), 5)
        self.assertTrue(all(subject.startswith("test.") for subject in self.subjects))
        self.client.flush.assert_awaited_once_with(timeout=5)
        self.stop.set()
        await task
        self.client.drain.assert_awaited_once()
        self.client.close.assert_awaited_once()
        self.assertFalse(self.ready.is_set())

    async def test_model_failure_happens_before_broker_connection(self):
        self.loader.side_effect = ValueError("synthetic model failure")
        self.ready.set()
        with self.assertRaises(ValueError):
            await application.run(ready_event=self.ready)
        self.factory.assert_not_called()
        self.assertFalse(self.ready.is_set())

    async def test_invalid_configuration_clears_readiness_before_any_resources(self):
        self.ready.set()
        with patch.object(application, "get_config", side_effect=ConfigurationError("synthetic invalid settings")):
            with self.assertRaises(ConfigurationError):
                await application.run(ready_event=self.ready)
        self.loader.assert_not_called()
        self.factory.assert_not_called()
        self.assertFalse(self.ready.is_set())

    async def test_authenticated_bootstrap_requires_pem_before_model_or_connection(self):
        with patch.object(application, "GLOBAL_SKIP_AUTH_FLAG", False), \
             patch.object(application, "load_public_key", side_effect=ConfigurationError("synthetic invalid PEM")):
            with self.assertRaises(ConfigurationError):
                await application.run()
        self.loader.assert_not_called()
        self.factory.assert_not_called()

    async def test_authenticated_staging_and_production_load_only_public_pem_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public.pem"
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            material = key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            path.write_text(material, encoding="utf-8")
            self.settings.jwt_public_key_path = str(path)
            with patch.object(application, "GLOBAL_SKIP_AUTH_FLAG", False):
                for environment in ("staging", "production"):
                    self.settings.py_env = environment
                    self.stop.clear()
                    task = await self.start()
                    self.assertEqual(application.PUBLIC_KEY, path.read_text(encoding="utf-8"))
                    self.stop.set()
                    await task

    async def test_connect_failure_closes_partial_client_without_draining(self):
        self.client.is_connected = False
        self.client.connect.side_effect = ConnectionError("synthetic connection failure")
        with self.assertRaises(ConnectionError):
            await application.run(ready_event=self.ready)
        self.client.close.assert_awaited_once()
        self.client.drain.assert_not_awaited()
        self.assertFalse(self.ready.is_set())

    async def test_subscription_failure_closes_client_and_never_marks_ready(self):
        self.client.subscribe.side_effect = RuntimeError("synthetic subscribe failure")
        with self.assertRaises(RuntimeError):
            await application.run(ready_event=self.ready)
        self.assertFalse(self.ready.is_set())
        self.client.close.assert_awaited_once()

    async def test_flush_failure_does_not_claim_readiness(self):
        self.client.flush.side_effect = TimeoutError("synthetic flush timeout")
        with self.assertRaises(TimeoutError):
            await application.run(ready_event=self.ready)
        self.assertFalse(self.ready.is_set())
        self.client.close.assert_awaited_once()

    async def test_disconnection_clears_readiness_and_reconnection_restores_it(self):
        task = await self.start()
        await self.callbacks["disconnected_cb"]()
        self.assertFalse(self.ready.is_set())
        await self.callbacks["reconnected_cb"]()
        self.assertTrue(self.ready.is_set())
        self.stop.set()
        await task

    async def test_reconnection_before_subscriptions_does_not_mark_ready(self):
        original = self.client.connect.side_effect

        async def connect(url, **callbacks):
            await original(url, **callbacks)
            await callbacks["reconnected_cb"]()
            self.assertFalse(self.ready.is_set())

        self.client.connect.side_effect = connect
        task = await self.start()
        self.stop.set()
        await task

    async def test_terminal_broker_close_ends_service_with_failure(self):
        task = await self.start()
        self.client.is_closed = True
        self.client.is_connected = False
        await self.callbacks["closed_cb"]()
        with self.assertRaises(ConnectionError):
            await task
        self.assertFalse(self.ready.is_set())

    async def test_cancellation_closes_the_connection(self):
        task = await self.start()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.client.close.assert_awaited_once()
        self.assertFalse(self.ready.is_set())

    async def test_drain_failure_still_closes_the_connection(self):
        task = await self.start()
        self.client.drain.side_effect = TimeoutError("synthetic drain timeout")
        self.stop.set()
        await task
        self.client.close.assert_awaited_once()

    async def test_production_uses_base_subjects_and_staging_uses_prefix(self):
        for environment in ("staging", "production"):
            self.settings.py_env = environment
            self.stop.clear()
            self.subjects.clear()
            task = await self.start()
            expected = ("inference.tox21.smiles", "rdkit_api.get_molecule_properties",
                        "rdkit_api.to_canonical_smiles", "rdkit_api.are_same_structure",
                        "pcp_api.get_iupac_name_from_smiles")
            actual = set(self.subjects)
            self.assertEqual(actual, set(expected) if environment == "production"
                             else {f"staging.{subject}" for subject in expected})
            self.stop.set()
            await task

    async def test_stopping_does_not_restore_readiness_on_late_reconnect(self):
        task = await self.start()
        await self.callbacks["disconnected_cb"]()
        self.stop.set()
        await self.callbacks["reconnected_cb"]()
        self.assertFalse(self.ready.is_set())
        await task


if __name__ == "__main__":
    unittest.main()
