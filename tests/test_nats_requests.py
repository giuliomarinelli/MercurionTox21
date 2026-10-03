import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


TOKEN = "test-access-token"
ROOT = Path(__file__).resolve().parents[1]


class StartupComplete(Exception):
    pass


class NatsRequestTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Isolate startup from local configuration, JWT keys and model files.
        config = SimpleNamespace(
            py_env="development",
            nats_url="nats://test.invalid:4222",
            version="test",
            jwt_public_key_path="test-only-public-key.pem",
        )
        dependencies = {
            "config": SimpleNamespace(get_config=lambda: config),
            "torch": Mock(),
            "api.inference": SimpleNamespace(predict=Mock(return_value={"prediction": 1})),
            "mercurion.model": SimpleNamespace(MercurionMLP=Mock()),
        }
        real_open = open

        def test_open(path, *args, **kwargs):
            if path == config.jwt_public_key_path:
                return io.StringIO("test-public-key")
            return real_open(path, *args, **kwargs)

        with patch.dict("sys.modules", dependencies), patch("builtins.open", test_open), patch("builtins.print"):
            module = runpy.run_path(str(ROOT / "main.py"), run_name="nats_validation_test")
        cls.run_service = staticmethod(module["run"])
        cls.runtime = cls.run_service.__globals__

    async def asyncSetUp(self):
        self.callbacks = {}

        async def subscribe(subject, cb):
            self.callbacks[subject.removeprefix("development.")] = cb
            if len(self.callbacks) == 4:
                raise StartupComplete()

        client = SimpleNamespace(connect=AsyncMock(), subscribe=subscribe)
        self.verify_jwt = Mock(return_value={"sub": "test-user"})
        self.properties = Mock(return_value=SimpleNamespace(to_dict=lambda: {"mwFreebase": 46.07}))
        self.canonical = Mock(return_value="CCO")
        self.same = Mock(return_value=True)
        replacements = {
            "NATS": lambda: client,
            "verify_jwt": self.verify_jwt,
            "get_molecule_properties": self.properties,
            "to_canonical_smiles": self.canonical,
            "are_same_structure": self.same,
        }
        patcher = patch.dict(self.runtime, replacements)
        patcher.start()
        self.addCleanup(patcher.stop)
        with self.assertRaises(StartupComplete):
            await self.run_service()

    async def request(self, subject, payload=None, raw=None):
        msg = SimpleNamespace(
            data=raw if raw is not None else json.dumps(payload).encode(),
            respond=AsyncMock(),
        )
        await self.callbacks[subject](msg)
        msg.respond.assert_awaited_once()
        return json.loads(msg.respond.await_args.args[0])

    def valid_requests(self):
        return {
            "inference.tox21.smiles": {"smiles": "CCO", "accessToken": TOKEN},
            "rdkit_api.get_molecule_properties": {"smiles": "CCO", "accessToken": TOKEN},
            "rdkit_api.to_canonical_smiles": {"smiles": "CCO", "accessToken": TOKEN},
            "rdkit_api.are_same_structure": {"a": "CCO", "b": "OCC", "accessToken": TOKEN},
        }

    async def test_valid_requests_accept_direct_and_nest_enveloped_payloads(self):
        for subject, payload in self.valid_requests().items():
            for enveloped in (False, True):
                with self.subTest(subject=subject, enveloped=enveloped):
                    wire = {"id": "request-id", "data": payload} if enveloped else payload
                    response = await self.request(subject, wire)
                    self.assertNotIn("error", response)
                    self.verify_jwt.assert_called_with(TOKEN)
        self.properties.assert_called_with("CCO")
        self.canonical.assert_called_with("CCO", isomeric=True, kekule=False)
        self.same.assert_called_with("CCO", "OCC")

    async def test_invalid_payloads_are_rejected_before_authentication(self):
        for subject, valid in self.valid_requests().items():
            smiles_field = "a" if "a" in valid else "smiles"
            cases = [None, [], "CCO", {}, {**valid, "unexpected": True}]
            cases += [{**valid, smiles_field: value} for value in (None, 123, True, [], " ", "C" * 1025)]
            cases += [{**valid, "accessToken": value} for value in (None, 123, "short", "x" * 4097)]
            for payload in cases:
                with self.subTest(subject=subject, payload_type=type(payload).__name__):
                    response = await self.request(subject, payload)
                    self.assertTrue(response["error"].startswith("Invalid request:"))
        self.verify_jwt.assert_not_called()
        self.properties.assert_not_called()
        self.canonical.assert_not_called()
        self.same.assert_not_called()

    async def test_both_comparison_smiles_are_validated(self):
        for value in (None, 123, " ", "C" * 1025):
            response = await self.request(
                "rdkit_api.are_same_structure", {"a": "CCO", "b": value, "accessToken": TOKEN}
            )
            self.assertTrue(response["error"].startswith("Invalid request:"))
        self.verify_jwt.assert_not_called()

    async def test_canonical_options_defaults_and_explicit_booleans(self):
        for opts in (None, {}, {"isomeric": False, "kekule": True}):
            with self.subTest(opts=opts):
                response = await self.request(
                    "rdkit_api.to_canonical_smiles", {"smiles": "CCO", "accessToken": TOKEN, "opts": opts}
                )
                self.assertEqual(response, {"data": "CCO"})
                self.canonical.assert_called_with(
                    "CCO", isomeric=(opts or {}).get("isomeric", True), kekule=(opts or {}).get("kekule", False)
                )

    async def test_invalid_canonical_options_are_rejected(self):
        cases = [[], False, 0, "", {"unknown": True}]
        cases += [{field: value} for field in ("isomeric", "kekule") for value in ("false", 0, 1, None)]
        for opts in cases:
            with self.subTest(opts=opts):
                response = await self.request(
                    "rdkit_api.to_canonical_smiles", {"smiles": "CCO", "accessToken": TOKEN, "opts": opts}
                )
                self.assertTrue(response["error"].startswith("Invalid request:"))
        self.verify_jwt.assert_not_called()
        self.canonical.assert_not_called()

    async def test_malformed_json_and_utf8_are_request_errors(self):
        for subject in self.callbacks:
            for raw in (b"{", b"", b"\xff"):
                with self.subTest(subject=subject, raw=raw):
                    response = await self.request(subject, raw=raw)
                    self.assertTrue(response["error"].startswith("Invalid request:"))
        self.verify_jwt.assert_not_called()

    async def test_validation_errors_do_not_echo_input_tokens(self):
        for subject, valid in self.valid_requests().items():
            for invalid in ({**valid, "unexpected": True}, {**valid, "accessToken": TOKEN * 300}):
                response = await self.request(subject, invalid)
                self.assertNotIn(TOKEN, response["error"])

    async def test_strings_are_trimmed_before_authentication_and_rdkit(self):
        response = await self.request(
            "rdkit_api.get_molecule_properties", {"smiles": " CCO ", "accessToken": f" {TOKEN} "}
        )
        self.assertNotIn("error", response)
        self.verify_jwt.assert_called_with(TOKEN)
        self.properties.assert_called_with("CCO")

    async def test_invalid_jwt_still_blocks_rdkit(self):
        self.verify_jwt.return_value = None
        for subject, payload in self.valid_requests().items():
            response = await self.request(subject, payload)
            self.assertEqual(response, {"error": "Invalid or expired access token"})
        self.properties.assert_not_called()
        self.canonical.assert_not_called()
        self.same.assert_not_called()


if __name__ == "__main__":
    unittest.main()
