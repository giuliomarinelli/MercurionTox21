import asyncio
import threading
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from api.pcp import CompoundNotFoundError
from api.rdkit import InvalidSmilesError


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
            "api.pcp": SimpleNamespace(
                CompoundNotFoundError=CompoundNotFoundError,
                get_iupac_name_from_smiles=Mock(return_value="ethanol"),
            ),
            "mercurion.model": SimpleNamespace(MercurionMLP=Mock()),
        }
        real_open = open

        def test_open(path, *args, **kwargs):
            if path == config.jwt_public_key_path:
                raise AssertionError("Public mode must not read the JWT public key")
            return real_open(path, *args, **kwargs)

        with patch.dict("sys.modules", dependencies), patch("builtins.open", test_open), patch("builtins.print"):
            module = runpy.run_path(str(ROOT / "main.py"), run_name="nats_validation_test")
        cls.runtime = module["run"].__globals__
        cls.run_service = staticmethod(module["run"])
        cls.real_verify_jwt = staticmethod(module["verify_jwt"])
        cls.default_skip_auth = module["GLOBAL_SKIP_AUTH_FLAG"]

    async def asyncSetUp(self):
        self.callbacks = {}

        async def subscribe(subject, cb):
            self.callbacks[subject.removeprefix("development.")] = cb
            if len(self.callbacks) == 5:
                raise StartupComplete()

        client = SimpleNamespace(connect=AsyncMock(), subscribe=subscribe)
        self.verify_jwt = Mock(return_value={"sub": "test-user"})
        self.properties = Mock(return_value=SimpleNamespace(to_dict=lambda: {"mwFreebase": 46.07}))
        self.canonical = Mock(return_value="CCO")
        self.same = Mock(return_value=True)
        self.iupac = Mock(return_value="ethanol")
        replacements = {
            "GLOBAL_SKIP_AUTH_FLAG": False,
            "NATS": lambda: client,
            "verify_jwt": self.verify_jwt,
            "get_molecule_properties": self.properties,
            "to_canonical_smiles": self.canonical,
            "are_same_structure": self.same,
            "get_iupac_name_from_smiles": self.iupac,
        }
        patcher = patch.dict(self.runtime, replacements)
        patcher.start()
        self.addCleanup(patcher.stop)
        with self.assertRaises(StartupComplete):
            await self.run_service()

    async def request(self, subject, payload=None, raw=None):
        msg = SimpleNamespace(
            data=raw if raw is not None else json.dumps(payload).encode(),
            reply="_INBOX.test",
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
            "pcp_api.get_iupac_name_from_smiles": {"smiles": "CCO", "accessToken": TOKEN},
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
        self.iupac.assert_called_with("CCO")

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
        self.iupac.assert_not_called()

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
        self.iupac.assert_not_called()

    async def test_public_mode_is_enabled_by_default(self):
        self.assertIs(self.default_skip_auth, True)
        self.assertIsNone(self.runtime["PUBLIC_KEY"])

    async def test_public_mode_ignores_missing_and_invalid_tokens_on_all_endpoints(self):
        self.runtime["GLOBAL_SKIP_AUTH_FLAG"] = True
        self.verify_jwt.side_effect = AssertionError("JWT verification must be skipped")
        for subject, valid in self.valid_requests().items():
            without_token = {key: value for key, value in valid.items() if key != "accessToken"}
            cases = [without_token]
            cases += [{**without_token, "accessToken": token} for token in (None, "", "expired-token", 123, [], {})]
            for payload in cases:
                for enveloped in (False, True):
                    with self.subTest(subject=subject, enveloped=enveloped, payload=payload):
                        wire = {"id": "request-id", "data": payload} if enveloped else payload
                        response = await self.request(subject, wire)
                        self.assertNotIn("error", response)
        self.verify_jwt.assert_not_called()

    async def test_public_mode_preserves_payload_validation(self):
        self.runtime["GLOBAL_SKIP_AUTH_FLAG"] = True
        for subject, valid in self.valid_requests().items():
            without_token = {key: value for key, value in valid.items() if key != "accessToken"}
            smiles_field = "a" if "a" in valid else "smiles"
            for invalid in ({**without_token, smiles_field: " "}, {**without_token, "unexpected": True}):
                response = await self.request(subject, invalid)
                self.assertTrue(response["error"].startswith("Invalid request:"))
        response = await self.request("rdkit_api.to_canonical_smiles", {"smiles": "CCO", "opts": {"isomeric": "false"}})
        self.assertTrue(response["error"].startswith("Invalid request:"))
        self.verify_jwt.assert_not_called()
        self.properties.assert_not_called()
        self.canonical.assert_not_called()
        self.same.assert_not_called()
        self.iupac.assert_not_called()

    async def test_authenticated_mode_requires_token_on_all_endpoints(self):
        for subject, valid in self.valid_requests().items():
            without_token = {key: value for key, value in valid.items() if key != "accessToken"}
            for payload in (without_token, {**without_token, "accessToken": None}):
                response = await self.request(subject, payload)
                self.assertTrue(response["error"].startswith("Invalid request:"))
        self.verify_jwt.assert_not_called()

    async def test_jwt_verifier_rejects_missing_token_or_key_without_decoding(self):
        # Test the real verifier, independently of callback authentication mocks.
        # asyncSetUp patches the verifier, so use the original saved at startup.
        verify = self.real_verify_jwt
        with patch.object(self.runtime["jwt"], "decode") as decode:
            self.assertIsNone(verify(None))
            self.assertIsNone(verify(TOKEN))
            decode.assert_not_called()
            with patch.dict(self.runtime, {"PUBLIC_KEY": "test-public-key"}):
                self.assertIsNone(verify(None))
                decode.assert_not_called()
                decode.return_value = {"sub": "test-user"}
                self.assertEqual(verify(TOKEN), {"sub": "test-user"})
                decode.assert_called_once_with(
                    TOKEN, "test-public-key", algorithms=["RS256"], audience="mercurion-api"
                )

    async def test_pubsub_without_reply_does_not_attempt_response(self):
        self.runtime["GLOBAL_SKIP_AUTH_FLAG"] = True
        for subject, valid in self.valid_requests().items():
            without_token = {key: value for key, value in valid.items() if key != "accessToken"}
            for raw in (json.dumps(without_token).encode(), b"{", b"{}"):
                with self.subTest(subject=subject, raw=raw):
                    msg = SimpleNamespace(data=raw, reply="", respond=AsyncMock())
                    await self.callbacks[subject](msg)
                    msg.respond.assert_not_awaited()
        self.verify_jwt.assert_not_called()

    async def test_invalid_chemical_smiles_are_request_errors(self):
        for subject, operation in (
            ("rdkit_api.get_molecule_properties", self.properties),
            ("rdkit_api.to_canonical_smiles", self.canonical),
            ("pcp_api.get_iupac_name_from_smiles", self.iupac),
        ):
            operation.side_effect = InvalidSmilesError("invalid")
            response = await self.request(subject, {"smiles": "invalid", "accessToken": TOKEN})
            self.assertEqual(response, {"error": "Invalid SMILES"})

    async def test_pubchem_not_found_is_distinct_from_invalid_smiles(self):
        self.iupac.side_effect = CompoundNotFoundError("not found")
        response = await self.request(
            "pcp_api.get_iupac_name_from_smiles", {"smiles": "CCO", "accessToken": TOKEN}
        )
        self.assertEqual(response, {"error": "Compound not found in PubChem"})

    async def test_pubchem_lookup_does_not_block_event_loop(self):
        started = threading.Event()
        release = threading.Event()

        def lookup(smiles):
            started.set()
            if not release.wait(timeout=3):
                raise TimeoutError("test lookup timed out")
            return "ethanol"

        self.iupac.side_effect = lookup
        task = asyncio.create_task(self.request(
            "pcp_api.get_iupac_name_from_smiles", {"smiles": "CCO", "accessToken": TOKEN}
        ))
        try:
            async def wait_for_start():
                while not started.is_set():
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(wait_for_start(), timeout=1)
            response = await self.request(
                "rdkit_api.get_molecule_properties", {"smiles": "CCO", "accessToken": TOKEN}
            )
            self.assertNotIn("error", response)
            self.assertFalse(task.done())
        finally:
            release.set()
            result = await task
        self.assertEqual(result, {"data": {"iupac_name": "ethanol"}})


if __name__ == "__main__":
    unittest.main()
