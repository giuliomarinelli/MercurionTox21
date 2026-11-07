"""Per-service bootstrap policy and real synthetic RS256 handler verification."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import jwt
from pydantic import ValidationError

import main as application
from config import ConfigurationError, get_config
from schemas.schemas import Configuration


SERVICES = {
    'AUTH_INFERENCE': ('inference.tox21.smiles', {'smiles': 'CCO'}),
    'AUTH_RDKIT_PROPERTIES': ('rdkit_api.get_molecule_properties', {'smiles': 'CCO'}),
    'AUTH_RDKIT_CANONICAL_SMILES': ('rdkit_api.to_canonical_smiles', {'smiles': 'CCO'}),
    'AUTH_RDKIT_SAME_STRUCTURE': ('rdkit_api.are_same_structure', {'a': 'CCO', 'b': 'OCC'}),
    'AUTH_PUBCHEM_IUPAC_NAME': ('pcp_api.get_iupac_name_from_smiles', {'smiles': 'CCO'}),
}
ENV = {'PY_ENV': 'production', 'NATS_URL': 'nats://synthetic.invalid:4222', 'VERSION': 'qa'}


class ServiceConfigurationTests(unittest.TestCase):
    def test_omitted_flags_default_to_disabled_in_pydantic_and_environment_loader(self):
        settings = get_config(ENV)
        direct = Configuration(py_env='production', nats_url=ENV['NATS_URL'], version='qa',
                               jwt_public_key_path='unused.pem')
        for field in SERVICES:
            self.assertEqual(getattr(settings, field.lower()), 'disabled')
            self.assertEqual(getattr(direct, field.lower()), 'disabled')
        for global_skip in (True, False):
            self.assertEqual(set(settings.resolved_skip_auth(global_skip).values()), {global_skip})

    def test_exact_allowed_values_and_invalid_values_for_every_flag(self):
        for field in SERVICES:
            for value in ('allow', 'require-jwt', 'disabled'):
                with self.subTest(field=field, value=value):
                    self.assertEqual(getattr(get_config({**ENV, field: value}), field.lower()), value)
            for value in ('', ' ', 'ALLOW', ' allow ', 'deny-withouth-access-token', 'null', None, 1, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ConfigurationError) as caught:
                        get_config({**ENV, field: value})
                    self.assertEqual(str(caught.exception), 'Invalid configuration fields: ' + field.lower())
                    with self.assertRaises(ValidationError):
                        Configuration(py_env='test', nats_url=ENV['NATS_URL'], version='qa',
                                      jwt_public_key_path='unused.pem', **{field.lower(): value})

    def test_dotenv_precedence_omission_and_bare_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field in SERVICES:
                (root / '.env.test').write_text(f'{field}=require-jwt\n')
                settings = get_config({**ENV, 'PY_ENV': 'test'}, root=root)
                self.assertEqual(getattr(settings, field.lower()), 'require-jwt')
                settings = get_config({**ENV, 'PY_ENV': 'test', field: 'allow'}, root=root)
                self.assertEqual(getattr(settings, field.lower()), 'allow')
                (root / '.env.test').write_text(field + '\n')
                with self.assertRaises(ConfigurationError):
                    get_config({**ENV, 'PY_ENV': 'test'}, root=root)

    def test_override_matrix_is_independent_for_each_service(self):
        for field, (subject, _) in SERVICES.items():
            for global_skip in (True, False):
                for policy in ('allow', 'require-jwt', 'disabled'):
                    with self.subTest(field=field, global_skip=global_skip, policy=policy):
                        resolved = get_config({**ENV, field: policy}).resolved_skip_auth(global_skip)
                        self.assertEqual(resolved[subject], global_skip if policy == 'disabled' else policy == 'allow')
                        self.assertTrue(all(value == global_skip for name, value in resolved.items() if name != subject))


class ServiceAuthenticationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.pem = cls.key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        now = datetime.now(timezone.utc)
        claims = {'sub': 'synthetic', 'aud': 'mercurion-api', 'exp': now + timedelta(minutes=10)}
        cls.tokens = {
            'valid': jwt.encode(claims, cls.key, algorithm='RS256'),
            'expired': jwt.encode({**claims, 'exp': now - timedelta(minutes=10)}, cls.key, algorithm='RS256'),
            'wrong-audience': jwt.encode({**claims, 'aud': 'other'}, cls.key, algorithm='RS256'),
            'wrong-signature': jwt.encode(claims, rsa.generate_private_key(public_exponent=65537, key_size=2048), algorithm='RS256'),
            'malformed': 'not-a-valid-access-token',
        }

    async def boot(self, settings, global_skip, *, mock_key=True):
        self.callbacks = {}

        async def subscribe(subject, cb):
            self.callbacks[subject] = cb

        client = SimpleNamespace(connect=AsyncMock(), subscribe=subscribe, flush=AsyncMock(),
                                 drain=AsyncMock(), close=AsyncMock(), is_connected=True, is_closed=False)
        self.operations = [Mock(return_value={}), Mock(return_value=SimpleNamespace(to_dict=lambda: {})),
                           Mock(return_value='CCO'), Mock(return_value=True), Mock(return_value='ethanol')]
        self.model_loader = Mock(return_value=(Mock(), 'cpu', self.operations[0]))
        self.factory = Mock(return_value=client)
        replacements = dict(get_config=Mock(return_value=settings), GLOBAL_SKIP_AUTH_FLAG=global_skip,
                            load_inference_runtime=self.model_loader, NATS=self.factory,
                            read_build_identity=Mock(return_value=None), get_molecule_properties=self.operations[1],
                            to_canonical_smiles=self.operations[2], are_same_structure=self.operations[3],
                            get_iupac_name_from_smiles=self.operations[4])
        if mock_key:
            replacements['load_public_key'] = Mock(return_value=self.pem)
        stop = asyncio.Event()
        stop.set()
        with patch.multiple(application, **replacements), patch('builtins.print'):
            await application.run(stop_event=stop)

    async def request(self, subject, payload):
        message = SimpleNamespace(data=json.dumps(payload).encode(), reply='_INBOX.synthetic', respond=AsyncMock())
        # Operations are patched while callbacks execute; verifier and schemas are real.
        with patch.multiple(application, get_molecule_properties=self.operations[1],
                            to_canonical_smiles=self.operations[2], are_same_structure=self.operations[3],
                            get_iupac_name_from_smiles=self.operations[4]):
            await self.callbacks[subject](message)
        message.respond.assert_awaited_once()
        return json.loads(message.respond.await_args.args[0])

    async def test_all_handler_policy_global_token_and_payload_combinations(self):
        for field, (subject, body) in SERVICES.items():
            for global_skip in (True, False):
                for policy in (None, 'disabled', 'allow', 'require-jwt'):
                    settings = get_config({**ENV, **({field: policy} if policy is not None else {})})
                    await self.boot(settings, global_skip)
                    skip = global_skip if policy in (None, 'disabled') else policy == 'allow'
                    for kind, token in [('missing', None), *self.tokens.items()]:
                        for wrapped in (False, True):
                            with self.subTest(service=field, global_skip=global_skip, policy=policy, token=kind, wrapped=wrapped):
                                for operation in self.operations:
                                    operation.reset_mock()
                                payload = {**body, **({'accessToken': token} if token is not None else {})}
                                response = await self.request(subject, {'data': payload} if wrapped else payload)
                                if skip or kind == 'valid':
                                    self.assertNotIn('error', response)
                                    self.assertEqual(sum(op.call_count for op in self.operations), 1)
                                else:
                                    self.assertIn('error', response)
                                    if kind == 'missing':
                                        self.assertTrue(response['error'].startswith('Invalid request:'))
                                    else:
                                        self.assertEqual(response, {'error': 'Invalid or expired access token'})
                                    self.assertTrue(all(op.call_count == 0 for op in self.operations))
                    for operation in self.operations:
                        operation.reset_mock()
                    response = await self.request(subject, {**body, 'accessToken': self.tokens['valid'], 'unexpected': True})
                    self.assertTrue(response['error'].startswith('Invalid request:'))
                    self.assertTrue(all(op.call_count == 0 for op in self.operations))

    async def test_one_required_service_needs_pem_even_with_public_global(self):
        for field in SERVICES:
            settings = get_config({**ENV, field: 'require-jwt', 'JWT_PUBLIC_KEY_PATH': '/not-present/qa.pem'})
            with self.subTest(field=field), self.assertRaises(ConfigurationError):
                await self.boot(settings, True, mock_key=False)
            self.model_loader.assert_not_called()
            self.factory.assert_not_called()

    async def test_all_allow_overrides_restrictive_global_without_reading_pem(self):
        settings = get_config({**ENV, **{field: 'allow' for field in SERVICES}, 'JWT_PUBLIC_KEY_PATH': '/not-present/qa.pem'})
        with patch('runtime.Path.read_text', side_effect=AssertionError('No key should be read')):
            await self.boot(settings, False, mock_key=False)
        self.assertIsNone(application.PUBLIC_KEY)
        for _, (subject, payload) in SERVICES.items():
            self.assertNotIn('error', await self.request(subject, payload))

    async def test_invalid_flag_aborts_before_key_model_or_connection(self):
        for field in SERVICES:
            ready = asyncio.Event()
            ready.set()
            with patch.object(application, 'get_config', side_effect=lambda: get_config({**ENV, field: 'invalid'})), \
                 patch.object(application, 'load_public_key') as key, \
                 patch.object(application, 'load_inference_runtime') as model, \
                 patch.object(application, 'NATS') as broker:
                with self.subTest(field=field), self.assertRaises(ConfigurationError):
                    await application.run(ready_event=ready)
                self.assertFalse(ready.is_set())
                key.assert_not_called()
                model.assert_not_called()
                broker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
