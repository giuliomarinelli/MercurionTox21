"""Real worker/broker auth, synthetic prediction and PubChem; no model/key I/O."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, '/app')
import main as application
from config import get_config, nats_subject
from nats.aio.client import Client
from nats.errors import AuthorizationError, Error

phase = 'readiness'
counts = {'authenticated_rpcs': 0, 'legacy_rpcs': 0, 'worker_auth_denials': 0}


async def ignore_error(_error):
    pass


class BoundedClient(Client):
    async def connect(self, *args, **kwargs):
        # Bound only this test; application reconnect policy is unchanged.
        kwargs.update(allow_reconnect=False, connect_timeout=1, error_cb=ignore_error)
        await super().connect(*args, **kwargs)


def credentials(environment, consumer='tox21'):
    return {'NATS_USERNAME': f'{consumer}_{environment}',
            'NATS_PASSWORD': f'synthetic-{environment}-{consumer}-password'}


async def main():
    global phase
    application.load_inference_runtime = lambda **_kwargs: (None, 'synthetic', lambda *_args: {'synthetic_prediction': True})
    application.get_iupac_name_from_smiles = lambda _smiles: 'synthetic-iupac'
    owned_clients = []

    def worker_client():
        client = BoundedClient()
        owned_clients.append(client)
        return client

    application.NATS = worker_client
    with tempfile.TemporaryDirectory(prefix='tox21-auth-probe-') as root:
        def configure(environment, auth):
            settings = get_config({'PY_ENV': 'test' if environment == 'legacy' else environment,
                                  'NATS_URL': f'nats://nats-{environment}:4222', 'VERSION': 'synthetic', **auth},
                                 root=Path(root))
            application.get_config = lambda: settings
            return settings

        for environment in ('staging', 'production', 'legacy'):
            phase = f'rpc-{environment}'
            settings = configure(environment, {} if environment == 'legacy' else credentials(environment))
            ready, stop = asyncio.Event(), asyncio.Event()
            requester = BoundedClient()
            worker = asyncio.create_task(application.run(ready_event=ready, stop_event=stop))
            try:
                # Observe worker failure too; a silent timeout never becomes a pass.
                waiter = asyncio.create_task(ready.wait())
                try:
                    done, _ = await asyncio.wait((worker, waiter), timeout=10, return_when=asyncio.FIRST_COMPLETED)
                    if worker in done:
                        await worker
                        raise AssertionError('worker-ended-before-readiness')
                    assert waiter in done and ready.is_set(), 'worker-not-ready'
                finally:
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)
                request_auth = {} if environment == 'legacy' else credentials(environment, 'nest')
                requester_settings = get_config({'PY_ENV': 'production', 'NATS_URL': settings.nats_url,
                                                 'VERSION': 'synthetic', **request_auth}, root=Path(root))
                await requester.connect(settings.nats_url, **requester_settings.nats_connect_credentials())
                cases = [
                    ('inference.tox21.smiles', {'smiles': 'CCO'}, {'synthetic_prediction': True}),
                    ('rdkit_api.to_canonical_smiles', {'smiles': 'OCC'}, {'data': 'CCO'}),
                    ('rdkit_api.are_same_structure', {'a': 'CCO', 'b': 'OCC'}, {'data': True}),
                    ('pcp_api.get_iupac_name_from_smiles', {'smiles': 'CCO'}, {'data': {'iupac_name': 'synthetic-iupac'}}),
                    ('rdkit_api.get_molecule_properties', {'smiles': 'CCO'}, None),
                ]
                for base, payload, expected in cases:
                    message = await requester.request(nats_subject(base, settings.py_env),
                                                      json.dumps({'id': 'synthetic', 'data': payload}).encode(), timeout=3)
                    actual = json.loads(message.data)
                    if expected is None:
                        assert abs(actual['data']['mwFreebase'] - 46.069) < .01
                    else:
                        assert actual == expected
                    counts['legacy_rpcs' if environment == 'legacy' else 'authenticated_rpcs'] += 1
            finally:
                stop.set()
                try:
                    await asyncio.wait_for(worker, timeout=10)
                finally:
                    await requester.close()
            assert not ready.is_set() and owned_clients[-1].is_closed

        for environment, other in (('staging', 'production'), ('production', 'staging')):
            phase = f'negative-{environment}'
            for auth in ({}, {**credentials(environment), 'NATS_PASSWORD': 'synthetic-wrong-password'},
                         {**credentials(environment), 'NATS_USERNAME': 'synthetic-unknown-user'}, credentials(other)):
                configure(environment, auth)
                ready = asyncio.Event()
                try:
                    await asyncio.wait_for(application.run(ready_event=ready), timeout=5)
                except Error as error:
                    # nats-py 2.9 initial handshake exposes the explicit -ERR as Error.
                    assert isinstance(error, AuthorizationError) or str(error) == "nats: 'Authorization Violation'"
                else:
                    raise AssertionError('unauthorized-worker-admitted')
                assert not ready.is_set() and owned_clients[-1].is_closed
                counts['worker_auth_denials'] += 1


if __name__ == '__main__':
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            asyncio.run(main())
        for environment in ('staging', 'production'):
            for consumer in ('tox21', 'nest'):
                assert credentials(environment, consumer)['NATS_PASSWORD'] not in captured.getvalue()
        print(json.dumps({'status': 'passed', **counts, 'prediction': 'synthetic', 'pubchem': 'synthetic',
                          'rdkit': 'real', 'vm_deployment': False}))
    except Exception as error:
        print(json.dumps({'status': 'failed', 'phase': phase, 'error_type': type(error).__name__, **counts}), file=sys.stderr)
        sys.exit(1)
