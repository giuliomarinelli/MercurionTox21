"""Executed inside the worker by the host smoke driver; no provider access needed."""
import argparse
import asyncio
import json
import os
from nats.aio.client import Client


async def probe(target):
    prefix = '' if target == 'production' else target + '.'
    nc = Client()
    try:
        await asyncio.wait_for(nc.connect(os.environ.get('RPC_NATS_URL', 'nats://nats:4222'),
                                         connect_timeout=1, allow_reconnect=False,
                                         max_reconnect_attempts=1), timeout=5)

        async def request(subject, payload):
            reply = await nc.request(prefix + subject, json.dumps(payload).encode(), timeout=3)
            return json.loads(reply.data)

        inference = await request('inference.tox21.smiles', {'data': {'smiles': 'CCO'}})
        assert set(inference) == {'SR-ATAD5', 'NR-AhR', 'SR-MMP', 'SR-p53'}
        for item in inference.values():
            assert set(item) == {'probability', 'threshold', 'is_positive'}
            assert 0 <= item['probability'] <= 1 and 0 <= item['threshold'] <= 1
            assert item['is_positive'] is (item['probability'] > item['threshold'])
        properties = await request('rdkit_api.get_molecule_properties', {'smiles': 'CCO'})
        assert abs(properties['data']['mwFreebase'] - 46.069) < 0.01
        canonical = await request('rdkit_api.to_canonical_smiles', {'smiles': 'OCC'})
        assert canonical == {'data': 'CCO'}
        same = await request('rdkit_api.are_same_structure', {'a': 'CCO', 'b': 'OCC'})
        assert same == {'data': True}
        # Validation must reject before the PubChem boundary; the network has no egress.
        pcp = await request('pcp_api.get_iupac_name_from_smiles', {})
        assert set(pcp) == {'error'} and pcp['error'].startswith('Invalid request:')
        invalid = await request('rdkit_api.to_canonical_smiles', {'smiles': 'not-a-smiles'})
        assert invalid == {'error': 'Invalid SMILES'}
        print(json.dumps({'subjects': 5, 'inference': 'real', 'pubchem': 'validation-only',
                          'invalid_smiles': 'passed', 'namespace': target}))
    finally:
        await nc.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, choices=['test', 'staging', 'production'])
    asyncio.run(probe(parser.parse_args().target))
