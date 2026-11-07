"""Real worker/NATS mixed-policy smoke; synthetic RSA keys, no host ports/provider."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from smoke_container import NATS_IMAGE, docker, state


KEYGEN = '''
import json
from cryptography.hazmat.primitives import serialization as s
from cryptography.hazmat.primitives.asymmetric import rsa
k=rsa.generate_private_key(public_exponent=65537,key_size=2048)
print(json.dumps({'private':k.private_bytes(s.Encoding.PEM,s.PrivateFormat.PKCS8,s.NoEncryption()).decode(),
                  'public':k.public_key().public_bytes(s.Encoding.PEM,s.PublicFormat.SubjectPublicKeyInfo).decode()}))
'''

PROBE = '''
import asyncio,json,os
from datetime import datetime,timedelta,timezone
from pathlib import Path
import jwt
from nats.aio.client import Client

async def probe():
    nc=Client()
    try:
        await nc.connect('nats://nats:4222',allow_reconnect=False,connect_timeout=2)
        now=datetime.now(timezone.utc)
        claims={'sub':'synthetic','aud':'mercurion-api','exp':now+timedelta(minutes=5)}
        key=Path('/qa/private.pem').read_text()
        valid=jwt.encode(claims,key,algorithm='RS256')
        expired=jwt.encode({**claims,'exp':now-timedelta(minutes=5)},key,algorithm='RS256')
        async def request(subject,body):
            response=await nc.request(subject,json.dumps(body).encode(),timeout=3)
            return json.loads(response.data)
        # Both globals are exercised: public override / inherited protected RDKit.
        public=os.environ['GLOBAL_SKIP']=='true'
        cases=[('inference.tox21.smiles',{'smiles':'CCO'},True),
               ('rdkit_api.get_molecule_properties',{'smiles':'CCO'},False),
               ('rdkit_api.to_canonical_smiles',{'smiles':'OCC'},public),
               ('rdkit_api.are_same_structure',{'a':'CCO','b':'OCC'},True),
               ('pcp_api.get_iupac_name_from_smiles',{'smiles':'not-a-smiles'},True)]
        checks=0
        for subject,body,protected in cases:
            for kind,token in [('missing',None),('valid',valid),('expired',expired),('malformed','not-a-valid-token')]:
                result=await request(subject,{**body,**({'accessToken':token} if token is not None else {})})
                if protected and kind!='valid':
                    assert 'error' in result,(subject,kind,result)
                    if kind=='missing': assert result['error'].startswith('Invalid request:')
                    else: assert result=={'error':'Invalid or expired access token'}
                elif subject.startswith('pcp_api'):
                    assert result=={'error':'Invalid SMILES'},result
                else:
                    assert 'error' not in result,(subject,kind,result)
                    if subject.startswith('inference'): assert set(result)=={'SR-ATAD5','NR-AhR','SR-MMP','SR-p53'}
                    elif subject.endswith('get_molecule_properties'): assert abs(result['data']['mwFreebase']-46.069)<.01
                    elif subject.endswith('to_canonical_smiles'): assert result=={'data':'CCO'}
                    else: assert result=={'data':True}
                checks+=1
            invalid=await request(subject,{'accessToken':valid})
            assert invalid['error'].startswith('Invalid request:')
            checks+=1
        print(json.dumps({'checks':checks,'global_skip_auth':public,'subjects':5,'pubchem':'local-invalid-smiles-only'}))
    finally:
        await nc.close()
asyncio.run(probe())
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='Production target image; base NATS subjects')
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = 't21-auth-' + uuid.uuid4().hex[:10]
    owned = []
    network = None
    report = {'status': 'failed', 'scenarios': [], 'host_ports': 'none'}
    try:
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            root = Path(directory)
            # Generated solely for this smoke, never use or read developer keys.
            keys = json.loads(docker('run', '--rm', '--network', 'none', '--entrypoint', 'python', args.image, '-c', KEYGEN))
            for name in ('private', 'public'):
                path = root / (name + '.pem')
                path.write_text(keys[name], encoding='utf-8')
                path.chmod(0o644)
            del keys
            network = docker('network', 'create', '--internal', prefix)
            broker = docker('run', '-d', '--name', prefix + '-nats', '--network', network,
                            '--network-alias', 'nats', NATS_IMAGE)
            owned.append(broker)
            for global_skip in (True, False):
                name = prefix + ('-public' if global_skip else '-protected')
                # False is a test harness assignment to the existing source constant,
                # not a new environment switch or a change to the shipped default.
                code = f'import asyncio,main;main.GLOBAL_SKIP_AUTH_FLAG={global_skip};asyncio.run(main.main())'
                worker = docker('run', '-d', '--name', name, '--network', network,
                                '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                                '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
                                '--mount', f'type=bind,src={root / "public.pem"},dst=/qa/public.pem,readonly',
                                '--env', 'NATS_URL=nats://nats:4222', '--env', 'JWT_PUBLIC_KEY_PATH=/qa/public.pem',
                                '--env', 'AUTH_INFERENCE=require-jwt', '--env', 'AUTH_RDKIT_PROPERTIES=allow',
                                '--env', 'AUTH_RDKIT_CANONICAL_SMILES=disabled', '--env', 'AUTH_RDKIT_SAME_STRUCTURE=require-jwt',
                                '--env', 'AUTH_PUBCHEM_IUPAC_NAME=require-jwt', '--entrypoint', 'python', args.image, '-c', code)
                owned.append(worker)
                deadline = time.monotonic() + 60
                while True:
                    logs = docker('logs', worker)
                    if '[MercurionTox21 > main] Ready in ' in logs:
                        break
                    if not state(worker)['Running'] or time.monotonic() > deadline:
                        raise RuntimeError('Auth worker did not become ready')
                    time.sleep(.5)
                for _ in range(2):
                    result = subprocess.run(['docker', 'run', '--rm', '-i', '--network', network,
                                             '--mount', f'type=bind,src={root / "private.pem"},dst=/qa/private.pem,readonly',
                                             '--env', 'GLOBAL_SKIP=' + str(global_skip).lower(), '--entrypoint', 'python',
                                             args.image, '-B', '-'], input=PROBE, capture_output=True, text=True, timeout=30)
                    if result.returncode:
                        raise RuntimeError('Auth RPC probe failed: ' + result.stderr)
                    report['scenarios'].append(json.loads(result.stdout))
                docker('stop', '--time', '15', worker)
                assert state(worker)['ExitCode'] == 0, 'Worker did not stop gracefully'
            # CLI validation must fail before connecting/loading model; no broker needed.
            for field, value in [('AUTH_INFERENCE', 'typo'), ('AUTH_RDKIT_PROPERTIES', '')]:
                result = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--env',
                                         'NATS_URL=nats://unreachable:4222', '--env', field + '=' + value,
                                         args.image], capture_output=True, text=True, timeout=20)
                assert result.returncode != 0
                assert 'FATAL: Invalid configuration fields: ' + field.lower() in result.stderr
            report['invalid_configuration_bootstrap'] = 'passed'
            report['status'] = 'passed'
    finally:
        for container in owned:
            logs = subprocess.run(['docker', 'logs', container], capture_output=True, text=True, timeout=20)
            (args.report_dir / (container + '.log')).write_text(logs.stdout + logs.stderr, encoding='utf-8')
        cleanup = True
        for container in reversed(owned):
            result = subprocess.run(['docker', 'rm', '--force', container], capture_output=True, timeout=20)
            cleanup &= result.returncode == 0
        if network:
            result = subprocess.run(['docker', 'network', 'rm', network], capture_output=True, timeout=20)
            cleanup &= result.returncode == 0
        report['cleanup'] = cleanup
        (args.report_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if not cleanup:
            raise RuntimeError('Auth smoke cleanup failed')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
