"""Local synthetic broker authentication smoke; no existing service or host port."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
NATS_IMAGE = 'nats@sha256:77a161c7d888a1791d5a46b50cd87a2f8466ce40a33cbb51238d227ace01e35a'
LABEL = 'mercurion.tox21-nats-auth-run'


def run_docker(context, *args):
    return subprocess.check_output(['docker', '--context', context, *args], text=True, timeout=120).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='Already-local Linux runtime image; resolved to an immutable image ID')
    parser.add_argument('--source-overlay', action='store_true', help='Test current selected source on cached dependencies; not a release image test')
    parser.add_argument('--docker-context', choices=('desktop-linux', 'default'), default='desktop-linux')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    docker = partial(run_docker, args.docker_context)
    endpoint = json.loads(docker('context', 'inspect', args.docker_context))[0]['Endpoints']['docker']['Host']
    if not endpoint.startswith(('unix://', 'npipe://')):
        raise ValueError('This test requires a local Docker socket')
    image_id = docker('image', 'inspect', args.image, '--format', '{{.Id}}')
    docker('image', 'inspect', NATS_IMAGE)
    run_id = uuid.uuid4().hex
    name = 'tox21-nats-auth-' + run_id
    network = False
    try:
        with tempfile.TemporaryDirectory(prefix='tox21-nats-auth-') as directory:
            docker('network', 'create', '--internal', '--label', f'{LABEL}={run_id}', name)
            network = True
            for environment in ('staging', 'production', 'legacy'):
                configuration = {'port': 4222}
                if environment != 'legacy':
                    configuration['authorization'] = {'users': [
                        {'user': f'{consumer}_{environment}', 'password': f'synthetic-{environment}-{consumer}-password'}
                        for consumer in ('nest', 'tox21')]}
                path = Path(directory) / f'{environment}.json'
                path.write_text(json.dumps(configuration), encoding='utf-8')
                docker('run', '-d', '--pull', 'never', '--name', f'{name}-{environment}', '--label', f'{LABEL}={run_id}',
                       '--network', name, '--network-alias', f'nats-{environment}', '--user', '10021:10021',
                       '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                       '--memory', '64m', '--cpus', '0.25', '--pids-limit', '32',
                       '--mount', f'type=bind,src={path},dst=/qa.conf,readonly', NATS_IMAGE, '--config', '/qa.conf')
            mounts = ['--mount', f'type=bind,src={ROOT / "scripts/fixtures/nats_broker_auth.py"},dst=/qa/probe.py,readonly']
            if args.source_overlay:
                for source in ('main.py', 'config.py', 'runtime.py', 'api', 'schemas', 'mercurion'):
                    mounts.extend(['--mount', f'type=bind,src={ROOT / source},dst=/app/{source},readonly'])
            result = subprocess.run(['docker', '--context', args.docker_context, 'run', '--pull', 'never',
                                     '--name', f'{name}-probe', '--label', f'{LABEL}={run_id}', '--network', name,
                                     '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                                     '--user', '10021:10021', '--memory', '1g', '--cpus', '1', '--pids-limit', '128',
                                     '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m,mode=1777',
                                     '--env', 'OMP_NUM_THREADS=1', '--env', 'OPENBLAS_NUM_THREADS=1',
                                     *mounts, '--entrypoint', 'python', image_id, '-B', '/qa/probe.py'],
                                    capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise RuntimeError('NATS auth probe failed: ' + result.stderr)
            report = {**json.loads(result.stdout), 'image_id': image_id, 'nats_image': NATS_IMAGE,
                      'source_overlay': args.source_overlay}
    finally:
        # Labels also find partially-created or timed-out containers.
        for container in docker('ps', '-aq', '--filter', f'label={LABEL}={run_id}').split():
            docker('rm', '--force', container)
        if network:
            docker('network', 'rm', name)
    report['cleanup'] = 'passed'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
