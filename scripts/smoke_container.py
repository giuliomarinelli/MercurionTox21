"""Five-subject worker smoke and broker recovery on an owned internal Docker network."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

NATS_IMAGE = 'nats@sha256:77a161c7d888a1791d5a46b50cd87a2f8466ce40a33cbb51238d227ace01e35a'


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True, timeout=120).strip()


def state(container):
    return json.loads(docker('inspect', '--format', '{{json .State}}', container))


def rpc(worker, target, broker_url):
    return subprocess.run(['docker', 'exec', '-i', '--env', 'RPC_NATS_URL=' + broker_url,
                           worker, 'python', '-B', '-', '--target', target],
                          input=Path(__file__).with_name('container_rpc_probe.py').read_text(encoding='utf-8'),
                          capture_output=True, text=True, timeout=30)


def wait_ready(worker, target, broker_url):
    deadline = time.monotonic() + 60
    while True:
        result = rpc(worker, target, broker_url)
        if result.returncode == 0:
            return json.loads(result.stdout)
        if not state(worker)['Running'] or time.monotonic() >= deadline:
            raise RuntimeError('Worker RPC readiness failed: ' + result.stderr + docker('logs', worker))
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--target', choices=['test', 'staging', 'production'], required=True)
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    prefix = 'mercurion-tox21-qa-' + uuid.uuid4().hex[:10]
    network = broker = worker = None
    try:
        network = docker('network', 'create', '--internal', prefix)
        broker = docker('run', '-d', '--name', prefix + '-nats', '--network', network,
                        '--network-alias', 'nats', NATS_IMAGE)
        address = docker('inspect', '--format', '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}', broker)
        broker_url = 'nats://' + address + ':4222'
        worker = docker('run', '-d', '--name', prefix + '-worker', '--network', network,
                        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                        '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m', '--env', 'NATS_URL=nats://nats:4222',
                        args.image)
        before = wait_ready(worker, args.target, broker_url)
        started = state(worker)['StartedAt']
        docker('stop', '--time', '5', broker)
        unavailable = rpc(worker, args.target, broker_url)
        assert unavailable.returncode != 0, 'RPC unexpectedly succeeded with broker stopped'
        assert state(worker)['Running'], 'Worker exited during broker outage'
        docker('start', broker)
        after = wait_ready(worker, args.target, broker_url)
        assert state(worker)['StartedAt'] == started, 'Worker restarted instead of reconnecting'
        docker('stop', '--time', '15', worker)
        stopped = state(worker)
        assert stopped['ExitCode'] == 0 and not stopped['OOMKilled'], stopped
        report = {'image': args.image, 'target': args.target, 'before_outage': before,
                  'after_recovery': after, 'broker_recovery': 'passed',
                  'stop_exit_code': stopped['ExitCode'], 'host_ports': 'none'}
        if args.report_dir:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report))
    finally:
        try:
            if args.report_dir:
                args.report_dir.mkdir(parents=True, exist_ok=True)
                for name, owned in [('worker', worker), ('broker', broker)]:
                    if owned:
                        logs = subprocess.run(['docker', 'logs', owned], capture_output=True, text=True, timeout=20)
                        (args.report_dir / (name + '.log')).write_text(logs.stdout + logs.stderr, encoding='utf-8')
        finally:
            for owned in (worker, broker):
                if owned:
                    subprocess.run(['docker', 'rm', '--force', owned], capture_output=True, check=False, timeout=20)
            if network:
                subprocess.run(['docker', 'network', 'rm', network], capture_output=True, check=False, timeout=20)


if __name__ == '__main__':
    main()
