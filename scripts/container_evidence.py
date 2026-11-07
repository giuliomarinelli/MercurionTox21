"""Export an untagged QA archive, SBOM and fail-closed security evidence; never deploy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def archive_config(path):
    # No extraction: archive member names never become host filesystem paths.
    with tarfile.open(path) as archive:
        inventory = json.load(archive.extractfile('manifest.json'))
        if len(inventory) != 1 or inventory[0].get('RepoTags'):
            raise ValueError('Expected one untagged image; refusing tagged archive import')
        member = archive.getmember(inventory[0]['Config'])
        if not member.isfile() or member.size > 1024 * 1024:
            raise ValueError('Invalid image config member')
        raw = archive.extractfile(member).read()
        return 'sha256:' + hashlib.sha256(raw).hexdigest(), json.loads(raw)


def security_summary(report, config_digest, policy):
    if (policy.get('schema_version') != 1 or policy.get('blocking_severities') != ['CRITICAL'] or
            policy.get('exceptions') != []):
        raise ValueError('Unsupported policy or unauthorized exception')
    if (report.get('SchemaVersion') != 2 or report.get('Metadata', {}).get('ImageID') != config_digest or
            not isinstance(report.get('Results'), list) or not report['Results']):
        raise ValueError('Missing, malformed or mismatched security scan')
    findings = [finding for result in report['Results'] for finding in result.get('Vulnerabilities', [])]
    if any(finding.get('Severity') not in ['UNKNOWN', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] or
           not finding.get('VulnerabilityID') or not finding.get('PkgName') for finding in findings):
        raise ValueError('Malformed vulnerability finding')
    counts = Counter(finding.get('Severity', 'UNKNOWN') for finding in findings)
    return {'eligible': counts['CRITICAL'] == 0, 'counts': dict(counts),
            'blocking': [finding for finding in findings if finding.get('Severity') == 'CRITICAL'],
            'os': report.get('Metadata', {}).get('OS')}


def run(*args):
    return subprocess.check_output(list(args), text=True, timeout=1200).strip()


def export(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    policy_path = Path(__file__).resolve().parents[1] / 'docs/container-policy.json'
    policy = json.loads(policy_path.read_text())
    image = json.loads(run('docker', 'image', 'inspect', args.image))[0]
    labels = image['Config']['Labels']
    revision = labels['org.opencontainers.image.revision']
    version = labels['org.opencontainers.image.version']
    target = labels['org.mercurion.environment']
    # Run the existing full offline content/model/identity audit first.
    import sys
    audit = run(sys.executable, '-B', str(Path(__file__).with_name('check_container_image.py')),
                image['Id'], '--target', target, '--revision', revision, '--version', version)
    (root / 'runtime-audit.json').write_text(audit + '\n', encoding='utf-8')
    archive = root / 'image.tar'
    run('docker', 'save', '--output', str(archive), image['Id'])
    config_digest, config = archive_config(archive)
    mounts = ['--mount', f'type=bind,src={root},dst=/evidence', '--mount',
              f'type=bind,src={args.cache.resolve()},dst=/cache']
    args.cache.mkdir(parents=True, exist_ok=True)
    scanner = policy['scanner']
    for filename, flags in [('sbom.cdx.json', ['--format', 'cyclonedx']),
                            ('vulnerabilities.json', ['--scanners', 'vuln', '--format', 'json'])]:
        result = subprocess.run(['docker', 'run', '--rm', *mounts, scanner, 'image', '--quiet',
                                 '--input', '/evidence/image.tar', '--cache-dir', '/cache',
                                 '--output', '/evidence/' + filename, *flags],
                                capture_output=True, text=True, timeout=1200)
        (root / (filename + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        result.check_returncode()
    report = json.loads((root / 'vulnerabilities.json').read_text())
    summary = security_summary(report, config_digest, policy)
    (root / 'policy-result.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    # Transport verification is separate from security approval. Untagged input cannot replace user tags.
    loaded = run('docker', 'load', '--input', str(archive))
    (root / 'load.log').write_text(loaded + '\n', encoding='utf-8')
    import re
    identifiers = re.findall(r'Loaded image ID: (sha256:[0-9a-f]{64})', loaded)
    if len(identifiers) != 1:
        raise ValueError('Cannot identify loaded image')
    restored = json.loads(run('docker', 'image', 'inspect', identifiers[0]))[0]
    if (restored['Config']['Labels'] != config['config']['Labels'] or
            restored['RootFS']['Layers'] != config['rootfs']['diff_ids'] or
            restored['Architecture'] != config['architecture']):
        raise ValueError('Loaded image identity differs from archive')
    model = json.loads(run('docker', 'run', '--rm', '--network', 'none', '--read-only',
                          '--entrypoint', 'python', image['Id'], '-B', '-c',
                          "from pathlib import Path; print(Path('/app/model/manifest.json').read_text())"))
    files = {name: checksum(root / name) for name in ['image.tar', 'runtime-audit.json', 'sbom.cdx.json',
                                                    'vulnerabilities.json', 'policy-result.json', 'load.log']}
    manifest = {'schema_version': 1, 'component': 'tox21', 'status': 'candidate',
                'exported_at': datetime.now(timezone.utc).isoformat(), 'source': labels['org.opencontainers.image.source'],
                'version': version, 'revision': revision, 'target': target, 'docker_image_id': image['Id'],
                'config_digest': config_digest, 'archive_load_verified': True,
                'loaded_image_id': identifiers[0], 'scanner': scanner, 'policy_sha256': checksum(policy_path),
                'security_eligible': summary['eligible'], 'model': model, 'files': files}
    (root / 'component.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(root), 'config_digest': config_digest,
                      'security_eligible': summary['eligible'], 'counts': summary['counts']}))
    return 0 if summary['eligible'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    raise SystemExit(export(parser.parse_args()))
