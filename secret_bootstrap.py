"""Opt-in, bounded credential-file reader; standard library only, no import I/O."""
import json
import os
import stat
from typing import Mapping, MutableMapping

BOOTSTRAP_FILE_VARIABLE = 'MERCURION_BOOTSTRAP_FILE'
MAXIMUM_BOOTSTRAP_BYTES = 1024 * 1024
ALLOWED_VARIABLES = frozenset(('NATS_USERNAME', 'NATS_PASSWORD'))


class SecretBootstrapError(ValueError):
    code = 'INVALID_SECRET_BOOTSTRAP'

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__('Secret bootstrap rejected: ' + reason)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON field')
        value[key] = item
    return value


def _reject_constant(_value):
    raise ValueError('Non-JSON number')


def parse_secret_environment(data: bytes, target: Mapping[str, str]) -> dict[str, str]:
    if not data or len(data) > MAXIMUM_BOOTSTRAP_BYTES:
        raise SecretBootstrapError('file-size')
    try:
        document = json.loads(data.decode('utf-8-sig'), object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise SecretBootstrapError('invalid-document') from None
    if (not isinstance(document, dict) or set(document) != {'schemaVersion', 'environment', 'variables'}
            or type(document['schemaVersion']) is not int or document['schemaVersion'] != 1
            or not isinstance(document['variables'], dict) or not document['variables']):
        raise SecretBootstrapError('invalid-document')
    if (document['environment'] not in ('staging', 'production')
            or document['environment'] != target.get('PY_ENV')):
        raise SecretBootstrapError('environment-mismatch')
    prepared = {}
    for name, value in document['variables'].items():
        if name not in ALLOWED_VARIABLES:
            raise SecretBootstrapError('unknown-variable')
        if not isinstance(value, str) or not value.strip() or '\0' in value:
            raise SecretBootstrapError('invalid-value')
        try:
            value.encode('utf-8')
        except UnicodeError:
            raise SecretBootstrapError('invalid-value') from None
        if name in target:
            raise SecretBootstrapError('environment-conflict')
        prepared[name] = value
    if set(prepared) != ALLOWED_VARIABLES:
        raise SecretBootstrapError('incomplete-credentials')
    return prepared


def _read_secret_document(path: str) -> bytes:
    descriptor = None
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise SecretBootstrapError('file-type')
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SecretBootstrapError('file-type')
        if os.name != 'nt' and metadata.st_mode & 0o037:
            raise SecretBootstrapError('file-permissions')
        if metadata.st_size == 0 or metadata.st_size > MAXIMUM_BOOTSTRAP_BYTES:
            raise SecretBootstrapError('file-size')
        data = bytearray()
        while len(data) <= MAXIMUM_BOOTSTRAP_BYTES:
            chunk = os.read(descriptor, min(65536, MAXIMUM_BOOTSTRAP_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > MAXIMUM_BOOTSTRAP_BYTES:
            raise SecretBootstrapError('file-size')
        return bytes(data)
    except SecretBootstrapError:
        raise
    except (OSError, ValueError):
        raise SecretBootstrapError('file-unreadable') from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise SecretBootstrapError('file-unreadable') from None


def load_secret_file_environment(target: MutableMapping[str, str] | None = None) -> None:
    environment = os.environ if target is None else target
    path = environment.get(BOOTSTRAP_FILE_VARIABLE)
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        raise SecretBootstrapError('missing-file')
    prepared = parse_secret_environment(_read_secret_document(path), environment)
    environment.update(prepared)
