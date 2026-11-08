"""Explicit file-bootstrap entrypoint; normal main/CMD remain unchanged."""
import json
import sys
from typing import Callable, MutableMapping

from secret_bootstrap import load_secret_file_environment, SecretBootstrapError


def _start_application() -> None:
    # Import scientific/application modules only after file validation/injection.
    import asyncio
    from main import main
    asyncio.run(main())


def _report(error: dict[str, str]) -> None:
    print('[SECRET_BOOTSTRAP_ERROR] ' + json.dumps(error), file=sys.stderr)


def run_secret_file_bootstrap(
    start: Callable[[], None] | None = None,
    environment: MutableMapping[str, str] | None = None,
    report: Callable[[dict[str, str]], None] = _report,
) -> int:
    try:
        load_secret_file_environment(environment)
        (_start_application if start is None else start)()
    except SecretBootstrapError as error:
        report({'code': error.code, 'reason': error.reason})
        return 1
    except Exception:
        report({'code': 'APPLICATION_START_FAILED', 'reason': 'application-startup'})
        return 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(run_secret_file_bootstrap())
    except KeyboardInterrupt:
        pass
