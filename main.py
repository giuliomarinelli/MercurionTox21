import asyncio
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from api.rdkit import (
    get_molecule_properties,
    to_canonical_smiles,
    are_same_structure,
    InvalidSmilesError,
)
from api.pcp import CompoundNotFoundError, get_iupac_name_from_smiles
from config import ConfigurationError, get_config, nats_subject
from runtime import load_inference_runtime, load_public_key, read_build_identity
import signal
import json
from schemas.schemas import (
    InferenceRequest,
    MoleculePropertiesRequest,
    CanonicalSmilesRequest,
    CanonicalSmilesOptions,
    SameStructureRequest,
)
from pydantic import ValidationError
import jwt
from jwt.exceptions import PyJWTError
from time import time_ns
import sys
from typing import Any

GLOBAL_SKIP_AUTH_FLAG = True

PUBLIC_KEY: str | None = None

ALGORITHM = "RS256"


class ResponseDeliveryError(RuntimeError):
    """Transport failure is not an application error to send as a second response."""


def verify_jwt(token: str | None) -> dict[str, Any] | None:
    public_key = PUBLIC_KEY
    if not token or public_key is None:
        return None
    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[ALGORITHM],
            audience='mercurion-api'
        )
        return payload
    except PyJWTError:
        return None


def _extract_payload(msg: Msg) -> Any:
    raw = msg.data.decode()
    obj = json.loads(raw)
    return obj['data'] if isinstance(obj, dict) and 'data' in obj else obj


async def _respond(msg: Msg, payload: dict[str, Any]) -> None:
    # A plain NATS publish has no reply subject.
    if msg.reply:
        encoded = json.dumps(payload, allow_nan=False).encode()
        try:
            await msg.respond(encoded)
        except Exception:
            raise ResponseDeliveryError("NATS response delivery failed") from None


async def _respond_invalid_request(
    msg: Msg, error: ValidationError | json.JSONDecodeError | UnicodeDecodeError
) -> None:
    if isinstance(error, ValidationError):
        details = error.errors(include_input=False, include_url=False)
        message = f"Invalid request: {details}"
    else:
        message = "Invalid request: malformed JSON or UTF-8"
    await _respond(msg, {"error": message})


async def run(
    *, stop_event: asyncio.Event | None = None, ready_event: asyncio.Event | None = None
) -> None:
    global PUBLIC_KEY
    started = time_ns()
    stop = stop_event if stop_event is not None else asyncio.Event()
    ready = ready_event if ready_event is not None else asyncio.Event()
    ready.clear()
    config = get_config()
    build_identity = read_build_identity()
    PUBLIC_KEY = load_public_key(config.jwt_public_key_path, skip_auth=GLOBAL_SKIP_AUTH_FLAG)
    model, device, predict = load_inference_runtime(package_dir=config.model_package_dir)
    env = config.py_env
    nc = NATS()
    subscriptions_ready = False
    shutting_down = False
    unexpected_close = False

    async def disconnected() -> None:
        ready.clear()

    async def reconnected() -> None:
        if subscriptions_ready and not shutting_down and not stop.is_set():
            ready.set()

    async def closed() -> None:
        nonlocal unexpected_close
        ready.clear()
        if not shutting_down:
            unexpected_close = True
            stop.set()

    try:
        await nc.connect(
            config.nats_url, disconnected_cb=disconnected,
            reconnected_cb=reconnected, closed_cb=closed,
        )
        # =========================
        # INFERENCE CALLBACK
        # =========================
        async def inference_cb(msg: Msg) -> None:
            try:
                payload = _extract_payload(msg)
                req = InferenceRequest.model_validate(payload, context={"skip_auth": GLOBAL_SKIP_AUTH_FLAG})

                if not GLOBAL_SKIP_AUTH_FLAG and not verify_jwt(req.accessToken):
                    await _respond(msg, {"error": "Invalid or expired access token"})
                    return

                result = predict(req.smiles, model, device)
                await _respond(msg, result)

            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                await _respond_invalid_request(msg, e)
            except ResponseDeliveryError:
                raise
            except Exception:
                await _respond(msg, {"error": "InternalError"})

        nats_inference_ns = nats_subject("inference.tox21.smiles", env)

        await nc.subscribe(nats_inference_ns, cb=inference_cb)

        # =========================
        # PCP CALLBACKS
        # =========================

        async def iupac_name_cb(msg: Msg) -> None:
            try:
                payload = _extract_payload(msg)
                req = MoleculePropertiesRequest.model_validate(payload, context={"skip_auth": GLOBAL_SKIP_AUTH_FLAG})

                if not GLOBAL_SKIP_AUTH_FLAG and not verify_jwt(req.accessToken):
                    await _respond(msg, {"error": "Invalid or expired access token"})
                    return

                iupac_name = await asyncio.to_thread(get_iupac_name_from_smiles, req.smiles)
                await _respond(msg, {"data": {"iupac_name": iupac_name}})

            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                await _respond_invalid_request(msg, e)
            except CompoundNotFoundError:
                await _respond(msg, {"error": "Compound not found in PubChem"})
            except InvalidSmilesError:
                await _respond(msg, {"error": "Invalid SMILES"})
            except ResponseDeliveryError:
                raise
            except Exception:
                await _respond(msg, {"error": "InternalError"})

        # =========================
        # RDKIT CALLBACKS
        # =========================

        async def rdkit_props_cb(msg: Msg) -> None:
            try:
                payload = _extract_payload(msg)
                req = MoleculePropertiesRequest.model_validate(payload, context={"skip_auth": GLOBAL_SKIP_AUTH_FLAG})

                if not GLOBAL_SKIP_AUTH_FLAG and not verify_jwt(req.accessToken):
                    await _respond(msg, {"error": "Invalid or expired access token"})
                    return

                props = get_molecule_properties(req.smiles).to_dict()
                await _respond(msg, {"data": props})

            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                await _respond_invalid_request(msg, e)
            except InvalidSmilesError:
                await _respond(msg, {"error": "Invalid SMILES"})
            except ResponseDeliveryError:
                raise
            except Exception:
                await _respond(msg, {"error": "InternalError"})

        async def rdkit_canon_cb(msg: Msg) -> None:
            try:
                payload = _extract_payload(msg)
                req = CanonicalSmilesRequest.model_validate(payload, context={"skip_auth": GLOBAL_SKIP_AUTH_FLAG})
                opts = req.opts or CanonicalSmilesOptions()

                if not GLOBAL_SKIP_AUTH_FLAG and not verify_jwt(req.accessToken):
                    await _respond(msg, {"error": "Invalid or expired access token"})
                    return

                canon = to_canonical_smiles(
                    req.smiles,
                    isomeric=opts.isomeric,
                    kekule=opts.kekule,
                )
                await _respond(msg, {"data": canon})

            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                await _respond_invalid_request(msg, e)
            except InvalidSmilesError:
                await _respond(msg, {"error": "Invalid SMILES"})
            except ResponseDeliveryError:
                raise
            except Exception:
                await _respond(msg, {"error": "InternalError"})

        async def rdkit_same_cb(msg: Msg) -> None:
            try:
                payload = _extract_payload(msg)
                req = SameStructureRequest.model_validate(payload, context={"skip_auth": GLOBAL_SKIP_AUTH_FLAG})

                if not GLOBAL_SKIP_AUTH_FLAG and not verify_jwt(req.accessToken):
                    await _respond(msg, {"error": "Invalid or expired access token"})
                    return

                same = are_same_structure(req.a, req.b)
                await _respond(msg, {"data": same})

            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                await _respond_invalid_request(msg, e)
            except ResponseDeliveryError:
                raise
            except Exception:
                await _respond(msg, {"error": "InternalError"})

        nats_rdkit_props_ns = nats_subject("rdkit_api.get_molecule_properties", env)
        nats_rdkit_canon_ns = nats_subject("rdkit_api.to_canonical_smiles", env)
        nats_rdkit_same_ns = nats_subject("rdkit_api.are_same_structure", env)
        nats_iupac_name_ns = nats_subject("pcp_api.get_iupac_name_from_smiles", env)

        await nc.subscribe(nats_rdkit_props_ns, cb=rdkit_props_cb)
        await nc.subscribe(nats_rdkit_canon_ns, cb=rdkit_canon_cb)
        await nc.subscribe(nats_rdkit_same_ns, cb=rdkit_same_cb)
        await nc.subscribe(nats_iupac_name_ns, cb=iupac_name_cb)

        await nc.flush(timeout=5)
        subscriptions_ready = True
        if not nc.is_connected:
            raise ConnectionError("NATS disconnected during startup")
        ready.set()
        print(f"[MercurionTox21 > main] Ready in {(time_ns() - started) / 1_000_000:.1f}ms")
        print(f"[MercurionTox21 > main] Environment: {env}; device: {device}; version: {config.version}")
        print('[MercurionTox21 > main] Build identity: ' + json.dumps(build_identity, sort_keys=True))
        await stop.wait()
        if unexpected_close:
            raise ConnectionError("NATS connection closed")
    finally:
        shutting_down = True
        ready.clear()
        if not nc.is_closed:
            try:
                if nc.is_connected:
                    await asyncio.wait_for(nc.drain(), timeout=5)
            except Exception:
                print("[MercurionTox21 > main] Drain incomplete; closing connection", file=sys.stderr)
            finally:
                await asyncio.wait_for(nc.close(), timeout=5)


async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
            installed.append(signum)
        except NotImplementedError:
            # Windows uses asyncio.run's KeyboardInterrupt cancellation path.
            pass
    try:
        await run(stop_event=stop)
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as error:
        message = str(error) if isinstance(error, ConfigurationError) else type(error).__name__
        print(f"[MercurionTox21 > main] FATAL: {message}", file=sys.stderr)
        sys.exit(1)
