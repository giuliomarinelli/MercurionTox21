import asyncio
import torch    
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from api.inference import predict
from api.rdkit import (
    get_molecule_properties, 
    to_canonical_smiles, 
    are_same_structure,
    InvalidSmilesError,
)
from api.pcp import CompoundNotFoundError, get_iupac_name_from_smiles
from config import get_config
from mercurion.model import MercurionMLP
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

start_ns = time_ns()
print('[MercurionTox21 > main] Starting application...')

# 🔐 Hardening CPU: limitiamo i thread Torch ad 1 per evitare oversubscription
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)  # Non tutte le versioni di torch hanno questa API
except (AttributeError, TypeError):
    pass

config = get_config()

env = config.py_env or "development"
nats_url = config.nats_url or "nats://localhost:4223"
version = config.version or "unknown"

PUBLIC_KEY: str | None = None
if not GLOBAL_SKIP_AUTH_FLAG:
    try:
        with open(config.jwt_public_key_path, "r") as f:
            PUBLIC_KEY = f.read()
    except OSError as e:
        print(
            f"[MercurionTox21 > main] FATAL: unable to read JWT public key file "
            f"'{config.jwt_public_key_path}': {e}",
            file=sys.stderr,
        )
        print("\n[MercurionTox21 > main] Python process terminated with exit_code = 1\n")
        sys.exit(1)

ALGORITHM = "RS256"


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
        await msg.respond(json.dumps(payload).encode())


async def _respond_invalid_request(
    msg: Msg, error: ValidationError | json.JSONDecodeError | UnicodeDecodeError
) -> None:
    if isinstance(error, ValidationError):
        details = error.errors(include_input=False, include_url=False)
        message = f"Invalid request: {details}"
    else:
        message = "Invalid request: malformed JSON or UTF-8"
    await _respond(msg, {"error": message})


def _rdkit_ns(fn_name: str) -> str:
    if env != "production":
        return f"{env}.rdkit_api.{fn_name}"
    return f"rdkit_api.{fn_name}"

def _pcp_ns(fn_name: str) -> str:
    if env != "production":
        return f"{env}.pcp_api.{fn_name}"
    return f"pcp_api.{fn_name}"


# ✔️ Main NATS client
async def run() -> None:
    nc = NATS()
    await nc.connect(nats_url)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MercurionMLP().to(device)
    model.load_state_dict(
        torch.load(
            'outputs/models/best_model.pt',
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

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
        except Exception:
            await _respond(msg, {"error": "InternalError"})

    if env != 'production':
        nats_inference_ns = f"{env}.inference.tox21.smiles"
    else:
        nats_inference_ns = "inference.tox21.smiles"

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
        except Exception:
            await _respond(msg, {"error": "InternalError"})

    nats_rdkit_props_ns = _rdkit_ns("get_molecule_properties")
    nats_rdkit_canon_ns = _rdkit_ns("to_canonical_smiles")
    nats_rdkit_same_ns = _rdkit_ns("are_same_structure")
    nats_iupac_name_ns = _pcp_ns("get_iupac_name_from_smiles")

    await nc.subscribe(nats_rdkit_props_ns, cb=rdkit_props_cb)
    await nc.subscribe(nats_rdkit_canon_ns, cb=rdkit_canon_cb)
    await nc.subscribe(nats_rdkit_same_ns, cb=rdkit_same_cb)
    await nc.subscribe(nats_iupac_name_ns, cb=iupac_name_cb)
    
    stop_ns = time_ns()
    diff_ms = (stop_ns - start_ns) / 1000000

    
    print(f"[MercurionTox21 > main] Application started in: {diff_ms}ms")
    print(f"[MercurionTox21 > main] Environment: {env.upper()}")
    print(f"[MercurionTox21 > main] Device: {device.upper()}")
    print(f"[MercurionTox21 > main] NATS url: {nats_url}")
    print(f"[MercurionTox21 > main] Version: {version}")
    print(f"[MercurionTox21 > main > inference] ✅ Subscribed on '{nats_inference_ns}'...")
    print(f"[MercurionTox21 > main > rdkit] ✅ Subscribed on '{nats_rdkit_props_ns}'...")
    print(f"[MercurionTox21 > main > rdkit] ✅ Subscribed on '{nats_rdkit_canon_ns}'...")
    print(f"[MercurionTox21 > main > rdkit] ✅ Subscribed on '{nats_rdkit_same_ns}'...")
    print(f"[MercurionTox21 > main > rdkit] ✅ Subscribed on '{nats_iupac_name_ns}'...")


    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run())
