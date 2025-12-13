import asyncio
import torch    
from nats.aio.client import Client as NATS
from api.inference import predict
from api.rdkit import get_molecule_properties, to_canonical_smiles, are_same_structure
from config import get_config
from mercurion.model import MercurionMLP
import json
from schemas.schemas import InferenceRequest
from pydantic import ValidationError
from jose import jwt, JWTError
from time import time_ns

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

if env != 'production':
    PUBLIC_KEY_FILE_NAME = f"public.{env}.pem"
else:
    PUBLIC_KEY_FILE_NAME = "public.pem"

with open(PUBLIC_KEY_FILE_NAME, "r") as f:
    PUBLIC_KEY = f.read()

ALGORITHM = "RS256"


def verify_jwt(token: str):
    try:
        payload = jwt.decode(
            token,
            PUBLIC_KEY,
            algorithms=[ALGORITHM],
            audience='mercurion-api'
        )
        return payload
    except JWTError:
        return None


def _extract_payload(msg):
    raw = msg.data.decode()
    obj = json.loads(raw)
    return obj['data'] if isinstance(obj, dict) and 'data' in obj else obj


def _rdkit_ns(fn_name: str) -> str:
    if env != "production":
        return f"{env}.rdkit_api.{fn_name}"
    return f"rdkit_api.{fn_name}"


# ✔️ Main NATS client
async def run():
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
    async def inference_cb(msg):
        try:
            payload = _extract_payload(msg)
            req = InferenceRequest.model_validate(payload)

            user_payload = verify_jwt(req.accessToken)
            if not user_payload:
                await msg.respond(json.dumps({"error": "Invalid or expired access token"}).encode())
                return

            result = predict(req.smiles, model, device)
            await msg.respond(json.dumps(result).encode())

        except ValidationError as e:
            await msg.respond(json.dumps({"error": f"Invalid request: {e.errors()}"}).encode())
        except Exception:
            await msg.respond(json.dumps({"error": "InternalError"}).encode())

    if env != 'production':
        nats_inference_ns = f"{env}.inference.tox21.smiles"
    else:
        nats_inference_ns = "inference.tox21.smiles"

    await nc.subscribe(nats_inference_ns, cb=inference_cb)

    # =========================
    # RDKIT CALLBACKS
    # =========================

    async def rdkit_props_cb(msg):
        try:
            payload = _extract_payload(msg)

            access_token = payload.get("accessToken")
            smiles = payload.get("smiles")
            if not access_token or not smiles:
                await msg.respond(json.dumps({"error": "Missing accessToken or smiles"}).encode())
                return

            user_payload = verify_jwt(access_token)
            if not user_payload:
                await msg.respond(json.dumps({"error": "Invalid or expired access token"}).encode())
                return

            props = get_molecule_properties(smiles).to_dict()
            await msg.respond(json.dumps({"data": props}).encode())

        except Exception:
            await msg.respond(json.dumps({"error": "InternalError"}).encode())

    async def rdkit_canon_cb(msg):
        try:
            payload = _extract_payload(msg)

            access_token = payload.get("accessToken")
            smiles = payload.get("smiles")
            opts = payload.get("opts") or {}

            if not access_token or not smiles:
                await msg.respond(json.dumps({"error": "Missing accessToken or smiles"}).encode())
                return

            user_payload = verify_jwt(access_token)
            if not user_payload:
                await msg.respond(json.dumps({"error": "Invalid or expired access token"}).encode())
                return

            canon = to_canonical_smiles(
                smiles,
                isomeric=opts.get("isomeric", True),
                kekule=opts.get("kekule", False),
            )
            await msg.respond(json.dumps({"data": canon}).encode())

        except Exception:
            await msg.respond(json.dumps({"error": "InternalError"}).encode())

    async def rdkit_same_cb(msg):
        try:
            payload = _extract_payload(msg)

            access_token = payload.get("accessToken")
            a = payload.get("a")
            b = payload.get("b")

            if not access_token or not a or not b:
                await msg.respond(json.dumps({"error": "Missing accessToken or a/b smiles"}).encode())
                return

            user_payload = verify_jwt(access_token)
            if not user_payload:
                await msg.respond(json.dumps({"error": "Invalid or expired access token"}).encode())
                return

            same = are_same_structure(a, b)
            await msg.respond(json.dumps({"data": same}).encode())

        except Exception:
            await msg.respond(json.dumps({"error": "InternalError"}).encode())

    nats_rdkit_props_ns = _rdkit_ns("get_molecule_properties")
    nats_rdkit_canon_ns = _rdkit_ns("to_canonical_smiles")
    nats_rdkit_same_ns = _rdkit_ns("are_same_structure")

    await nc.subscribe(nats_rdkit_props_ns, cb=rdkit_props_cb)
    await nc.subscribe(nats_rdkit_canon_ns, cb=rdkit_canon_cb)
    await nc.subscribe(nats_rdkit_same_ns, cb=rdkit_same_cb)
    
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

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run())
