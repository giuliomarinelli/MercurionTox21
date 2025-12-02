import asyncio
import torch
import numpy as np
from nats.aio.client import Client as NATS
from config import get_config
from mercurion.model import MercurionMLP
from mercurion.labels import tox21_labels
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import json
from schemas.schemas import InferenceRequest
from pydantic import ValidationError
from jose import jwt, JWTError
import os

# 🔐 Hardening CPU: limitiamo i thread Torch ad 1 per evitare oversubscription
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1) # Non tutte le versioni di torch hanno questa API
except (AttributeError, TypeError):
    pass

# ✔️ Label e indice: definiti una sola volta, in ordine
ALL_LABELS = tox21_labels
TOP4_LABELS = ['SR-ATAD5', 'NR-AhR', 'SR-MMP', 'SR-p53']
TOP4_INDICES = [ALL_LABELS.index(label) for label in TOP4_LABELS]

# ENV = os.getenv('PY_ENV', 'development')

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
    
with open("outputs/best_threshold.json", "r") as f:
    BEST_THRESHOLD = json.load(f)["best_threshold"]
    
with open("outputs/best_thresholds.json", "r") as f:
    PER_LABEL_THRESHOLDS = json.load(f)["per_label_thresholds"]

ALGORITHM = "RS256"

def verify_jwt(token: str):
    try:
        payload = jwt.decode(token, PUBLIC_KEY, algorithms=[ALGORITHM], audience='mercurion-api')
        return payload  
    
    except JWTError:
        return None

# ✔️ Preprocessing SMILES → fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol).ToBitString()
    return np.array([int(b) for b in fp], dtype=np.float32)

# ✔️ Inference function
def predict(smiles, model, device):
    fp = smiles_to_fp(smiles)
    if fp is None:
        return {'error': 'Invalid SMILES'}

    tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output).cpu().numpy().flatten()
        top4_probs = [float(probs[i]) for i in TOP4_INDICES]

    # Crea output con soglia per-label
    results = {}
    for i, label in enumerate(TOP4_LABELS):
        prob = top4_probs[i]
        threshold = PER_LABEL_THRESHOLDS[label]
        is_positive = prob > threshold
        results[label] = {
            "probability": prob,
            "is_positive": is_positive,
            "threshold": threshold 
        }
    return results


# ✔️ Main NATS client
async def run():
    nc = NATS()
    await nc.connect(nats_url)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MercurionMLP().to(device)
    model.load_state_dict(torch.load('outputs/models/best_model.pt', map_location=device, weights_only=True))
    model.eval()

    async def message_handler(msg):
        try:
            raw = msg.data.decode()
            obj = json.loads(raw)
            payload = obj['data'] if 'data' in obj else obj
            req = InferenceRequest.model_validate(payload)

            user_payload = verify_jwt(req.accessToken)
            if not user_payload:
                await msg.respond(json.dumps({"error": "Invalid or expired access token"}).encode())
                return

            result = predict(req.smiles, model, device)
            await msg.respond(json.dumps(result).encode())

        except ValidationError as e:
            await msg.respond(json.dumps({"error": f"Invalid request: {e.errors()}"}).encode())
        except Exception as e:
            await msg.respond(json.dumps({"error": "InternalError"}).encode())

    if env != 'production':
        nats_namespace = f"{env}.inference.tox21.smiles"
    else:
        nats_namespace = "inference.tox21.smiles"

    await nc.subscribe(nats_namespace, cb=message_handler)
    print(f"[MercurionTox21 > inference] Environment: {env.upper()}")
    print(f"[MercurionTox21 > inference] Device: {device.upper()}")
    print(f"[MercurionTox21 > inference] NATS url: {nats_url}")
    print(f"[MercurionTox21 > inference] Version: {version}")
    print(f"[MercurionTox21 > inference] ✅ Microservice subscribed on '{nats_namespace}'...")

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run())
