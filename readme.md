# MercurionTox21 🚀

MercurionTox21 is a **PyTorch** project for predicting chemical toxicity on the
**Tox21** dataset. A lightweight 🌟 multi‑layer perceptron (MLP) analyzes a
compound's SMILES string and estimates whether it triggers any of twelve
toxicological endpoints. The top four labels with the highest validation scores
are **SR‑ATAD5**, **NR‑AhR**, **SR‑MMP** and **SR‑p53**.

The repository also contains a production‑ready microservice that exposes the
model over a blazing‑fast 🔥 [NATS](https://nats.io/) messaging channel. The
service verifies RS256 access tokens and responds with per‑label probabilities
and thresholds.

---

## 🗂️ Repository layout

| Path                         | Description |
|------------------------------|-------------|
| `mercurion/model.py`        | Definition of the MLP network 🤖 |
| `mercurion/train.py`        | Training loop with early stopping and focal loss 🏋️ |
| `mercurion/preprocessing.py`| Converts the original CSV into NumPy tensors 🧬 |
| `data/raw/`                 | Original Tox21 CSV 📝 |
| `data/processed/`           | Vectorized `*.npy` files produced by preprocessing 💾 |
| `outputs/`                  | Saved models and thresholds after training 🏁 |
| `inference.py`              | NATS‑based inference microservice 📡 |

---

## 🛠️ Getting started

1. **Clone** the repository

   ```bash
   git clone https://github.com/your-org/MercurionTox21.git
   cd MercurionTox21
   ```

2. **Create a virtual environment** (optional but recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the requirements**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧬 Preparing the data

The repository ships with the original Tox21 CSV in `data/raw/tox21.csv`.
Convert it into train/validation/test tensors with:

```bash
python mercurion/preprocessing.py
```

This creates the `data/processed/` directory containing NumPy arrays for model
training.

---

## 🤖 Training the model

Run the training script to fit the MLP and compute optimal thresholds:

```bash
python mercurion/train.py
```

Key outputs:

* `outputs/models/best_model.pt` – parameters of the best model ✨
* `outputs/best_threshold.json` – global decision threshold 📏
* `outputs/best_thresholds.json` – per‑label thresholds for the four top labels 📊

You can adjust training parameters (epochs, learning rate, batch size) by
editing `mercurion/train.py`.

---

## 🔐 Generating an RSA key pair

The inference service validates JWT access tokens using **RS256**. Generate a
2048‑bit key pair with [OpenSSL](https://www.openssl.org/):

```bash
# Private key (keep this secret!)
openssl genpkey -algorithm RSA -out private.pem -pkeyopt rsa_keygen_bits:2048

# Public key (commit or deploy alongside the microservice)
openssl rsa -in private.pem -pubout -out public.pem
```

Place `public.pem` in the project root so `inference.py` can verify incoming
tokens. The private key is used by your authentication service to sign JWTs.

---

## 🚀 Starting inference in production

1. **Ensure a NATS server is running.** The code connects to
   `nats://localhost:4223`; adjust the URL in `inference.py` if your server uses a
   different host or port.

2. **Train the model** (see above) and keep `outputs/models/best_model.pt` and
   the threshold files in `outputs/`.

3. **Provide the public RSA key** as `public.pem` in the project root.

4. **Launch the microservice**:

   ```bash
   python inference.py
   ```

   The service subscribes to the subject `inference.tox21.smiles` and responds to
   requests with a JSON payload describing per‑label probabilities and whether
   each top label is positive according to its threshold. 📨

5. **Send a request** from your gateway or from the
   [`nats` CLI](https://github.com/nats-io/natscli):

   ```bash
   nats req inference.tox21.smiles '{"smiles": "CCO", "accessToken": "<jwt>"}'
   ```

---

## ❤️ Contributing

Pull requests, issues and ideas are welcome! Open an issue to discuss a feature
or bug, or fork the project and submit a PR. 🛠️

---

## 📜 License

This project is released under the MIT License. See `LICENSE` for details.

