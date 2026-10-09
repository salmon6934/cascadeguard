# ☁️ Google Colab Deployment & Reproduction Guide

## 1. Target Hardware & Prerequisites

* **Environment**: Google Colab
* **Hardware Accelerator**: **GPU (NVIDIA Tesla T4 16GB or A100 80GB)**
  *(In Colab: Navigate to `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ select `T4 GPU`)*
* **Disk Storage**: Google Drive persistent mount (~2 GB storage needed for models, datasets, and logs)
* **Python Runtime**: Python 3.10+

---

## 2. Directory Structure on Google Drive

Ensure the project folder is uploaded to the root of your Google Drive as `Capstone_Project`:

```
/content/drive/MyDrive/Capstone_Project/
├── src/
│   ├── 01_environment_setup.py
│   ├── 02_state_interceptor.py
│   ├── 03_charm_detector.py
│   ├── 04_qwen_xai_translator.py
│   ├── 05_simulation_dashboard.py
│   ├── 06_generate_charm_eval_dataset.py
│   ├── 07_evaluate_charm_performance.py
│   ├── 08_generate_qwen_training_dataset.py
│   └── 09_train_qwen.py
├── results/
│   ├── optimization_research_log.md
│   ├── qwen_training_report.md
│   └── qwen_training_metrics.json
└── requirements.txt
```

---

## 3. Cell-by-Cell Colab Execution Runbook

### Cell 1: Mount Google Drive & Install Requirements
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Capstone_Project
!pip install -q -r requirements.txt
```

---

### Cell 2: Phase 1 — Environment Initialization & GPU Verification
```python
!python src/01_environment_setup.py
```
* **Expected Output**:
  * Confirms CUDA availability and prints GPU name (`Tesla T4`, ~14.56 GB VRAM).
  * Automatically creates subdirectories: `logs/`, `models/`, `datasets/`, `workspace/`, `results/`, `user_study/`.
  * Validates native workspace sandboxing (`SUBPROCESS_OK`).

---

### Cell 3: Phase 2 — State Interceptor & Async Parallelism Smoke Test
```python
!python src/02_state_interceptor.py
```
* **Expected Output**:
  * Initializes the `StateInterceptor` middleware and daemon worker queue.
  * Validates non-blocking speculative agent handoff (~0 ms perceived latency).
  * Confirms thread-safe asynchronous abort tokens and Positional Causal Matrix graph tracking.

---

### Cell 4: Phase 3 — CHARM Neural Detection Backend Smoke Test
```python
!python src/03_charm_detector.py
```
* **Expected Output**:
  * Loads `cross-encoder/nli-deberta-v3-base` onto GPU memory (~400 MB).
  * Loads `sentence-transformers/all-mpnet-base-v2` onto GPU memory (~420 MB).
  * Evaluates simulated clean handoff: $p_{\text{cascade}} = 0.04$ (`cascade_flag = False`).
  * Evaluates simulated cascaded handoff: $p_{\text{cascade}} = 0.84$ (`cascade_flag = True`, `Remedy = PVA`).
  * Verifies Dual-Anchor Grounding ($E_{static}$) and Directional Entailment Gating ($P \ge 0.60$).

---

### Cell 5: Phase 8 & 9 — Dataset Generation & Qwen 2.5 Fine-Tuning
```python
# 1. Synthesize 1,200 multi-domain training samples (with 200 benign controls & formal splits)
!python src/08_generate_qwen_training_dataset.py

# 2. OPTIONAL: Execute 3-Epoch QLoRA Fine-Tuning (~31 minutes on T4 GPU)
# NOTE: If you already have 'models/qwen_xai_adapter' in Google Drive, SKIP this line!
!python src/09_train_qwen.py
```
* **What happens**:
  * `08` generates 1,200 multi-domain samples (4 failure classes $\times$ 250 + 200 `NOMINAL` clean controls) partitioned into `qwen_xai_train.jsonl` (960), `qwen_xai_val.jsonl` (120), and `qwen_xai_test.jsonl` (120).
  * `09` loads dedicated train/val partitions, activates 4-bit NF4 Quantization (QLoRA) with double quantization.
  * Trains 8.8M LoRA parameters across all linear projections.
  * Training loss smoothly converges from **3.4190 down to 0.0272**.
  * Saves trained adapter (~35 MB) to `/content/drive/MyDrive/Capstone_Project/models/qwen_xai_adapter`.

---

### Cell 6: Phase 4 — Local Qwen XAI Translation Layer Smoke Test
```python
!python src/04_qwen_xai_translator.py
```
* **Expected Output**:
  * Loads the newly trained LoRA adapter weights from `models/qwen_xai_adapter`.
  * Generates a sample 2-sentence plain-English explanation, root-cause localization, and prescribed remedy without calling external APIs.

---

### Cell 7A: Phase 5 — Launch the Streamlit Interactive Reliability Studio

Use this cell if you want to run the reference Streamlit Interactive Studio:

```python
# 1. Download and install Cloudflare Tunnel (one-time setup)
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb

# 2. Run Streamlit with CORS/proxy flags and expose via Cloudflare
!streamlit run src/05_simulation_dashboard.py \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false & \
cloudflared tunnel --url http://localhost:8501
```
* **How to access**: Look for the public URL printed in the output (e.g., `https://xxxx-xxxx-xxxx.trycloudflare.com`). Click it to open the dashboard immediately.

---

### Cell 7B: Workflow 1 — Launch the FastAPI Backend for the React Frontend

Use this cell to power the modern **React + TypeScript frontend** from Google Colab with full GPU neural acceleration (`DeBERTa-v3` + `MPNet` + `Qwen 2.5`):

```python
# ============================================================
# WORKFLOW 1: FASTAPI BACKEND + CLOUDFLARE QUICK TUNNEL
# (POWERS THE REACT + TYPESCRIPT FRONTEND)
# ============================================================
import os
import re
import time
import signal
import subprocess
from pathlib import Path
import requests

# 1. INSTALL BACKEND DEPENDENCIES
print("Installing FastAPI backend dependencies...")
!pip install -q fastapi uvicorn pydantic python-multipart pypdf python-docx

# 2. CLEAN OLD PROCESSES
print("\nCleaning old processes on port 8000...")
os.system("pkill -f 'uvicorn app.main:app' 2>/dev/null || true")
os.system("pkill -f cloudflared 2>/dev/null || true")
time.sleep(2)

# 3. VERIFY BACKEND APP PATH
backend_dir = Path("/content/drive/MyDrive/Capstone_Project/frontend+backend/backend")
if not backend_dir.exists():
    # Local fallback path check
    backend_dir = Path("frontend+backend/backend")
if not backend_dir.exists():
    raise FileNotFoundError(f"Backend directory not found at: {backend_dir.resolve()}")
print(f"Backend directory found at: {backend_dir.resolve()}")

# 4. START FASTAPI BACKEND
print("\nStarting FastAPI backend server on port 8000...")
backend_log = Path("/content/backend.log")
backend_process = subprocess.Popen(
    [
        "uvicorn",
        "app.main:app",
        "--app-dir", str(backend_dir),
        "--host", "0.0.0.0",
        "--port", "8000",
    ],
    stdout=open(backend_log, "w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
)

# 5. WAIT FOR FASTAPI HEALTH CHECK
print("Waiting for FastAPI server to become ready...")
server_ready = False
for attempt in range(30):
    time.sleep(1)
    try:
        res = requests.get("http://127.0.0.1:8000/api/health", timeout=3)
        if res.status_code == 200:
            data = res.json()
            server_ready = True
            print(f"✅ FastAPI is ready after {attempt + 1}s!")
            print(f"   * Neural Models Loaded: {data.get('modelsLoaded', False)}")
            print(f"   * Neural CHARM Active:  {data.get('neuralCharmActive', False)}")
            break
    except Exception:
        pass

if not server_ready:
    print("\n❌ FastAPI failed to start.\n" + "=" * 70 + "\nBACKEND LOG\n" + "=" * 70)
    if backend_log.exists():
        print(backend_log.read_text(encoding="utf-8", errors="replace"))
    raise RuntimeError("FastAPI failed to start. Tunnel was not attempted.")

# 6. DOWNLOAD CLOUDFLARED (IF NOT ALREADY PRESENT)
cloudflared_path = Path("/content/cloudflared")
if not cloudflared_path.exists():
    print("\nDownloading official Cloudflare Linux binary...")
    download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    subprocess.run(["curl", "-L", "--fail", "--silent", "-o", str(cloudflared_path), download_url], check=True)
    os.chmod(cloudflared_path, 0o755)
    print("✅ Cloudflared binary ready.")

# 7. START CLOUDFLARE TUNNEL FOR PORT 8000
print("\n" + "=" * 70 + "\nSTARTING CLOUDFLARE QUICK TUNNEL FOR FRONTEND\n" + "=" * 70)
tunnel_process = subprocess.Popen(
    [str(cloudflared_path), "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)

# 8. EXTRACT PUBLIC URL
public_url = None
url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
start_time = time.time()
while time.time() - start_time < 60:
    if tunnel_process.poll() is not None:
        break
    line = tunnel_process.stdout.readline()
    if not line:
        time.sleep(0.2)
        continue
    line = line.strip()
    match = url_pattern.search(line)
    if match:
        public_url = match.group(0)
        break

if public_url is None:
    raise RuntimeError("Cloudflare tunnel failed to generate a public URL.")

# 9. DISPLAY CONNECTION INSTRUCTIONS
print("\n" + "=" * 70)
print("✅ PUBLIC BACKEND URL CREATED:")
print(f"\n   👉 {public_url}\n")
print("=" * 70)
print("HOW TO CONNECT YOUR REACT FRONTEND:")
print("1. Open your React frontend (locally via 'npm run dev' or on Vercel).")
print("2. Click 'Connect' in the top navigation bar.")
print(f"3. Paste: {public_url}")
print("4. Click 'Save and connect'.")
print("=" * 70)
print("Keep this Colab cell running. (Stopping cell will stop the backend).")
print("=" * 70 + "\n")

# 10. KEEP TUNNEL & SERVER ALIVE
try:
    while tunnel_process.poll() is None and backend_process.poll() is None:
        time.sleep(5)
except KeyboardInterrupt:
    print("\nStopping backend and tunnel...")
    tunnel_process.send_signal(signal.SIGTERM)
    backend_process.send_signal(signal.SIGTERM)
    print("Processes stopped.")
```

---

### Cell 8: Phase 6 & 7 — Neural Adversarial Benchmark & CHARM Evaluation
```python
# 1. Synthesize balanced 50/50 benchmark via Local Qwen Perturber
!python src/06_generate_charm_eval_dataset.py

# 2. Run CHARM Evaluation & Statistical Significance over the benchmark
!python src/07_evaluate_charm_performance.py
```
* **Expected Output**:
  * `06` uses local Qwen 2.5 to synthesize 500 balanced benchmark trajectories (250 clean controls and 250 adversarial mutations across 15 domains) into `datasets/adversarial_benchmark.json`.
  * `07` automatically loads the GPU neural detector via `MultiSignalCharmDetector` (`cross-encoder/nli-deberta-v3-base` + `all-mpnet-base-v2`) and computes real tensor-based inference across all trajectories.
  * Calculates formal academic metrics: **CDR = 90.4%**, **FPR = 4.4%**, **EPR = 74.34%**, **CDD = 1.79 stages**, **LO/s = 215.0 ms**.
  * Executes Paired Bootstrap Hypothesis Test ($N=10,000$ resamples) confirming statistical significance ($p < 0.001$).
  * Writes evaluation summary and confusion matrix to `results/results.json`.

---

## 4. Common Colab Gotchas & Solutions

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `torch.OutOfMemoryError` | Unquantized FP32 weights loaded into VRAM | 4-bit NF4 quantization (`bitsandbytes`) is pre-configured in `09_train_qwen.py`. Peak VRAM is capped at **2.25 GB**, leaving ample headroom. |
| Localtunnel chunk drops (`TypeError: Failed to fetch...`) | Localtunnel proxy connection drops | Replaced with the direct **Cloudflare Quick Tunnel** in Cell 7, which handles dynamic JS chunks and WebSockets natively. |
| PEFT `torchao` import warning | Python 3.13 / PEFT compatibility check | Pre-patched in `04_qwen_xai_translator.py` and `09_train_qwen.py` with an automatic `torchao` bypass shim. |
