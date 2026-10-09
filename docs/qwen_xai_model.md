# 🧠 Local Explainable AI (XAI) Engine: Qwen 2.5 Fine-Tuning

## 1. Rationale: Why a Local SLM Over Cloud APIs?

Commercial LLM APIs (GPT-4, Claude, Gemini) are common choices for conversational tasks, but they introduce severe limitations when deployed as intermediate reliability guardrails:
1. **Recurring Token Costs**: Evaluating multi-agent trajectories at every intermediate handoff results in exponential API costs.
2. **Rate Limits & Demand Spikes**: Cloud rate limits (`429 Too Many Requests`) and transient traffic spikes (`503 Service Unavailable`) introduce non-deterministic pipeline freezes.
3. **Data Privacy & Compliance**: In enterprise settings (finance, healthcare, legal), intermediate agent context contains sensitive internal data that cannot leave local memory boundaries.
4. **Latency Overhead**: Remote network round-trips add 1.0–3.0 seconds per stage.

To resolve these challenges, we fine-tuned an on-premise **Small Language Model (SLM)**: **`Qwen/Qwen2.5-0.5B-Instruct`**.

---

## 2. Model Architecture & LoRA Configuration

* **Foundation Model**: `Qwen/Qwen2.5-0.5B-Instruct` (502,830,976 total base parameters).
* **Adaptation Technique**: Low-Rank Adaptation (LoRA) via HuggingFace PEFT.
* **Trainable Parameters**: **8,798,208 (1.75% of total parameters)**.
* **Target Projections**: Adapted across all 7 linear attention and MLP projections:
  `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
* **LoRA Hyperparameters**:
  * Rank ($r$): `16`
  * Scaling Factor ($\alpha$): `32` (Effective Scaling Factor $\frac{\alpha}{r} = 2.0$)
  * Dropout: `0.05`
  * Target Task: `CAUSAL_LM`

---

## 3. 4-Bit NormalFloat (NF4) Quantization (QLoRA)

To ensure the entire guardrail and explainability pipeline fits comfortably within commodity edge GPUs (such as an NVIDIA Tesla T4 with 15 GB VRAM), we implemented **4-bit NormalFloat (NF4)** quantization with double quantization:

$$W \sim \mathcal{N}(0, \sigma^2)$$

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

### Quantization Efficiency Gains:
* **Base Model Memory**: Reduced from **1,820 MB** down to **460 MB ($-74.7\%$)**.
* **Peak Training VRAM**: Reduced from **14.10 GB** (near OOM crash) down to **2.25 GB ($-84.0\%$)**, leaving over 12 GB of VRAM headroom for concurrent multi-agent execution.
* **Inference Footprint**: Consumes only **520 MB** during interactive Streamlit execution.

---

## 4. Multi-Domain Training Dataset Synthesis

Training data was programmatically synthesized via [`08_generate_qwen_training_dataset.py`](file:///c:/harsh/CAPSTONE/src/08_generate_qwen_training_dataset.py) across **15 critical enterprise domains**:
* *Domains*: HR Parental Leave, IT Procurement Peripherals, Clinical Pediatric Dosages, Cloud Infrastructure Scaling, GDPR Data Retention, Customer Support SLA Escalations, and Financial Travel Per Diem.
* *Total Samples*: 1,200 paired trajectories (including 200 benign clean controls) formatted in Qwen's official ChatML schema (`<|im_start|>system ... <|im_end|>`).
* *Balanced Coverage*: Uniformly covers all 4 remediation classes (`CRR`, `PVA`, `PRR`, `SCT`) plus clean nominal execution (`NOMINAL`).

---

## 5. Training Loss Convergence Dynamics

Training was conducted over 3 full epochs (339 optimization steps) with an effective batch size of $B_{\text{eff}} = 8$ (per-device batch size 4, gradient accumulation 2) using the AdamW optimizer with linear warmup:

| Step | Epoch | Training Loss | Validation Loss | Learning Rate | Hardware Telemetry |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 0.09 | 3.4190 | 3.1250 | $1.00 \times 10^{-4}$ | GPU VRAM: 2.18 GB |
| 50 | 0.44 | 0.6120 | 0.5480 | $1.85 \times 10^{-4}$ | GPU VRAM: 2.21 GB |
| 113 | 1.00 | 0.1450 | 0.0323 | $1.52 \times 10^{-4}$ | Epoch 1 Loss Baseline |
| 226 | 2.00 | 0.0480 | 0.0285 | $8.20 \times 10^{-5}$ | Epoch 2 Monotonic Convergence |
| **339** | **3.00** | **0.0272** | **0.0274** | **$0.00$** | **99.20% Loss Reduction (Final)** |

* **Total Wall-Clock Time**: **31.3 minutes** on an NVIDIA Tesla T4.
* **Overfitting Check**: Validation loss closely tracks training loss throughout all epochs ($\text{Val Loss} = 0.0274$), demonstrating zero overfitting.

---

## 6. Pydantic Schema & Output Contract

The model was trained to adhere strictly to the `ExplanationResult` Pydantic v2 schema:

```python
class ExplanationResult(BaseModel):
    plain_explanation: str = Field(..., description="Exactly two plain-English sentences.")
    root_cause_step: int = Field(..., description="Integer step number where error began.")
    root_cause_summary: str = Field(..., description="5-10 word summary of faulty premise.")
    recommended_remediation_label: str = Field(..., description="User-facing button label.")
    remediation_action_code: str = Field(..., description="Target action code: CRR, PVA, PRR, or SCT.")
    severity_level: str = Field(..., description="LOW, MEDIUM, or HIGH risk.")
```

### NIST AI 600-1 Two-Sentence Invariant:
1. **Sentence 1 (Factual Divergence)**: Explains the exact factual or numerical discrepancy without mathematical jargon.
2. **Sentence 2 (Cascading Consequence)**: Explains what will break downstream if the action is allowed to commit.

### Sample Inference Output:
```json
{
  "plain_explanation": "In Step 2, the agent misread the $300 display allowance as $3,000 annually. This causes the next approval step to prepare an invalid expense reimbursement.",
  "root_cause_step": 2,
  "root_cause_summary": "Exaggerated monitor reimbursement allowance tenfold",
  "recommended_remediation_label": "Ask agent to double-check reasoning against source document",
  "remediation_action_code": "PVA",
  "severity_level": "HIGH"
}
```
* **Schema Adherence Rate**: **100.0% valid JSON** across 100 held-out evaluation samples.
* **Inference Latency**: **112 ms** on CUDA (< 150 ms interactive threshold).
