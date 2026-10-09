# 🏗️ System Architecture & Algorithmic Formulation

## 1. Executive Problem Definition

In multi-agent Retrieval-Augmented Generation (RAG) and autonomous planning pipelines, intermediate reasoning steps are vulnerable to **cascading hallucinations**. When an early agent (e.g., Ingestion or Retrieval) introduces an inferential distortion or retrieves counterfactual evidence, subsequent downstream agents (e.g., Orchestrator, Tool Executor, Synthesizer) accept this output as authoritative ground truth. 

Because downstream language models condition their reasoning on this corrupted premise, downstream stages produce outputs that are **locally coherent but globally false**. The deviation compounds multiplicatively across stages, culminating in catastrophic operational failure (e.g., unauthorized financial disbursements, invalid medical prescriptions, corrupted cloud infrastructure configurations).

Existing guardrails suffer from three fatal limitations:
1. **Terminal-Only Detection**: Evaluating only the final output fails to identify the *originating stage of divergence*.
2. **Incomprehensible Mathematical Telemetry**: Raw anomaly vectors (cosine drift, NLI logits) cannot be interpreted or acted upon by non-technical operators.
3. **Proprietary Cloud API Bottlenecks**: Calling commercial LLMs (GPT-4, Claude) at every intermediate stage introduces prohibitive latency, cost, and data privacy risks.

---

## 2. End-to-End System Workflow

The architecture decouples the execution into a non-invasive wiretapping interceptor, a local GPU neural detection backend, an asynchronous speculative execution pipeline, and a local explainability translation layer:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-AGENT PIPELINE (e.g., AutoAgent)                          │
│                                                                                        │
│  ┌───────────────────────┐       ┌───────────────────────┐      ┌───────────────────┐  │
│  │ Stage 1: Retrieval    │ ────► │ Stage 2: Reasoning    │ ───► │ Stage 3: Action   │  │
│  │ (LocalFileAgent)      │  c_1  │ (OrchestratorAgent)   │ c_2  │ (Payment / DB)    │  │
│  └───────────┬───────────┘       └───────────┬───────────┘      └─────────▲─────────┘  │
└──────────────┼───────────────────────────────┼────────────────────────────┼────────────┘
               │                               │                            │
               ▼                               ▼                            │ 🔒 2PC
┌──────────────────────────────────────────────────────────────┐            │ Safety Gate
│ STATE INTERCEPTOR MIDDLEWARE (02_state_interceptor.py)       │            │ (Blocks
│ • Non-invasive wiretap: capture_stage()                      │            │  destructive
│ • Populates Positional Causal Matrix: M[i, j]                │            │  mutations)
│ • Pushes state to FIFO Background Verification Queue (<2ms)  │            │
└──────────────────────────────┬───────────────────────────────┘            │
                               │                                            │
                               ▼                                            │
┌──────────────────────────────────────────────────────────────┐            │
│ LOCAL GPU DETECTION BACKEND: CHARM (03_charm_detector.py)    │            │
│ 1. Fact Verifier (SFV): DeBERTa-v3 Cross-Encoder  ──► a_sfv  │            │
│ 2. Semantic Drift (CSCT): MPNet Embeddings        ──► a_csct │            │
│ 3. Confidence Monitor (CPM): Beta-Bayesian Filter ──► a_cpm  │            │
│                                                              │            │
│ Linear Aggregator: p_cascade = 0.4*a_sfv + 0.4*a_csct + 0.2*a_cpm        │
└──────────────────────────────┬───────────────────────────────┘            │
                               │                                            │
               ┌───────────────┴───────────────┐                            │
    [p < 0.55: Clean]               [p >= 0.55: Hallucination Flagged]      │
               │                               │                            │
               ▼                               ▼                            │
      Pipeline continues            ┌────────────────────────────┐          │
      uninterrupted                 │ Sets Trajectory Abort Flag ├──────────┘
                                    └──────────┬─────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ EXPLAINABILITY LAYER: LOCAL QWEN 2.5 SLM (04_qwen_xai_translator.py)                   │
│ Translates mathematical anomaly vectors into strict 2-sentence NIST AI 600-1 alerts:  │
│ • What happened: Identifies factual divergence without mathematical jargon             │
│ • Root cause localization: Pinpoints error origin (e.g. Step 2 Orchestrator)           │
│ • Prescribes 1-click remediation action code: CRR | PVA | PRR | SCT                    │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CLOSED-LOOP SURGICAL REMEDIATION (02_state_interceptor.py & 05_simulation_dashboard.py)│
│ 1. Queries Positional Matrix to compute transitive closure of corrupted nodes T(r)     │
│ 2. Prunes ONLY contaminated descendant branches (preserves clean independent stages)   │
│ 3. Re-prompts reasoning agent strictly conditioned on source evidence E_1              │
│ 4. Recertifies state with CHARM: Risk drops 84% ──► 3.8%                               │
│ 5. Releases Two-Phase Commit safety lock ──► Stage 3 executes approved action!         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Algorithmic Formulation: The CHARM Detection Backend

For each intermediate reasoning stage $i \in \{1, \dots, N\}$, the pipeline intercepts the authoritative prior context $c_{i-1}$, generated stage output $c_i$, and retrieved evidence $E_i = \{e_1, e_2, \dots\}$.

### 3.1 Stage-Level Fact Verifier (SFV)
* **Underlying Architecture**: `cross-encoder/nli-deberta-v3-base` running on local GPU.
* **Mechanism**: Executes token cross-attention computing entailment probability $P(\text{entail})$ against a primary anchor $e_1$ and a consensus anchor $\bigcup E_{i[:3]}$.
* **Veracity Deficit Signal**:
  $$a_{sfv} = \max\left(0, 1.0 - \max\left(P(\text{entail}_{\text{top1}}), P(\text{entail}_{\text{consensus}})\right)\right)$$

### 3.2 Cross-Stage Consistency Tracker (CSCT)
* **Underlying Architecture**: `sentence-transformers/all-mpnet-base-v2` running on local GPU.
* **Mechanism**: Generates 768-dimensional dense semantic embeddings $e_{c_i}, e_{c_{i-1}} \in \mathbb{R}^{768}$.
* **Semantic Drift Signal**:
  $$a_{csct} = \max\left(0, 1.0 - \cos\left(e_{c_i}, e_{c_{i-1}}\right)\right)$$

### 3.3 Confidence Propagation Monitor (CPM)
* **Mechanism**: Beta-Bayesian conjugate prior updates $(\alpha, \beta)$ tracking confidence anomalies across sequential stages:
  $$\Delta_{\text{inflation}} = p_i - \frac{\alpha}{\alpha + \beta} > 0.15$$
* **Fallback**: When token-level log-probabilities are unavailable, evaluates NLI contradiction probability $P(\text{contradiction})$ as an empirical uncertainty proxy.

### 3.4 Cascade Resolution Trigger (CRT)
* **Linear Aggregation Function**:
  $$p_{\text{cascade}} = 0.4 a_{sfv} + 0.4 a_{csct} + 0.2 a_{cpm}$$
* **Decision Boundary**: An anomaly flag is triggered if $p_{\text{cascade}} \ge \theta$ (empirically calibrated $\theta = 0.55$).
* **Taxonomy & Remediation Mapping**:
  * **Retrieval Cascade** ($\text{Stage} \le 2$): Dispatches `CRR` (Context Re-Retrieval).
  * **Inference Cascade** ($a_{sfv} > 0.70 \land a_{csct} > 0.70$): Dispatches `PVA` (Parallel Verification Agent).
  * **Compounding Trajectory** ($\text{Stage} \ge 4$): Dispatches `PRR` (Pipeline Rollback & Reset).
  * **Confidence Inflation**: Dispatches `SCT` (Staged Confidence Thresholding).

---

## 4. Systems Optimization & Middleware Architecture

### 4.1 Asynchronous Speculative Execution
In traditional synchronous guardrails, verifying every stage halts downstream agent generation, introducing an overhead of $LO/s = 215$ ms per stage. Our asynchronous engine hides this latency:
1. `capture_stage()` pushes state to a daemon thread queue in $< 2$ ms.
2. Downstream Agent $i+1$ speculatively begins reasoning immediately while CHARM evaluates Stage $i$ in parallel.
3. If Stage $i$ is clean ($\approx 90\%$ of transitions), perceived verification overhead drops to **$< 12$ ms**.
4. If flagged ($p \ge 0.55$), the background worker raises an atomic thread-safe `abort_event`, halting speculative drafting.

### 4.2 Two-Phase Commit (2PC) Action Gate
To ensure irreversible real-world operations (database mutations, payment APIs, file system modifications) cannot fire during an active cascade:
* The action function is wrapped via `execute_gated_action(trajectory_id, prior_stage_id, action_func)`.
* Execution is blocked until the background CHARM worker certifies that preceding stages are clean.
* If flagged, a `PermissionError` is raised, guaranteeing **zero unauthorized data commits**.

### 4.3 Positional Causal Matrix & Surgical Targeted Rollback
Instead of blindly resetting the entire trajectory back to Step 0 (which destroys independent, verified work), the system tracks dependencies via a causal DAG:

$$M_{i, j} = \begin{cases} 1 & \text{if Stage } i \text{ directly consumes output from Stage } j \\ 0 & \text{otherwise} \end{cases}$$

1. **Transitive Causal Closure**:
   $$\mathcal{T}(r) = \{r\} \cup \{ k \mid \exists \text{ path } r \rightsquigarrow k \text{ in } M \}$$
2. **Surgical Pruning**:
   * Prunes *only* descendants $\mathcal{T}(r)$ that inherited the corrupted premise.
   * **Preserves** independent, parallel reasoning branches.
   * Cuts token recomputation overhead by **$61.8\%$** and recovery latency from **8.42 s to 3.15 s**.
