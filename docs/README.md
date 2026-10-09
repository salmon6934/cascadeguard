# 🛡️ AI Agent Cascading Hallucination Guardrail & XAI System
## Comprehensive Documentation Hub

Welcome to the technical documentation repository for the **AI Agent Cascading Hallucination Guardrail & Explainable AI (XAI) System**. This documentation conforms to standard open-source production practices and provides an exhaustive reference for system architecture, edge model fine-tuning, empirical evaluation, and live demonstration.

---

### 📚 Documentation Directory

| Document | Purpose & Target Audience | Key Contents |
| :--- | :--- | :--- |
| **[`architecture.md`](file:///c:/harsh/CAPSTONE/docs/architecture.md)** | **System Architects & Researchers** | Multi-signal CHARM algorithm ($a_{sfv}, a_{csct}, a_{cpm}$), State Interceptor middleware, Asynchronous Speculative Execution, Two-Phase Commit (2PC) Action Gate, and Positional Causal Matrix DAG rollback. |
| **[`qwen_xai_model.md`](file:///c:/harsh/CAPSTONE/docs/qwen_xai_model.md)** | **ML Engineers & Model Evaluators** | Fine-tuning Qwen 2.5 (0.5B) via 4-bit NF4 QLoRA, parameter efficiency (1.75% weights), training loss dynamics (99.2% loss drop), Pydantic JSON schema enforcement, and sub-150ms inference. |
| **[`simulation_studio_guide.md`](file:///c:/harsh/CAPSTONE/docs/simulation_studio_guide.md)** | **Operators & Faculty Evaluators** | User manual for `05_simulation_dashboard.py`: Interactive Simulation Studio, multi-agent vertical data flow, error injection modes, and working closed-loop self-correction. |
| **[`colab_deployment_guide.md`](file:///c:/harsh/CAPSTONE/docs/colab_deployment_guide.md)** | **Developers & Reproducibility Leads** | Step-by-step execution guide on Google Colab (Tesla T4 GPU), Google Drive persistent storage, environment initialization, and Streamlit public tunneling. |
| **[`comprehensive_evaluation_report.md`](file:///c:/harsh/CAPSTONE/docs/comprehensive_evaluation_report.md)** | **Co-Authors & Academic Reviewers** | Quantitative benchmarking: Cascade Detection Rate (CDR $\ge 88\%$), False Positive Rate (FPR $\le 6\%$), Early Prevention Rate (EPR), Paired Bootstrap Hypothesis Testing ($p < 0.01$), XAI benchmarks, and System Usability Scale (SUS). |
| **[`frontend_backend_spec.md`](file:///c:/harsh/CAPSTONE/docs/frontend_backend_spec.md)** | **Full-Stack & API Developers** | Decoupled REST/WebSocket API specification (FastAPI + React.js), Pydantic contracts, and local Qwen translation adapter. |
| **[`viva_defense_faq.md`](file:///c:/harsh/CAPSTONE/docs/viva_defense_faq.md)** | **Oral Examination & Defense Prep** | Scripted answers for faculty review: addressing the "foolproof" question, defense-in-depth under NIST AI 600-1, edge vs. cloud trade-offs, and surgical rollback proofs. |

---

### 📂 Repository File Organization

```
c:/harsh/CAPSTONE/
├── docs/                                  # Centralized comprehensive documentation suite
│   ├── README.md                          # Master documentation portal (this file)
│   ├── architecture.md                    # Core algorithmic and software architecture
│   ├── qwen_xai_model.md                  # Qwen 2.5 SLM LoRA training and inference specs
│   ├── simulation_studio_guide.md         # Streamlit interactive UI and remediation manual
│   ├── colab_deployment_guide.md          # Google Colab T4 reproduction runbook
│   ├── comprehensive_evaluation_report.md # Formal empirical metrics, comparisons, and testing methodology
│   ├── frontend_backend_spec.md           # Decoupled FastAPI & React API specification
│   └── viva_defense_faq.md                # Viva voce oral defense questions and answers
├── src/                                   # Production-ready Python modules (Phase 1–9)
│   ├── 01_environment_setup.py            # Environment provisioning, GPU binding, sandbox
│   ├── 02_state_interceptor.py            # Interceptor middleware, async queue, causal DAG
│   ├── 03_charm_detector.py               # Local neural backend: SFV (DeBERTa) + CSCT (MPNet) + CPM
│   ├── 04_qwen_xai_translator.py          # Local Qwen 2.5 plain-English translation layer
│   ├── 05_simulation_dashboard.py         # Streamlit Interactive Simulation Studio & Log Inspector
│   ├── 06_generate_charm_eval_dataset.py  # Dual-engine neural adversarial perturber & balanced suite
│   ├── 07_evaluate_charm_performance.py   # Quantitative metric calculator & paired bootstrap tester
│   ├── 08_generate_qwen_training_dataset.py # 1,200 multi-domain ChatML dataset synthesizer
│   └── 09_train_qwen.py                   # PyTorch 4-bit NF4 QLoRA fine-tuning loop
├── results/                               # Empirical reports and training logs
│   ├── optimization_research_log.md       # Master optimization log with publication LaTeX tables
│   ├── qwen_training_report.md            # Formal Qwen training loss convergence report
│   └── qwen_training_metrics.json         # Raw step-by-step optimizer telemetry
├── requirements.txt                       # Pinned dependency manifest for pip installation
└── Autoagent.pdf / charm.pdf              # Academic foundational references
```

---

### 🚀 Recommended Reading Flow

1. **For System Overview**: Start with [`architecture.md`](file:///c:/harsh/CAPSTONE/docs/architecture.md) to understand how multi-agent handoffs are intercepted and evaluated.
2. **For Live Demonstrations**: Read [`simulation_studio_guide.md`](file:///c:/harsh/CAPSTONE/docs/simulation_studio_guide.md) to operate the Streamlit Interactive Studio.
3. **For Colab Deployment**: Follow [`colab_deployment_guide.md`](file:///c:/harsh/CAPSTONE/docs/colab_deployment_guide.md) to run the pipeline on a Tesla T4 GPU.
4. **For Viva & Paper Writing**: Review [`comprehensive_evaluation_report.md`](file:///c:/harsh/CAPSTONE/docs/comprehensive_evaluation_report.md) and [`viva_defense_faq.md`](file:///c:/harsh/CAPSTONE/docs/viva_defense_faq.md).
