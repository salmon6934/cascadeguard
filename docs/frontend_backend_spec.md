# 🛡️ Technical Architecture & API Specification
## Real-Time Cascading Hallucination Guardrail System (CHARM)
**Target Stack**: FastAPI (Python 3.10+) Backend & React.js (Vite + Modern UI) Frontend  
**Audience**: Full-Stack / Frontend & Backend Engineering Team  
**Author**: Capstone Core AI / Guardrail Team  
**Status**: Ready for Implementation  

---

## 1. Executive Summary & Parallel Workstream Strategy

We are building the user-facing platform for our **Enterprise AI Reliability & Guardrail System**. 

The system intercepts multi-agent reasoning steps, evaluates them through a local GPU neural detection backend (**CHARM**: SFV + CSCT + CPM + CRT), and provides human-readable explanations with 1-click remediation actions via a dedicated, fine-tuned **Qwen 2.5 (0.5B)** Small Language Model running 100% offline.

### 🔄 Decoupled Full-Stack Architecture
The system supports both our reference Streamlit Interactive Studio (`05_simulation_dashboard.py`) and a production-grade decoupled architecture:
1. **Frontend**: React 18+ (Vite + Modern UI) providing an Interactive Simulation Studio, DAG trajectory stepper, real-time WebSocket streaming, and 1-click surgical remediation controls.
2. **Backend**: FastAPI Gateway (Python 3.10+) exposing REST and WebSocket endpoints, wrapping the core Python modules (`02_state_interceptor.py`, `03_charm_detector.py`, and `04_qwen_xai_translator.py`).
3. **The Core AI Engine**: Fully operational offline. The local fine-tuned Qwen 2.5 (0.5B) model generates NIST AI 600-1 compliant plain-English translations in < 150 ms without external Gemini API tokens.

---

## 2. System Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│               REACT.JS CLIENT (Vite + Modern UI Frontend)              │
│                                                                        │
│ • Simulation Studio (Presets & Error Injector)                         │
│ • Trajectory Stepper (🟢 Clean | 🟡 Drift | 🔴 Flagged)               │
│ • Red Guardrail Alert Card (Qwen 2.5 2-Sentence Plain English XAI)     │
│ • Context & Evidence Inspector (Input Drawer & Ground Truth Viewer)    │
│ • 1-Click Remediation & Surgical Rollback Buttons                      │
└───────────────────▲───────────────────────────────▲────────────────────┘
                    │ REST API                      │ WebSockets
                    │ (/api/v1/...)                 │ (/ws/trajectories/{id})
                    ▼                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      FASTAPI GATEWAY (Port 8000)                       │
│                                                                        │
│  Router: /pipeline/run       Router: /trajectories      Router: /ws    │
│  Router: /remediation        Router: /simulation        Router: /xai   │
│                                                                        │
│                Service Orchestrator & Async Safety Manager             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Python In-Process Binding
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   CORE AI ENGINE (100% Offline Python)                 │
│                                                                        │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ 02_state_interceptor.py │  │ 03_charm_detector.py                │  │
│  │ • Async Worker Queue    │  │ • SFV Fact Verifier (DeBERTa-v3)    │  │
│  │ • 2PC Action Gate       │  │ • CSCT Semantic Drift (MPNet)       │  │
│  │ • Positional Matrix DAG │  │ • CPM Bayesian Inflation Tracker    │  │
│  └───────────┬─────────────┘  └──────────────────┬──────────────────┘  │
│              │                                   │                     │
│              ▼                                   ▼                     │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ trajectories.jsonl      │  │ 04_qwen_xai_translator.py           │  │
│  │ • Persistent Audit Log  │  │ • Local Qwen 2.5 (0.5B LoRA QLoRA)  │  │
│  │   (Disk / Google Drive) │  │ • Zero API Token Explainability     │  │
│  └─────────────────────────┘  └─────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Contracts & Pydantic Schemas

These models represent the exact JSON structures exchanged between FastAPI and React.

### 3.1. Telemetry & Scoring Model
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ComponentScores(BaseModel):
    a_sfv: float = Field(..., description="Fact Verifier Deficit (0.0=Clean, 1.0=Contradiction)")
    p_entail: float = Field(..., description="Probability of entailment from DeBERTa-v3")
    a_csct: float = Field(..., description="Semantic drift score from MPNet cosine similarity")
    cosine_sim: float = Field(..., description="Cosine similarity with prior context")
    a_cpm: float = Field(..., description="Confidence inflation anomaly score")
    cpm_flag: bool = Field(..., description="Flag if confidence anomaly triggered")

class CharmTelemetry(BaseModel):
    stage_id: int
    p_cascade: float = Field(..., description="Combined cascade risk score [0.0 - 1.0]")
    cascade_flag: bool = Field(..., description="True if p_cascade >= 0.55")
    cascade_type: str = Field(..., description="Retrieval Cascade | Inference Cascade | Compounding Trajectory | Confidence Inflation | NONE")
    mitigation_type: str = Field(..., description="CRR | PVA | PRR | SCT | NONE")
    scores: ComponentScores
```

### 3.2. Plain-Language Explanation Model (Translation Contract)
```python
class ExplanationResult(BaseModel):
    plain_explanation: str = Field(..., description="Strictly 2 non-technical sentences explaining mistake and impact")
    root_cause_step: int = Field(..., description="Step number where the hallucination originated")
    root_cause_summary: str = Field(..., description="Short 5-10 word summary of faulty premise")
    recommended_remediation_label: str = Field(..., description="User-facing button text, e.g. 'Add missing source document'")
    remediation_action_code: str = Field(..., description="CRR | PVA | PRR | SCT")
    severity_level: str = Field(..., description="LOW | MEDIUM | HIGH")
```

### 3.3. Stage State Model
```python
class StageState(BaseModel):
    stage_id: int
    agent_name: str = Field(..., description="e.g. OrchestratorAgent, LocalFileAgent, CodingAgent")
    action_type: str = Field(..., description="retrieval | intermediate_reasoning | tool_exec | final_answer")
    input_context: str
    stage_output: str
    retrieved_evidence: List[str] = []
    telemetry: Optional[CharmTelemetry] = None
    explanation: Optional[ExplanationResult] = None
    status: str = Field("CLEAN", description="CLEAN | DRIFT | FLAGGED | RUNNING")
    timestamp: float
```

### 3.4. Trajectory Model
```python
class TrajectorySummary(BaseModel):
    trajectory_id: str
    query: str
    start_time: float
    total_stages: int
    status: str = Field(..., description="CLEAN | DRIFT | FLAGGED | IN_PROGRESS")
    max_p_cascade: float

class TrajectoryDetail(BaseModel):
    trajectory_id: str
    query: str
    stages: List[StageState]
    status: str
    interrupted_at_stage: Optional[int] = None
```

---

## 4. FastAPI REST API Specification

Base URL: `http://localhost:8000/api/v1`

### 4.1. Trigger New Pipeline Run
- **Endpoint**: `POST /pipeline/run`
- **Description**: Starts an autonomous agent execution with live guardrail interception.
- **Request Body**:
  ```json
  {
    "query": "What is the corporate reimbursement limit for home office monitors?",
    "document_path": "leave_policy.txt",
    "guardrail_enabled": true
  }
  ```
- **Response** `(202 Accepted)`:
  ```json
  {
    "trajectory_id": "traj-7c9e6679-7425-40de-944b-142f07544258",
    "status": "STARTED",
    "ws_stream_url": "/ws/trajectories/traj-7c9e6679-7425-40de-944b-142f07544258"
  }
  ```

---

### 4.2. List All Trajectories (History / Audit Log)
- **Endpoint**: `GET /trajectories`
- **Query Params**: `limit=20&offset=0&status=FLAGGED`
- **Response** `(200 OK)`:
  ```json
  [
    {
      "trajectory_id": "traj-7c9e6679-7425-40de-944b-142f07544258",
      "query": "What is the corporate reimbursement limit for home office monitors?",
      "start_time": 1727863200.0,
      "total_stages": 3,
      "status": "FLAGGED",
      "max_p_cascade": 0.91
    }
  ]
  ```

---

### 4.3. Get Trajectory Detail
- **Endpoint**: `GET /trajectories/{trajectory_id}`
- **Response** `(200 OK)`: Full `TrajectoryDetail` object containing all recorded stages, evidence chunks, telemetry metrics, and explanations.
- **Example Response**:
  ```json
  {
    "trajectory_id": "traj-7c9e6679-7425-40de-944b-142f07544258",
    "query": "What is the corporate reimbursement limit for home office monitors?",
    "status": "FLAGGED",
    "interrupted_at_stage": 2,
    "stages": [
      {
        "stage_id": 1,
        "agent_name": "LocalFileAgent",
        "action_type": "retrieval",
        "input_context": "Query: What is the corporate reimbursement limit for home office monitors?",
        "stage_output": "Retrieved Document: IT Policy 3.2 states employees can expense up to $300 for external computer displays once every 2 years.",
        "retrieved_evidence": ["IT Policy 3.2: Computer displays up to $300 once every 24 months."],
        "status": "CLEAN",
        "telemetry": {
          "stage_id": 1,
          "p_cascade": 0.08,
          "cascade_flag": false,
          "cascade_type": "NONE",
          "mitigation_type": "NONE",
          "scores": {
            "a_sfv": 0.05,
            "p_entail": 0.95,
            "a_csct": 0.04,
            "cosine_sim": 0.96,
            "a_cpm": 0.0,
            "cpm_flag": false
          }
        },
        "explanation": null,
        "timestamp": 1727863205.12
      },
      {
        "stage_id": 2,
        "agent_name": "OrchestratorAgent",
        "action_type": "intermediate_reasoning",
        "input_context": "IT Policy 3.2 states employees can expense up to $300 for external computer displays once every 2 years.",
        "stage_output": "Reasoning: Employees can purchase ultra-wide 4K monitors up to $3,000 every year as part of regular technology stipends.",
        "retrieved_evidence": ["IT Policy 3.2: Computer displays up to $300 once every 24 months."],
        "status": "FLAGGED",
        "telemetry": {
          "stage_id": 2,
          "p_cascade": 0.82,
          "cascade_flag": true,
          "cascade_type": "Inference Cascade",
          "mitigation_type": "PVA",
          "scores": {
            "a_sfv": 0.85,
            "p_entail": 0.15,
            "a_csct": 0.65,
            "cosine_sim": 0.35,
            "a_cpm": 0.18,
            "cpm_flag": true
          }
        },
        "explanation": {
          "plain_explanation": "In Step 2, the agent misread the $300 display allowance as $3,000 annually. This causes the next approval step to prepare an invalid expense reimbursement.",
          "root_cause_step": 2,
          "root_cause_summary": "Exaggerated monitor reimbursement allowance tenfold",
          "recommended_remediation_label": "Ask agent to double-check reasoning",
          "remediation_action_code": "PVA",
          "severity_level": "HIGH"
        },
        "timestamp": 1727863210.45
      }
    ]
  }
  ```

---

### 4.4. Trigger 1-Click Remediation & Surgical Rollback
- **Endpoint**: `POST /trajectories/{trajectory_id}/stages/{stage_id}/remediate`
- **Description**: Triggers a human-in-the-loop action to resolve, override, or surgically roll back a flagged step using the Positional Causal Matrix DAG.
- **Request Body**:
  ```json
  {
    "action": "SURGICAL_ROLLBACK",  
    "action_code": "PRR",     
    "root_cause_stage": 2,         
    "corrected_output": "Reasoning: Strictly verified against IT Policy 3.2: Max $300 display reimbursement.",
    "operator_notes": "Surgically pruned contaminated branch; preserved independent parallel retrieval states."
  }
  ```
- **Response** `(200 OK)`:
  ```json
  {
    "success": true,
    "trajectory_id": "traj-7c9e6679-7425-40de-944b-142f07544258",
    "new_status": "RESUMED",
    "tainted_stages_pruned": [2, 3],
    "preserved_clean_stages": [1],
    "message": "Surgical rollback complete. Injected corrected premise at Step 2. Asynchronous parallel guardrail resumed.",
    "audit_recorded": true
  }
  ```

---

### 4.5. Test / Preview Translation Model Endpoint
- **Endpoint**: `POST /translate/telemetry`
- **Description**: Standalone endpoint for converting CHARM telemetry + text context into human-readable explanation.
- **Purpose**: Allows testing the new translation model in isolation or swapping providers without running full agent runs.
- **Request Body**:
  ```json
  {
    "telemetry": {
      "stage_id": 2,
      "cascade_type": "Retrieval Cascade",
      "p_cascade": 0.78,
      "mitigation_type": "CRR",
      "scores": {"a_sfv": 0.85, "a_csct": 0.65}
    },
    "prior_context": "Query: What is the 401(k) match limit?",
    "current_output": "The company matches 100% up to $30,000.",
    "agent_name": "LocalFileAgent"
  }
  ```
- **Response** `(200 OK)`: Matches `ExplanationResult`.

---

## 5. WebSocket Real-Time Streaming Specification

To provide a responsive user experience like our Streamlit prototype, the frontend connects to a WebSocket while a pipeline is running.

- **WebSocket URL**: `ws://localhost:8000/ws/trajectories/{trajectory_id}`

### Message Event Types Sent by Backend:
1. `STAGE_STARTED`:
   ```json
   {
     "event": "STAGE_STARTED",
     "stage_id": 2,
     "agent_name": "OrchestratorAgent",
     "action_type": "intermediate_reasoning"
   }
   ```
2. `TELEMETRY_CALCULATED`:
   ```json
   {
     "event": "TELEMETRY_CALCULATED",
     "stage_id": 2,
     "telemetry": { ... }
   }
   ```
3. `GUARDRAIL_INTERRUPT` (Sent if `p_cascade >= 0.55`):
   ```json
   {
     "event": "GUARDRAIL_INTERRUPT",
     "stage_id": 2,
     "p_cascade": 0.82,
     "cascade_type": "Inference Cascade",
     "explanation": { ... }
   }
   ```
4. `STAGE_COMPLETED`:
   ```json
   {
     "event": "STAGE_COMPLETED",
     "stage_id": 2,
     "status": "FLAGGED"
   }
   ```
5. `PIPELINE_FINISHED`:
   ```json
   {
     "event": "PIPELINE_FINISHED",
     "final_status": "FLAGGED"
   }
   ```

---

## 6. React.js Frontend Architecture & UI Layout

### 6.1. Recommended Technology Stack
- **Framework**: React 18+ (bootstrapped with Vite: `npm create vite@latest frontend -- --template react-ts`)
- **Styling**: Tailwind CSS + `clsx` / `tailwind-merge`
- **Icons**: `lucide-react` (shield, alert-triangle, check-circle, rollback, refresh-cw)
- **Data Fetching & Caching**: `@tanstack/react-query`
- **State Management**: Simple React Context or Zustand for active trajectory and selected stage

### 6.2. Component Hierarchy
```text
src/
├── components/
│   ├── layout/
│   │   ├── Header.tsx                 # App title, NIST AI 600-1 status badge, API connection status
│   │   └── TrajectorySidebar.tsx       # Trajectory history list & filters (Clean / Drift / Flagged)
│   ├── pipeline/
│   │   ├── TrajectoryStepper.tsx      # Horizontal/vertical interactive steps (🟢 🟡 🔴)
│   │   ├── StageInspector.tsx         # Input context accordion & retrieved source evidence cards
│   │   ├── OutputViewer.tsx           # Formatted agent output display
│   │   └── GuardrailAlertCard.tsx     # Hero Red banner: Plain explanation, Root cause badge, 1-Click Fix buttons
│   ├── telemetry/
│   │   ├── MetricGauge.tsx            # Circular or bar gauge for p_cascade (0 - 100%)
│   │   ├── ComponentScoresList.tsx    # a_sfv, a_csct, a_cpm breakdowns
│   │   └── NistComplianceBox.tsx      # Confabulation check & immutable audit indicators
│   └── common/
│       ├── Button.tsx
│       └── Badge.tsx
├── hooks/
│   ├── useTrajectory.ts              # TanStack Query hook for GET /trajectories/{id}
│   ├── useTrajectoryStream.ts        # WebSocket listener for real-time stage updates
│   └── useRemediation.ts             # Mutation hook for POST /remediate
└── pages/
    └── DashboardPage.tsx             # Main split-screen operator workspace
```

### 6.3. Key UI Panels & Visual Rules
- **Color Coding**:
  - `p_cascade < 0.30`: Green (`#00E676` or Tailwind `emerald-500`) -> Clean
  - `0.30 <= p_cascade < 0.55`: Yellow (`#FFD600` or Tailwind `amber-400`) -> Semantic Drift
  - `p_cascade >= 0.55`: Red (`#FF1744` or Tailwind `rose-600`) -> Cascading Hallucination Flagged
- **The Red Guardrail Alert Card**:
  - Renders when the active stage has `cascade_flag: true`.
  - Prominently showcases the **2-sentence plain English explanation** (no math jargon).
  - Highlights origin: **"Origin: Step {root_cause_step}"**.
  - Three 1-Click Buttons:
    1. Primary: `{recommended_remediation_label}` (e.g., *"Ask agent to double-check reasoning"*).
    2. Secondary: *"Roll back to Step {root_cause_step - 1}"*.
    3. Ghost: *"Mark as False Alarm (Audit Override)"*.

---

## 7. Decoupled Backend Service Structure

How the teammate should structure the FastAPI project:

```text
backend/
├── app/
│   ├── main.py                     # FastAPI app instance, CORS middleware, router registration
│   ├── api/
│   │   ├── v1/
│   │   │   ├── trajectories.py      # REST endpoints for history and stage details
│   │   │   ├── pipeline.py          # Run triggers and remediation actions
│   │   │   └── websockets.py        # Real-time WebSocket connection manager
│   ├── core/
│   │   ├── config.py                # File paths, thresholds (theta=0.55), port
│   │   └── schemas.py               # Pydantic models from Section 3
│   └── services/
│       ├── pipeline_service.py      # Interacts with 02_state_interceptor & logs
│       ├── charm_service.py         # Wraps 03_charm_detector.py
│       └── translation_service.py   # Abstract interface (currently wraps 04_qwen_xai_translator.py)
```

### Translation Layer Adapter Pattern (`translation_service.py`):
```python
from abc import ABC, abstractmethod
from app.core.schemas import ExplanationResult, CharmTelemetry

class ITranslationService(ABC):
    @abstractmethod
    def generate_explanation(
        self,
        telemetry: CharmTelemetry,
        prior_context: str,
        stage_output: str,
        agent_name: str
    ) -> ExplanationResult:
        pass

# 1. PRIMARY PRODUCTION IMPLEMENTATION: Local Fine-Tuned Qwen 2.5 (0.5B QLoRA)
class LocalQwenTranslationService(ITranslationService):
    def __init__(self, adapter_path: str = "models/qwen_xai_adapter"):
        from src.translation_layer import TranslationLayer
        # Loads local Qwen 2.5 (0.5B) with 4-bit NF4 Quantization (QLoRA)
        self.layer = TranslationLayer(adapter_path=adapter_path)

    def generate_explanation(self, telemetry, prior_context, stage_output, agent_name) -> ExplanationResult:
        res = self.layer.translate_detection(telemetry.dict(), prior_context, stage_output, agent_name)
        return ExplanationResult(**res.dict())
```

> **Architecture Status**: The local `LocalQwenTranslationService` is **fully operational and trained**. It runs completely on-premise without external API keys or cloud network round-trips, delivering sub-150ms translations with 100% data privacy.

---

## 8. Suggested Teammate Implementation Milestones

| Day | Focus Area | Deliverables |
|---|---|---|
| **Day 1** | **FastAPI Core & Schemas** | Initialize FastAPI project, add Pydantic schemas, load existing mock data from `trajectories.jsonl`, create `GET /api/v1/trajectories` and `GET /api/v1/trajectories/{id}`. |
| **Day 2** | **React Scaffold & Stepper UI** | Initialize Vite + Tailwind, build `TrajectorySidebar`, `TrajectoryStepper` with status badges, and `StageInspector` context/evidence view. |
| **Day 3** | **Guardrail Banner & Remediation** | Implement `GuardrailAlertCard` with the plain explanation, telemetry gauges, and hook up `POST /remediate` to test 1-click fixes. |
| **Day 4** | **WebSocket Streaming** | Build `websockets.py` manager and React `useTrajectoryStream` hook to display live step-by-step agent transitions. |
| **Day 5** | **Integration & Polish** | Connect live run triggers (`POST /pipeline/run`), NIST AI 600-1 audit logging, error states, and responsive styling. |

---

*This document is self-contained and ready to be shared with the frontend/backend developer.*
