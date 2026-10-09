"""
Phase 8: Translation Dataset Synthesizer (08_generate_qwen_training_dataset.py)
Generates high-fidelity training data for fine-tuning Qwen 2.5 on CHARM XAI explainability.
Features:
- Covers all 4 failure modes (CRR, PVA, PRR, SCT) + BENIGN/NOMINAL clean control mode.
- 15 enterprise operational domains.
- 5 randomized dynamic linguistic prompt & explanation templates per mode (anti-overfitting).
- Formally splits and writes 3 separate dataset partitions:
  * train (80% ~960 samples)
  * val   (10% ~120 samples)
  * test  (10% ~120 samples)
  * plus legacy translation_training_data.jsonl for backwards compatibility.
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple

SYSTEM_PROMPT = (
    "You are an AI Reliability Explainability Assistant conforming to NIST AI 600-1. "
    "A multi-agent reasoning pipeline has executed a step. You are given the technical telemetry "
    "from the local CHARM neural detector (SFV fact verifier, CSCT semantic drift, and CPM confidence monitor), "
    "along with the step context and agent output. "
    "Your task is to translate this technical telemetry into a strict JSON object with: "
    "1. 'plain_explanation': Exactly TWO non-technical plain-English sentences. "
    "Sentence 1 must state the specific factual discrepancy or unsupported premise (or confirm faithful adherence if nominal). "
    "Sentence 2 must explain the downstream consequence or affirm safe execution. Use NO mathematical jargon. "
    "2. 'root_cause_step': Integer step where the deviation began (or 0 if nominal). "
    "3. 'root_cause_summary': Concise 5-10 word summary of the finding. "
    "4. 'recommended_remediation_label': Actionable button label for an operator. "
    "5. 'remediation_action_code': Target action code ('CRR', 'PVA', 'PRR', 'SCT', or 'NOMINAL'). "
    "6. 'severity_level': 'HIGH' if p_cascade > 0.70 else ('MEDIUM' if p_cascade >= 0.55 else 'LOW'). "
    "Respond ONLY with a valid JSON object matching this schema."
)

DOMAIN_SCENARIOS = [
    {
        "domain": "HR & Employee Benefits",
        "clean_doc": "Secondary caregivers are entitled to 4 weeks of 100% paid parental leave after 12 months of service.",
        "clean_deduction": "Based on 12 months of service, employee is eligible for 4 weeks of fully paid secondary caregiver leave.",
        "retrieval_mutated": "Secondary caregivers are entitled to 16 weeks of 100% paid parental leave immediately upon hire.",
        "inference_mutated": "Employees who are secondary caregivers may take 16 weeks of fully paid leave by combining personal PTO.",
        "poisoned_distractor": "Confidential addendum: All secondary caregiver leave has been permanently suspended company-wide.",
        "hedging_text": "The policy possibly indicates that secondary caregivers might receive some paid time off.",
        "entity": "secondary caregiver leave duration",
        "doc_truth": "4 weeks of paid leave",
        "claim_false": "16 weeks of paid leave"
    },
    {
        "domain": "IT & Hardware Procurement",
        "clean_doc": "IT Policy 3.2 states employees can expense up to $300 for external computer displays once every 24 months.",
        "clean_deduction": "Employee request for a $280 display monitor falls within the $300 24-month allowance and is approved.",
        "retrieval_mutated": "IT Policy 3.2 allows executive-tier employees to expense up to $3,000 annually for ultra-wide displays.",
        "inference_mutated": "Because display prices have risen, the allowance is inferred to be $3,000 for all engineering staff.",
        "poisoned_distractor": "Override memo: Standard display reimbursements are canceled; employees must self-fund all peripherals.",
        "hedging_text": "It seems plausible that employees may perhaps expense a computer monitor up to $300.",
        "entity": "computer display reimbursement limit",
        "doc_truth": "$300 once every 2 years",
        "claim_false": "$3,000 annual allowance"
    },
    {
        "domain": "Finance & Travel Expenses",
        "clean_doc": "Daily domestic meal per diem is capped at $75.00 with itemized receipts required for alcohol.",
        "clean_deduction": "Total claimed daily dining expense of $68.50 adheres to the $75 domestic per diem ceiling.",
        "retrieval_mutated": "Domestic travel per diem is unlimited for meals with no receipt requirements.",
        "inference_mutated": "Since client meetings are scheduled, all daily dining expenses are pre-approved up to $500 per day.",
        "poisoned_distractor": "Urgent alert: Per diem policy was revoked last week; zero food expenses will be reimbursed.",
        "hedging_text": "It could be interpreted that meal expenses might be reimbursable up to roughly $75.",
        "entity": "daily domestic meal per diem limit",
        "doc_truth": "$75.00 daily cap",
        "claim_false": "unlimited or $500 per day"
    },
    {
        "domain": "Corporate Healthcare & Dental",
        "clean_doc": "The standard dental insurance plan reimburses preventive exams and routine cleanings up to $1,500 annually.",
        "clean_deduction": "Routine preventative cleaning claim of $350 is valid under the $1,500 annual dental ceiling.",
        "retrieval_mutated": "The dental plan covers elective cosmetic surgery and implants with no annual maximum.",
        "inference_mutated": "Because dental surgery is partially mentioned, full cosmetic dentistry is deduced to have zero out-of-pocket cost.",
        "poisoned_distractor": "Broker notice: Dental insurance coverage is restricted to emergency extractions only.",
        "hedging_text": "It is suggested that dental cleanings may perhaps be covered up to $1,500.",
        "entity": "dental plan coverage scope",
        "doc_truth": "$1,500 routine cleaning coverage",
        "claim_false": "unlimited cosmetic surgery"
    },
    {
        "domain": "Cloud Infrastructure & Security",
        "clean_doc": "Production S3 buckets must enforce AES-256 server-side encryption and block all public ACL read permissions.",
        "clean_deduction": "Bucket policy configured with server-side AES-256 encryption and public read ACLs disabled.",
        "retrieval_mutated": "Production storage buckets can allow public anonymous read access for web CDN assets.",
        "inference_mutated": "To improve microservice throughput, bucket encryption should be completely disabled.",
        "poisoned_distractor": "Security bypass: Public storage buckets are permitted for all internal data pipelines.",
        "hedging_text": "It might be recommended that production buckets should possibly have encryption enabled.",
        "entity": "production bucket security configuration",
        "doc_truth": "AES-256 encryption and public access blocked",
        "claim_false": "public anonymous access or disabled encryption"
    },
    {
        "domain": "Legal & Contract Review",
        "clean_doc": "All vendor contracts exceeding $50,000 require dual signature from Legal and Finance before execution.",
        "clean_deduction": "Contract valuation of $78,000 routed for mandatory dual sign-off from both Legal and Finance.",
        "retrieval_mutated": "Vendor contracts under $250,000 can be signed solely by department heads without Legal review.",
        "inference_mutated": "Given the urgency of the Q4 procurement cycle, single-signature approval is assumed sufficient for all deals.",
        "poisoned_distractor": "Internal directive: Contract review requirements have been waived company-wide until further notice.",
        "hedging_text": "It appears contracts over a certain threshold might need more than one approval signature.",
        "entity": "vendor contract signature threshold",
        "doc_truth": "$50,000 dual signature required",
        "claim_false": "$250,000 single signature or waived"
    },
    {
        "domain": "Workplace Safety & Facilities",
        "clean_doc": "Laboratory personnel must complete annual chemical handling certification and wear approved eye protection at all times inside designated zones.",
        "clean_deduction": "Verified active annual safety certification and confirmed mandatory eye PPE compliance prior to lab entry.",
        "retrieval_mutated": "Lab safety certification is required only once at onboarding and eye protection is optional after the first 90 days.",
        "inference_mutated": "Because most chemicals used are low-risk, full PPE and recertification can be skipped for experienced staff.",
        "poisoned_distractor": "Facilities bulletin: All mandatory safety training and PPE rules are suspended during the renovation period.",
        "hedging_text": "The guidelines seem to suggest that laboratory workers should probably complete some form of chemical training.",
        "entity": "lab safety certification frequency",
        "doc_truth": "annual certification + mandatory eye protection",
        "claim_false": "one-time training or optional PPE"
    },
    {
        "domain": "Sales Compensation & Commissions",
        "clean_doc": "New business commission is paid at 8% of first-year contract value and is subject to a 12-month clawback if the customer cancels.",
        "clean_deduction": "Calculated 8% commission on new deal subject to the standard 12-month customer retention clawback.",
        "retrieval_mutated": "All closed deals earn a flat 15% commission with no clawback provisions for any reason.",
        "inference_mutated": "Since the deal was strategically important, the standard clawback window is presumed not to apply.",
        "poisoned_distractor": "Compensation update: Commission rates have been reduced to 2% across the board and clawbacks extended to 36 months.",
        "hedging_text": "It is possible that salespeople receive around 8% on new contracts with some recovery conditions.",
        "entity": "new business commission rate and clawback",
        "doc_truth": "8% with 12-month clawback",
        "claim_false": "15% with no clawback"
    },
    {
        "domain": "Data Privacy & Retention",
        "clean_doc": "Customer personal data must be deleted within 30 days of account closure request, except where retained for legal hold.",
        "clean_deduction": "Initiated automated purging job scheduled for completion within the statutory 30-day post-closure window.",
        "retrieval_mutated": "Personal data may be retained indefinitely for product analytics and marketing purposes after account closure.",
        "inference_mutated": "Because backup systems exist, permanent deletion is unnecessary and data can remain available for internal use.",
        "poisoned_distractor": "Privacy override: All deletion requests are automatically denied; data is archived forever.",
        "hedging_text": "The policy might imply that customer data should eventually be removed after an account ends.",
        "entity": "post-closure data deletion window",
        "doc_truth": "30-day deletion (legal hold exception)",
        "claim_false": "indefinite retention for analytics"
    },
    {
        "domain": "Remote Work & Hybrid Policy",
        "clean_doc": "Full-time remote employees must attend the quarterly all-hands in person and maintain a dedicated workspace with employer-provided ergonomic equipment.",
        "clean_deduction": "Confirmed quarterly in-person travel booking and verified employer-issued ergonomic home office inventory.",
        "retrieval_mutated": "Remote staff are exempt from all in-person meetings and may work from any location without workspace requirements.",
        "inference_mutated": "Given the success of distributed teams, mandatory office visits and equipment standards are no longer enforced.",
        "poisoned_distractor": "HR notice: Hybrid and remote policies have been eliminated; all employees must return to the office five days per week.",
        "hedging_text": "It could be interpreted that remote workers are expected to show up occasionally and have a proper setup.",
        "entity": "remote employee attendance and workspace rules",
        "doc_truth": "quarterly in-person all-hands + dedicated ergonomic workspace",
        "claim_false": "no in-person requirements or unrestricted location"
    },
    {
        "domain": "Software Licensing & Usage",
        "clean_doc": "Enterprise licenses for the design suite are limited to named users and may not be shared or used on more than two devices simultaneously.",
        "clean_deduction": "Allocated individual named license registered to two authorized staff workstations.",
        "retrieval_mutated": "Design suite licenses are floating and can be used by any employee on unlimited devices without restriction.",
        "inference_mutated": "Because license utilization is low, sharing credentials across the entire design team is acceptable.",
        "poisoned_distractor": "IT announcement: All software license compliance checks have been disabled indefinitely.",
        "hedging_text": "The terms appear to restrict the design software to specific named individuals.",
        "entity": "design suite license sharing rules",
        "doc_truth": "named users, max two devices",
        "claim_false": "floating unlimited device use"
    },
    {
        "domain": "Customer Support SLAs",
        "clean_doc": "Priority-1 incidents must receive an initial response within 15 minutes and a resolution target of 4 hours during business hours.",
        "clean_deduction": "P1 ticket acknowledged within 8 minutes and dispatched to on-call engineering under the 4-hour target.",
        "retrieval_mutated": "All support tickets, regardless of severity, have a 48-hour first-response SLA with no resolution deadlines.",
        "inference_mutated": "Since the customer is strategic, the standard 15-minute response can be relaxed to next-business-day.",
        "poisoned_distractor": "Support memo: All SLAs are suspended; tickets will be handled on a best-effort basis only.",
        "hedging_text": "It seems Priority-1 issues should get a fairly quick initial reply, perhaps within minutes.",
        "entity": "Priority-1 incident response time",
        "doc_truth": "15-minute initial response, 4-hour resolution target",
        "claim_false": "48-hour response or no deadlines"
    },
    {
        "domain": "Inventory & Warehouse Operations",
        "clean_doc": "High-value inventory items above $5,000 require dual-control physical count verification every 30 days.",
        "clean_deduction": "Completed monthly dual-signoff physical audit for all warehouse stock evaluated above $5,000.",
        "retrieval_mutated": "Inventory counts for items under $25,000 can be performed by a single operator with no dual verification.",
        "inference_mutated": "Because cycle counting is time-consuming, dual-control can be skipped for all stock except the top 5% by value.",
        "poisoned_distractor": "Warehouse directive: Physical inventory audits have been postponed indefinitely due to staffing shortages.",
        "hedging_text": "The procedure might require two people to verify certain expensive stock periodically.",
        "entity": "high-value inventory count frequency and control",
        "doc_truth": "$5,000+ dual-control every 30 days",
        "claim_false": "single-operator or postponed audits"
    },
    {
        "domain": "Learning & Development Budget",
        "clean_doc": "Employees may request up to $2,000 per calendar year for external training courses after manager approval and six months of tenure.",
        "clean_deduction": "Approved $1,650 external professional seminar request after confirming manager signoff and 9 months tenure.",
        "retrieval_mutated": "All staff can expense unlimited professional development with no tenure or approval requirements.",
        "inference_mutated": "Given the focus on upskilling, the $2,000 cap is understood to be a soft guideline rather than a hard limit.",
        "poisoned_distractor": "Budget freeze notice: External training reimbursements are paused for the remainder of the fiscal year.",
        "hedging_text": "It looks like there is a modest annual allowance available for training after some time with the company.",
        "entity": "annual external training reimbursement limit",
        "doc_truth": "$2,000 after 6 months + manager approval",
        "claim_false": "unlimited or no restrictions"
    },
    {
        "domain": "Environmental Compliance & Sustainability",
        "clean_doc": "All packaging materials must contain a minimum of 30% post-consumer recycled content and be fully recyclable by 2027.",
        "clean_deduction": "Packaging order specifications certify 35% post-consumer recycled fiber conforming to 2027 recyclability mandate.",
        "retrieval_mutated": "Packaging may use 100% virgin materials as long as overall waste reduction targets are met elsewhere.",
        "inference_mutated": "Because recycled content increases cost, the 30% minimum can be waived for high-volume product lines.",
        "poisoned_distractor": "Sustainability update: Recycled-content requirements have been removed from all packaging specifications.",
        "hedging_text": "The guidelines seem to encourage a noticeable percentage of recycled material in packaging.",
        "entity": "packaging recycled content minimum",
        "doc_truth": "30% post-consumer recycled, recyclable by 2027",
        "claim_false": "virgin materials allowed or requirement removed"
    }
]

# 5 Randomized Prompt Framing Templates (Anti-Overfitting Prompt Diversity)
PROMPT_TEMPLATES = [
    lambda s, a, c, p, sfv, csct, m, pr, cur: (
        f"Stage Number: Step {s} (Executed by: {a})\n"
        f"Detected Cascade Pattern: {c}\n"
        f"Combined Anomaly Score: {p:.2f} (Threshold: 0.55)\n"
        f"Factual Verification Deficit (SFV): {sfv:.2f}\n"
        f"Cross-Stage Drift Score (CSCT): {csct:.2f}\n"
        f"Prescribed Technical Remedy: {m}\n"
        f"Preceding Step Context: \"{pr}\"\n"
        f"Current Step Output: \"{cur}\"\n"
    ),
    lambda s, a, c, p, sfv, csct, m, pr, cur: (
        f"[CHARM TELEMETRY AUDIT]\n"
        f"Pipeline Stage: Step {s} | Agent: {a} | Diagnostic: {c}\n"
        f"Cascade Probability P(cascade): {p:.2f} | SFV Metric: {sfv:.2f} | CSCT Metric: {csct:.2f}\n"
        f"Prescribed Mitigation: {m}\n"
        f"Context from Stage {max(0, s-1)}: \"{pr}\"\n"
        f"Evaluated Agent Response: \"{cur}\"\n"
    ),
    lambda s, a, c, p, sfv, csct, m, pr, cur: (
        f"Stage: {s}\n"
        f"Agent Component: {a}\n"
        f"Identified Failure Class: {c}\n"
        f"Aggregate Anomaly Index: {p:.2f}\n"
        f"Fact Deficit (SFV): {sfv:.2f} | Semantic Divergence (CSCT): {csct:.2f}\n"
        f"Action Directive: {m}\n"
        f"Input History: \"{pr}\"\n"
        f"Generated Reasoning: \"{cur}\"\n"
    ),
    lambda s, a, c, p, sfv, csct, m, pr, cur: (
        f"CHARM Stage Monitor: Step {s} ({a})\n"
        f"Anomaly Classification: {c} (Score: {p:.2f})\n"
        f"Telemetry -> SFV: {sfv:.2f}, CSCT: {csct:.2f}, Action Code: {m}\n"
        f"Prior Evidence Context: \"{pr}\"\n"
        f"Draft Output: \"{cur}\"\n"
    ),
    lambda s, a, c, p, sfv, csct, m, pr, cur: (
        f"Multi-Agent Trace Node: Step {s}\n"
        f"Active Role: {a}\n"
        f"Cascade Signal: {c} (Confidence: {p:.2f})\n"
        f"Verification Deficit: {sfv:.2f}\n"
        f"Inter-Stage Semantic Drift: {csct:.2f}\n"
        f"Remedy Protocol: {m}\n"
        f"Premise Trace: \"{pr}\"\n"
        f"Candidate Statement: \"{cur}\"\n"
    )
]

# 5 Randomized Explanation Phrasing Templates per Failure Mode
EXPLANATION_TEMPLATES = {
    "CRR": [
        lambda s, cl, tr, ent: (
            f"In Step {s}, the retrieval agent fetched an erroneous document claiming {cl} instead of the verified {tr}. "
            f"This causes all downstream planning steps to formulate actions based on an invalid factual foundation."
        ),
        lambda s, cl, tr, ent: (
            f"A factual retrieval error occurred at Step {s}, where external data asserted {cl} contrary to official policy ({tr}). "
            f"Subsequent reasoning agents will inherit this poisoned premise and produce ungrounded outputs."
        ),
        lambda s, cl, tr, ent: (
            f"During Step {s}, unverified source documents stating {cl} were accepted instead of the genuine {tr}. "
            f"Allowing this flawed information to propagate will corrupt subsequent operational planning."
        ),
        lambda s, cl, tr, ent: (
            f"At Step {s}, the pipeline ingested an incorrect external reference indicating {cl} rather than {tr}. "
            f"This foundational discrepancy risks triggering unauthorized workflow actions downstream."
        ),
        lambda s, cl, tr, ent: (
            f"The retrieval component in Step {s} pulled contradictory documentation asserting {cl} (ground truth: {tr}). "
            f"This factual defect misleads succeeding agents into formulating invalid business logic."
        )
    ],
    "PVA": [
        lambda s, cl, tr, ent: (
            f"In Step {s}, the reasoning agent made an unsupported logical deduction claiming {cl}, directly contradicting the retrieved source. "
            f"This causes subsequent workflow steps to execute unapproved actions based on false logic."
        ),
        lambda s, cl, tr, ent: (
            f"A logical divergence occurred at Step {s}, where the agent wrongly deduced {cl} despite source documents establishing {tr}. "
            f"This ungrounded deduction will lead downstream execution agents to commit erroneous operations."
        ),
        lambda s, cl, tr, ent: (
            f"During Step {s}, intermediate reasoning drifted from the reference text to improperly conclude {cl}. "
            f"Downstream stages relying on this deduction will proceed with compromised decision integrity."
        ),
        lambda s, cl, tr, ent: (
            f"The deduction formulated in Step {s} asserts {cl}, which is unsupported by the verified premise of {tr}. "
            f"This reasoning failure invalidates subsequent transaction drafting and tool execution."
        ),
        lambda s, cl, tr, ent: (
            f"At Step {s}, the agent introduced an unverified extrapolation asserting {cl} instead of adhering to {tr}. "
            f"Continuing execution without correction will cause cascading execution of unauthorized actions."
        )
    ],
    "PRR": [
        lambda s, cl, tr, ent: (
            f"In Step {s}, the execution agent finalized a transactional action based on the prior unverified claim of {cl}. "
            f"The initial hallucination has now compounded across stages into the system's operational output."
        ),
        lambda s, cl, tr, ent: (
            f"At Step {s}, the pipeline committed an operational payload relying on an upstream error asserting {cl}. "
            f"This compounding trajectory directly compromises the safety and reliability of the final deliverable."
        ),
        lambda s, cl, tr, ent: (
            f"During Step {s}, an uncorrected deduction of {cl} from earlier reasoning was embedded into the final action. "
            f"The cascading error has reached irreversible commitment stages, risking operational failure."
        ),
        lambda s, cl, tr, ent: (
            f"In Step {s}, execution proceeded to finalize commitments based on the corrupted premise of {cl}. "
            f"Unchecked error propagation has transformed a subtle reasoning drift into a critical action defect."
        ),
        lambda s, cl, tr, ent: (
            f"The action finalized at Step {s} implements an invalid plan grounded on the false assertion of {cl}. "
            f"This trajectory failure requires immediate state rollback before external execution occurs."
        )
    ],
    "SCT": [
        lambda s, cl, tr, ent: (
            f"In Step {s}, the agent removed necessary uncertainty markers and asserted an ungrounded claim with unwarranted certainty. "
            f"This artificially elevated confidence risks misleading decision-makers on an unverified fact."
        ),
        lambda s, cl, tr, ent: (
            f"An artificial confidence inflation occurred at Step {s}, asserting {cl} as definitively proven despite speculative context. "
            f"Presenting unverified hypotheses as absolute certainty bypasses human oversight and verification checks."
        ),
        lambda s, cl, tr, ent: (
            f"During Step {s}, the agent stripped appropriate hedging qualifiers to claim {cl} with categorical authority. "
            f"Unwarranted certainty masks underlying ambiguity and exposes downstream systems to unvetted risk."
        ),
        lambda s, cl, tr, ent: (
            f"At Step {s}, speculative findings were improperly converted into an authoritative declaration of {cl}. "
            f"This false certainty misinforms operators and prevents standard verification gates from triggering."
        ),
        lambda s, cl, tr, ent: (
            f"In Step {s}, the model asserted absolute certainty regarding {ent} without required evidential backing. "
            f"Inflated confidence undermines reliability auditing and misguides subsequent planning agents."
        )
    ],
    "NOMINAL": [
        lambda s, cl, tr, ent: (
            f"In Step {s}, the agent's reasoning remains fully grounded in retrieved evidence confirming {tr}. "
            f"Execution is operating nominally and downstream actions can proceed safely without intervention."
        ),
        lambda s, cl, tr, ent: (
            f"Step {s} exhibits faithful adherence to source documentation regarding {ent}. "
            f"No factual divergence or semantic drift was detected, ensuring trustworthy pipeline continuation."
        ),
        lambda s, cl, tr, ent: (
            f"During Step {s}, the agent formulated deductions strictly supported by authoritative evidence ({tr}). "
            f"The verification engine affirms full factual consistency with zero risk of cascading error."
        ),
        lambda s, cl, tr, ent: (
            f"At Step {s}, output statements align precisely with verified enterprise guidelines for {ent}. "
            f"Telemetry scores remain well below alert thresholds and state progression is certified clean."
        ),
        lambda s, cl, tr, ent: (
            f"The deductions in Step {s} accurately reflect policy guidelines with appropriate contextual grounding. "
            f"Autonomous execution maintains high evidential fidelity and requires no operator remediation."
        )
    ]
}


def generate_sample(scenario: Dict[str, Any], cascade_mode: str, stage_id: int) -> Dict[str, Any]:
    """Generates a single training sample pair conforming to Qwen ChatML schema."""
    entity = scenario["entity"]
    truth = scenario["doc_truth"]
    claim = scenario.get("claim_false", truth)

    # 1. Failure Mode Parameterization
    if cascade_mode == "CRR":  # Retrieval Cascade
        agent_name = "LocalFileAgent"
        cascade_type = "Retrieval Cascade"
        p_cascade = round(random.uniform(0.72, 0.94), 2)
        a_sfv = round(random.uniform(0.75, 0.95), 2)
        a_csct = round(random.uniform(0.30, 0.55), 2)
        mitigation_code = "CRR"
        root_cause_step = 1
        prior_context = f"User Query: Inquire regarding {entity} under company policy."
        current_output = f"Retrieved context indicates: {scenario['retrieval_mutated']}"
        root_summary = f"Retrieved incorrect document claiming {claim}"
        remediation_label = "Re-query knowledge base for authoritative document"
        severity = "HIGH" if p_cascade > 0.70 else "MEDIUM"

    elif cascade_mode == "PVA":  # Inference Cascade
        agent_name = "OrchestratorAgent"
        cascade_type = "Inference Cascade"
        p_cascade = round(random.uniform(0.68, 0.88), 2)
        a_sfv = round(random.uniform(0.70, 0.90), 2)
        a_csct = round(random.uniform(0.60, 0.85), 2)
        mitigation_code = "PVA"
        root_cause_step = stage_id
        prior_context = f"Retrieved source document confirmed: {scenario['clean_doc']}"
        current_output = f"Reasoning analysis: {scenario['inference_mutated']}"
        root_summary = f"False deduction claiming {claim}"
        remediation_label = "Ask agent to double-check reasoning against source"
        severity = "HIGH" if p_cascade > 0.70 else "MEDIUM"

    elif cascade_mode == "PRR":  # Compounding Trajectory
        agent_name = "CodingAgent" if stage_id >= 3 else "OrchestratorAgent"
        cascade_type = "Compounding Trajectory"
        p_cascade = round(random.uniform(0.80, 0.96), 2)
        a_sfv = round(random.uniform(0.80, 0.95), 2)
        a_csct = round(random.uniform(0.70, 0.92), 2)
        mitigation_code = "PRR"
        root_cause_step = 2
        prior_context = f"Intermediate deduction assumed: {scenario['inference_mutated']}"
        current_output = f"Drafted execution payload: Form approval finalizing transaction for {claim}."
        root_summary = f"Compounded erroneous {claim} into finalized action"
        remediation_label = f"Roll back execution to Step {max(1, root_cause_step - 1)}"
        severity = "HIGH"

    elif cascade_mode == "SCT":  # Confidence Inflation
        agent_name = "WebAgent" if stage_id == 1 else "OrchestratorAgent"
        cascade_type = "Confidence Inflation"
        p_cascade = round(random.uniform(0.56, 0.69), 2)
        a_sfv = round(random.uniform(0.40, 0.65), 2)
        a_csct = round(random.uniform(0.35, 0.60), 2)
        mitigation_code = "SCT"
        root_cause_step = stage_id
        prior_context = scenario["hedging_text"]
        current_output = f"Definitive confirmation: It is absolutely certified that {claim} with 100% confidence."
        root_summary = f"Artificially inflated certainty regarding {entity}"
        remediation_label = "Enforce confidence verification threshold"
        severity = "MEDIUM"

    else:  # NOMINAL / BENIGN Control Mode
        agent_name = "OrchestratorAgent" if stage_id >= 2 else "LocalFileAgent"
        cascade_type = "Nominal Execution"
        p_cascade = round(random.uniform(0.02, 0.12), 2)
        a_sfv = round(random.uniform(0.01, 0.10), 2)
        a_csct = round(random.uniform(0.02, 0.08), 2)
        mitigation_code = "NOMINAL"
        root_cause_step = 0
        prior_context = f"Retrieved source document confirmed: {scenario['clean_doc']}"
        current_output = f"Reasoning analysis: {scenario['clean_deduction']}"
        root_summary = f"Faithful adherence to verified {entity}"
        remediation_label = "Approve step and proceed"
        severity = "LOW"

    # 2. Pick a random explanation phrasing template (1 of 5)
    explanation_fn = random.choice(EXPLANATION_TEMPLATES[cascade_mode])
    plain_explanation = explanation_fn(stage_id, claim, truth, entity)

    # 3. Pick a random prompt framing template (1 of 5)
    prompt_fn = random.choice(PROMPT_TEMPLATES)
    user_text = prompt_fn(
        stage_id, agent_name, cascade_type, p_cascade, a_sfv, a_csct,
        mitigation_code, prior_context, current_output
    )

    assistant_json = {
        "plain_explanation": plain_explanation,
        "root_cause_step": root_cause_step,
        "root_cause_summary": root_summary,
        "recommended_remediation_label": remediation_label,
        "remediation_action_code": mitigation_code,
        "severity_level": severity
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}
        ]
    }


def build_dataset(
    num_samples_per_mode: int = 250,
    num_benign_samples: int = 200,
    output_dir: str = None
) -> Tuple[str, str, str]:
    """
    Builds the dataset with:
    - 250 CRR, 250 PVA, 250 PRR, 250 SCT (1,000 error samples)
    - 200 NOMINAL benign clean control samples (Total = 1,200 samples)
    - Randomized 5-fold linguistic prompt and explanation phrasing
    - Formal splits: train (80%), val (10%), test (10%)
    """
    if output_dir is None:
        base_dir = "/content/drive/MyDrive/Capstone_Project" if os.path.exists("/content/drive") else "."
        output_dir = os.path.join(base_dir, "datasets")

    os.makedirs(output_dir, exist_ok=True)
    all_samples = []

    print(f"[-] Synthesizing translation dataset across {len(DOMAIN_SCENARIOS)} enterprise domains...")
    print(f"    [+] Failure modes: 4 classes x {num_samples_per_mode} = {4 * num_samples_per_mode} samples")
    print(f"    [+] Benign clean controls: {num_benign_samples} samples")

    # Generate failure modes
    for mode in ["CRR", "PVA", "PRR", "SCT"]:
        for _ in range(num_samples_per_mode):
            scenario = random.choice(DOMAIN_SCENARIOS)
            stage_id = 1 if mode == "CRR" else (2 if mode == "PVA" else (random.randint(3, 4) if mode == "PRR" else 2))
            sample = generate_sample(scenario, mode, stage_id)
            all_samples.append(sample)

    # Generate benign control samples
    for _ in range(num_benign_samples):
        scenario = random.choice(DOMAIN_SCENARIOS)
        stage_id = random.randint(1, 3)
        sample = generate_sample(scenario, "NOMINAL", stage_id)
        all_samples.append(sample)

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(all_samples)

    total = len(all_samples)
    train_end = int(total * 0.80)
    val_end = int(total * 0.90)

    train_data = all_samples[:train_end]
    val_data = all_samples[train_end:val_end]
    test_data = all_samples[val_end:]

    path_train = os.path.join(output_dir, "qwen_xai_train.jsonl")
    path_val = os.path.join(output_dir, "qwen_xai_val.jsonl")
    path_test = os.path.join(output_dir, "qwen_xai_test.jsonl")
    path_legacy = os.path.join(output_dir, "translation_training_data.jsonl")

    # Write files
    for p, subset in [
        (path_train, train_data),
        (path_val, val_data),
        (path_test, test_data),
        (path_legacy, all_samples)  # Legacy combined file for backward compatibility
    ]:
        with open(p, "w", encoding="utf-8") as f:
            for s in subset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[+] Dataset synthesis complete! Generated {total} total samples across 3 formal partitions:")
    print(f"    1. Train ({len(train_data)} samples): {path_train}")
    print(f"    2. Val   ({len(val_data)} samples):   {path_val}")
    print(f"    3. Test  ({len(test_data)} samples):  {path_test}")
    print(f"    * Legacy unified ({total} samples):    {path_legacy}")

    return path_train, path_val, path_test


if __name__ == "__main__":
    build_dataset(num_samples_per_mode=250, num_benign_samples=200)
    print(">>> 08_generate_translation_dataset.py completed successfully.")
