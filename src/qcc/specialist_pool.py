"""Specialist Pool — Configurable pool of Specialist Agents with optional RAG,
plus an LLM-based Router Agent that selects top-3 specialists from S_obs.

Paper reference — Sec 2.2, "Specialist-Guided Planning":

  EDPM maintains a candidate pool of specialist agents:
    A = {Acard, Aneuro, Apulm, Aortho, ...}

  Given S_obs, the system computes a relevance score for each specialist:
    Score(Ai | Sobs) = ψ(Sobs, KAi)

  and selects the best-matching specialist(s) as Planner(s).

Architecture:

  SpecialistAgent — Lightweight agent wrapping specialist metadata + optional RAG.
                     Each SpecialistAgent holds a KnowledgeRetriever (optional).
  SpecialistPool   — Configurable container of SpecialistAgent instances.
  RouterAgent      — LLM-based agent that takes S_obs and returns top-3 specialists.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from qcc.weci import KnowledgeRetriever

logger = logging.getLogger("specialist_pool")


# =============================================================================
# RAG import helper (avoid circular imports)
# =============================================================================

def _get_default_retriever() -> "KnowledgeRetriever":
    """Lazy-import NoOpRetriever to avoid circular dependency at module level."""
    from qcc.weci import NoOpRetriever
    return NoOpRetriever()


# =============================================================================
# Specialist Agent
# =============================================================================

@dataclass
class SpecialistAgent:
    """A specialist agent in the EDPM candidate pool.

    Each specialist has:
    - Metadata: id, name, clinical focus, key evidence types
    - Optional RAG: a KnowledgeRetriever that provides specialty-specific
      medical knowledge. When None, the specialist relies purely on its
      prompt + LLM capability.
    - System prompt: built from metadata + optional RAG context.
    """

    specialist_id: str
    name: str
    focus: str
    key_evidence: str
    retriever: "KnowledgeRetriever | None" = None
    prompt_suffix: str = ""

    def get_system_prompt(self, base_prompt: str | None = None) -> str:
        """Build the specialist's system prompt.

        Parameters
        ----------
        base_prompt : str or None
            A generic Planner system prompt to augment.  When None,
            returns only the specialist context.

        Returns
        -------
        str
            Specialist-augmented system prompt.
        """
        lines = []
        if base_prompt:
            lines.append(base_prompt)

        lines.append("\n[SPECIALIST_CONTEXT]")
        lines.append(f"You are acting as a **{self.name}** specialist.")
        lines.append(f"Domain focus: {self.focus}")
        lines.append(f"Key evidence in your specialty: {self.key_evidence}")
        lines.append(
            f"Frame your diagnostic directions within {self.name}. "
            "Keep an open mind to cross-system possibilities, but prioritize "
            "investigations most relevant to your specialty's expertise."
        )

        if self.retriever is not None:
            rag_context = self._retrieve_knowledge()
            if rag_context:
                lines.append(f"\n[SPECIALIST_RAG_KNOWLEDGE]\n{rag_context}")

        if self.prompt_suffix:
            lines.append(f"\n{prompt_suffix}")

        return "\n".join(lines)

    def _retrieve_knowledge(self) -> str:
        """Call the retriever for specialty-relevant knowledge.

        Uses the specialist's focus as the query since no case-specific
        diagnosis/evidence is available at construction time.
        """
        if self.retriever is None:
            return ""
        try:
            return self.retriever.retrieve(
                diagnosis=self.name,       # specialty as query anchor
                evidence=self.key_evidence,
            )
        except Exception as exc:
            logger.warning(
                "RAG retrieval failed for %s: %s", self.specialist_id, exc
            )
            return ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise metadata (retriever is not serialisable)."""
        return {
            "specialist_id": self.specialist_id,
            "name": self.name,
            "focus": self.focus,
            "key_evidence": self.key_evidence,
            "has_retriever": self.retriever is not None,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        retriever: "KnowledgeRetriever | None" = None,
    ) -> "SpecialistAgent":
        return cls(
            specialist_id=d["id"],
            name=d["name"],
            focus=d.get("focus", ""),
            key_evidence=d.get("key_evidence", ""),
            retriever=retriever,
            prompt_suffix=d.get("prompt_suffix", ""),
        )


# =============================================================================
# Specialist Pool
# =============================================================================

# Default specialist definitions (no RAG attached — users call attach_retriever()
# or pass retrievers via from_dicts).
DEFAULT_SPECIALIST_DEFINITIONS: list[dict[str, str]] = [
    {
        "id": "cardiology",
        "name": "Cardiology",
        "focus": "cardiovascular system — chest pain, arrhythmias, heart failure, "
        "valvular disease, pericardial conditions, vascular emergencies",
        "key_evidence": "ECG, troponin, BNP, echocardiogram, chest X-ray, blood pressure",
    },
    {
        "id": "neurology",
        "name": "Neurology",
        "focus": "nervous system — headache, seizure, stroke, altered mental status, "
        "neuropathy, movement disorders, vertigo",
        "key_evidence": "CT head, MRI brain, lumbar puncture, EEG, neurological exam",
    },
    {
        "id": "pulmonology",
        "name": "Pulmonology",
        "focus": "respiratory system — dyspnoea, cough, haemoptysis, respiratory failure, "
        "pneumonia, COPD, asthma, pulmonary embolism",
        "key_evidence": "chest X-ray, CT chest, ABG, spirometry, sputum culture, D-dimer",
    },
    {
        "id": "gastroenterology",
        "name": "Gastroenterology",
        "focus": "digestive system — abdominal pain, GI bleeding, nausea/vomiting, "
        "diarrhoea, jaundice, liver disease, pancreatitis",
        "key_evidence": "abdominal CT, ultrasound, LFTs, lipase, endoscopy, stool studies",
    },
    {
        "id": "nephrology",
        "name": "Nephrology",
        "focus": "renal system — acute kidney injury, electrolyte disorders, "
        "haematuria, proteinuria, acid–base disturbances, urinary tract infection",
        "key_evidence": "BUN, creatinine, eGFR, urinalysis, urine electrolytes, renal ultrasound",
    },
    {
        "id": "infectious_disease",
        "name": "Infectious Disease",
        "focus": "infections — fever of unknown origin, sepsis, meningitis, "
        "endocarditis, tropical diseases, antibiotic selection",
        "key_evidence": "blood cultures, CRP, procalcitonin, WBC differential, specific serologies",
    },
    {
        "id": "orthopedics",
        "name": "Orthopedics",
        "focus": "musculoskeletal system — fractures, joint pain, back pain, "
        "trauma, compartment syndrome, septic arthritis",
        "key_evidence": "X-ray, CT, MRI, joint aspiration, CK, inflammatory markers",
    },
    {
        "id": "endocrinology",
        "name": "Endocrinology",
        "focus": "endocrine system — diabetes, thyroid disorders, adrenal "
        "insufficiency, electrolyte imbalances from hormonal causes",
        "key_evidence": "blood glucose, HbA1c, TSH, cortisol, electrolytes",
    },
    {
        "id": "hematology",
        "name": "Hematology",
        "focus": "blood disorders — anaemia, bleeding disorders, thrombosis, "
        "leukaemia, lymphoma, transfusion reactions",
        "key_evidence": "CBC, coagulation panel, peripheral smear, bone marrow biopsy, LDH",
    },
    {
        "id": "general_emergency",
        "name": "General Emergency Medicine",
        "focus": "undifferentiated emergency presentations — multi-system complaints, "
        "trauma, toxicology, shock, cardiac arrest, environmental emergencies",
        "key_evidence": "vital signs, point-of-care ultrasound, ECG, basic labs, toxicology screen",
    },
]


@dataclass
class SpecialistPool:
    """Configurable pool of SpecialistAgent instances.

    Usage::

        # Bare pool (no RAG)
        pool = SpecialistPool.default()

        # Attach a shared RAG backend to all specialists
        from my_rag import PubMedRetriever
        pool.attach_retriever(PubMedRetriever())

        # Attach per-specialist retrievers
        pool.attach_retriever_to("cardiology", CardioRetriever())
    """

    specialists: list[SpecialistAgent] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_definitions(
        cls,
        definitions: list[dict[str, Any]],
        retrievers: dict[str, "KnowledgeRetriever"] | None = None,
    ) -> "SpecialistPool":
        """Build pool from raw dict definitions with optional per-specialist RAG.

        Parameters
        ----------
        definitions : list[dict]
            Raw specialist definitions.
        retrievers : dict or None
            Mapping specialist_id → KnowledgeRetriever.
        """
        retrievers = retrievers or {}
        specialists = []
        for d in definitions:
            sid = d["id"]
            specialists.append(
                SpecialistAgent.from_dict(d, retriever=retrievers.get(sid))
            )
        return cls(specialists=specialists)

    @classmethod
    def default(cls) -> "SpecialistPool":
        """Pool with the 10 default specialists and no RAG attached."""
        return cls.from_definitions(DEFAULT_SPECIALIST_DEFINITIONS)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, specialist_id: str) -> SpecialistAgent | None:
        for s in self.specialists:
            if s.specialist_id == specialist_id:
                return s
        return None

    def get_many(self, specialist_ids: list[str]) -> list[SpecialistAgent]:
        """Return multiple specialists by id, skipping unknown ids."""
        found = []
        for sid in specialist_ids:
            s = self.get(sid)
            if s is not None:
                found.append(s)
        return found

    def __len__(self) -> int:
        return len(self.specialists)

    def __iter__(self):
        return iter(self.specialists)

    # ------------------------------------------------------------------
    # RAG attachment
    # ------------------------------------------------------------------

    def attach_retriever(self, retriever: "KnowledgeRetriever") -> None:
        """Attach the same RAG backend to every specialist in the pool."""
        for s in self.specialists:
            s.retriever = retriever

    def attach_retriever_to(
        self, specialist_id: str, retriever: "KnowledgeRetriever"
    ) -> bool:
        """Attach a RAG backend to a specific specialist.  Returns False if id not found."""
        s = self.get(specialist_id)
        if s is None:
            return False
        s.retriever = retriever
        return True

    # ------------------------------------------------------------------
    # Serialisation for the Router prompt
    # ------------------------------------------------------------------

    def describe_for_router(self) -> str:
        """Compact description of all specialists for the Router Agent's prompt."""
        lines = []
        for i, s in enumerate(self.specialists):
            rag_flag = " [RAG]" if s.retriever is not None else ""
            lines.append(
                f"{i + 1}. **{s.name}** (`{s.specialist_id}`){rag_flag}\n"
                f"   Focus: {s.focus}\n"
                f"   Key evidence: {s.key_evidence}"
            )
        return "\n".join(lines)


# =============================================================================
# Router Agent  —  S_obs → top-3 specialists
# =============================================================================

ROUTER_SYSTEM_PROMPT = (
    "You are a senior emergency triage physician acting as a **Router Agent**. "
    "Your job is to review the evidence collected so far (S_obs) and select the "
    "**top {top_k} most relevant specialists** to guide further investigation. "
    "Rank them by relevance to the current clinical picture."
)

ROUTER_INSTRUCTION_TEMPLATE = """Based on the evidence collected so far (S_obs), select the **top {top_k} most relevant specialists** from the pool below.  Rank them from most relevant to least relevant.

AVAILABLE SPECIALIST POOL:
{specialist_list}

COLLECTED EVIDENCE (S_obs):
{observed_evidence}

Return a JSON object with exactly this structure:
{{
  "top_specialists": [
    {{
      "rank": 1,
      "specialist_id": "<id>",
      "rationale": "<1-2 sentences: why this specialist given S_obs>"
    }},
    ...
  ],
  "overall_assessment": "<1 sentence summary of the clinical picture and why these were chosen>"
}}

IMPORTANT:
- Choose EXACTLY {top_k} distinct specialists.
- Rank them by clinical relevance (rank 1 = best match).
- If S_obs is sparse or non-specific, still make your best judgment.
"""


class RouterAgent:
    """LLM-based Router Agent: S_obs → top-N SpecialistAgents.

    This is the ψ(S_obs, K_Ai) placeholder from the paper — an LLM-call-based
    relevance scorer.  It is a lightweight agent (not a full CAMEL ChatAgent
    with multi-turn memory) because routing is a single-step classification task.

    The Router reads S_obs, scores each specialist implicitly via the LLM, and
    returns the top-N ranked SpecialistAgents for EDPM to use.

    Usage::

        router = RouterAgent(pool, top_k=3)
        top3, meta = router.route(dialogue_history)
    """

    def __init__(
        self,
        pool: SpecialistPool,
        top_k: int = 3,
        model_name: str = "",
        api_url: str = "",
        api_key: str = "",
    ) -> None:
        self.pool = pool
        self.top_k = top_k
        self.model_name = model_name or os.getenv("ROUTER_MODEL", "gpt-4o-mini")
        self._api_url = api_url
        self._api_key = api_key

    def route(
        self, observed_evidence: str
    ) -> tuple[list[SpecialistAgent], dict[str, Any]]:
        """Route S_obs to the top-K specialists.

        Parameters
        ----------
        observed_evidence : str
            Currently collected evidence S_obs (dialogue so far).

        Returns
        -------
        (top_specialists, routing_meta)
            top_specialists is a list of 1–K SpecialistAgents (fewer on errors).
            routing_meta is the raw routing metadata for trace/debug.
        """
        if not self.pool.specialists:
            logger.warning("Empty specialist pool — cannot route")
            return [], {"error": "empty pool"}

        if not observed_evidence.strip():
            logger.info("No S_obs — returning generic top-%d", self.top_k)
            return self._fallback_topk("No evidence to route on")

        agent = self._build_camel_agent()
        instruction = ROUTER_INSTRUCTION_TEMPLATE.format(
            top_k=self.top_k,
            specialist_list=self.pool.describe_for_router(),
            observed_evidence=observed_evidence[:4000],
        )

        try:
            response = agent.step(instruction)
            content = (
                str(response.msg.content)
                if hasattr(response, "msg")
                else str(response.msgs[0].content)
            )
        except Exception as exc:
            logger.error("Router LLM call failed: %s", exc)
            return self._fallback_topk(f"LLM error: {exc}")

        parsed = _parse_json_response(content)
        if parsed is None:
            logger.warning("Router response unparseable, using fallback")
            return self._fallback_topk(f"unparseable: {content[:300]}")

        # Support both old key "top3_specialists" and new "top_specialists"
        top_data = parsed.get("top_specialists", parsed.get("top3_specialists", []))
        if not top_data:
            return self._fallback_topk("no specialists returned")

        # Resolve ids to SpecialistAgent instances, respecting rank order
        selected: list[SpecialistAgent] = []
        for entry in sorted(top_data, key=lambda e: e.get("rank", 99)):
            sid = entry.get("specialist_id", "")
            specialist = self.pool.get(sid)
            if specialist is not None:
                selected.append(specialist)
            if len(selected) >= self.top_k:
                break

        if not selected:
            return self._fallback_topk("none of the returned ids matched")

        logger.info(
            "Router: S_obs → %s",
            " > ".join(s.specialist_id for s in selected),
        )
        return selected, parsed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_camel_agent(self):
        """Build a single-shot CAMEL ChatAgent for routing."""
        from camel.agents import ChatAgent
        from camel.configs.openai_config import ChatGPTConfig
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType

        model_config_dict = ChatGPTConfig(
            temperature=0.0, max_tokens=2048
        ).as_dict()

        kwargs: dict[str, Any] = {
            "model_platform": ModelPlatformType.OPENAI,
            "model_type": self.model_name,
            "model_config_dict": model_config_dict,
        }
        if self._api_url:
            kwargs["url"] = self._api_url
        elif os.getenv("OPENAI_API_BASE_URL"):
            kwargs["url"] = os.getenv("OPENAI_API_BASE_URL")
        if self._api_key:
            kwargs["api_key"] = self._api_key
        elif os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

        model = ModelFactory.create(**kwargs)
        system_prompt = ROUTER_SYSTEM_PROMPT.format(top_k=self.top_k)
        return ChatAgent(system_message=system_prompt, model=model)

    def _fallback_topk(
        self, reason: str
    ) -> tuple[list[SpecialistAgent], dict[str, Any]]:
        """Return generic top-K when routing fails."""
        default = self.pool.get("general_emergency")
        fallback = [default] if default else []
        for s in self.pool.specialists:
            if s not in fallback:
                fallback.append(s)
            if len(fallback) >= self.top_k:
                break
        logger.info("Router fallback → %s (reason: %s)", [s.specialist_id for s in fallback], reason)
        return fallback, {"fallback": True, "reason": reason, "specialists": [s.specialist_id for s in fallback]}


# =============================================================================
# Per-specialist PPE proposal prompt
# =============================================================================

PPE_PROPOSAL_INSTRUCTION = """You are a **{specialty_name}** specialist reviewing a patient case.

Based on the evidence collected so far (S_obs), identify the **single most critical piece of evidence** that should be collected next from your specialty's perspective. This evidence should be:
- Not yet collected (based on S_obs)
- Highly discriminative for confirming or ruling out conditions in your specialty
- Actionable (can be obtained via physical exam, lab test, or imaging)

COLLECTED EVIDENCE (S_obs):
{observed_evidence}

Return a JSON object:
{{
  "specialist": "{specialty_id}",
  "ppe": "<concise description of the single evidence target>",
  "rationale": "<1-2 sentences: why this evidence is critical from your specialty's perspective>"
}}
"""


def build_ppe_agent(
    specialist: SpecialistAgent,
    model_name: str = "",
) -> Any:
    """Build a single-shot CAMEL ChatAgent that asks a specialist for 1 PPE.

    Parameters
    ----------
    specialist : SpecialistAgent
        The specialist whose perspective to use.
    model_name : str
        LLM model.  Uses env vars or gpt-4o-mini when empty.
    """
    from camel.agents import ChatAgent
    from camel.configs.openai_config import ChatGPTConfig
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    model_config_dict = ChatGPTConfig(temperature=0.0, max_tokens=1024).as_dict()

    kwargs: dict[str, Any] = {
        "model_platform": ModelPlatformType.OPENAI,
        "model_type": model_name or os.getenv("ROUTER_MODEL", "gpt-4o-mini"),
        "model_config_dict": model_config_dict,
    }
    if os.getenv("OPENAI_API_BASE_URL"):
        kwargs["url"] = os.getenv("OPENAI_API_BASE_URL")
    if os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    model = ModelFactory.create(**kwargs)
    system_prompt = (
        f"You are a **{specialist.name}** specialist. "
        f"Your domain: {specialist.focus}. "
        f"Key evidence you focus on: {specialist.key_evidence}. "
        "Given collected evidence, propose the single most critical next evidence to collect."
    )
    return ChatAgent(system_message=system_prompt, model=model)


def query_specialist_for_ppe(
    specialist: SpecialistAgent,
    observed_evidence: str,
    model_name: str = "",
) -> dict[str, Any] | None:
    """Ask one specialist for their single most critical PPE.

    Parameters
    ----------
    specialist : SpecialistAgent
    observed_evidence : str
        S_obs — evidence collected so far.
    model_name : str

    Returns
    -------
    dict with keys specialist, ppe, rationale, or None on failure.
    """
    agent = build_ppe_agent(specialist, model_name=model_name)
    instruction = PPE_PROPOSAL_INSTRUCTION.format(
        specialty_name=specialist.name,
        specialty_id=specialist.specialist_id,
        observed_evidence=observed_evidence[:4000],
    )
    try:
        response = agent.step(instruction)
        content = (
            str(response.msg.content)
            if hasattr(response, "msg")
            else str(response.msgs[0].content)
        )
    except Exception as exc:
        logger.error("PPE query failed for %s: %s", specialist.specialist_id, exc)
        return None

    parsed = _parse_json_response(content)
    if parsed is None:
        logger.warning("PPE response unparseable for %s", specialist.specialist_id)
        return None
    return parsed


# =============================================================================
# Agent Consistency Check — EBVM Criterion 3
# =============================================================================


class ConsistencyAgent:
    """Pluggable agent for EBVM Criterion 3: Agents Consistent Approve.

    Checks agreement between the Diagnostician's confidence level and the
    Verifier's independent evidence–diagnosis assessment.  Default implementation
    always returns CONSISTENT — subclass to plug in a real consistency checker.

    Usage::

        class MyConsistencyAgent(ConsistencyAgent):
            def check(self, diagnosis, confidence, verifier_assessment):
                # Custom logic, e.g. LLM call to compare assessments
                return True, "..."

        scenario = ProgressiveQCCScenario(..., consistency_agent=MyConsistencyAgent())
    """

    def check(
        self,
        diagnosis: str,
        confidence: str,
        verifier_assessment: str,
    ) -> tuple[bool, str]:
        """Evaluate agreement between Diagnostician and Verifier.

        Parameters
        ----------
        diagnosis : str
            Diagnostician's proposed diagnosis.
        confidence : str
            Diagnostician's confidence level (CONFIDENT / UNCERTAIN).
        verifier_assessment : str
            Full Verifier response including EVIDENCE_ASSESSMENT and
            PLANNER_ASSESSMENT sections.

        Returns
        -------
        (is_consistent, reason)
            is_consistent : bool — True if the agents' assessments agree.
            reason : str — short explanation.
        """
        return True, "ConsistencyAgent not configured — defaulting to CONSISTENT."


# =============================================================================
# JSON helpers (shared)
# =============================================================================

def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse JSON from an LLM response, with fallback extraction."""
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        pass
    # Extract the first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
