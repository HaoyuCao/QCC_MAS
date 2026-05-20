#!/usr/bin/env python3
"""WECI (Weighted Evidence Convergent Index) computation.

Quantify stage of QCC: importance-weights collected evidence to measure
how well an interactive MAS captures decision-critical signals.

Two subcommands:
  weci-annotate  – Call LLM to assign importance scores (0–1) to each
                    evidence item given the ground-truth diagnosis.
  weci-compute   – Compute WECI from a scenario result trace using
                    pre-annotated evidence weights.

Reference
---------
Sec 2.1, "Weighted Evidence Convergent Index", QCC paper.
  WECI = Σ_{e ∈ E ∩ E*} w_e  /  Σ_{e ∈ E*} w_e
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("weci")


# =============================================================================
# RAG Interface (placeholder)
# =============================================================================


class KnowledgeRetriever(ABC):
    """Retrieval-augmented knowledge interface.

    Subclass this to plug in a real RAG backend (vector DB, MedCPT, PubMed
    search, etc.).  The default implementation returns an empty context.
    """

    @abstractmethod
    def retrieve(self, diagnosis: str, evidence: str) -> str:
        """Return knowledge context relevant to *diagnosis* and *evidence*."""
        ...


class NoOpRetriever(KnowledgeRetriever):
    """Default no-op retriever — returns empty context."""

    def retrieve(self, diagnosis: str, evidence: str) -> str:
        return ""


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class EvidenceItem:
    """A single atomic evidence item with its importance weight."""

    index: int  # 1-based index matching the original fact list
    text: str
    category: str  # "patient" or "exam"
    weight: float = 0.0  # w_e ∈ [0, 1]


@dataclass
class AnnotatedSample:
    """One dataset sample with evidence weights annotated."""

    sample_id: str
    correct_diagnosis: str
    patient_evidence: list[EvidenceItem] = field(default_factory=list)
    exam_evidence: list[EvidenceItem] = field(default_factory=list)

    @property
    def all_evidence(self) -> list[EvidenceItem]:
        return self.patient_evidence + self.exam_evidence

    @property
    def total_weight(self) -> float:
        """Σ_{e ∈ E*} w_e  – denominator of WECI."""
        return sum(e.weight for e in self.all_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "correct_diagnosis": self.correct_diagnosis,
            "patient_evidence": [
                {"index": e.index, "text": e.text, "weight": e.weight}
                for e in self.patient_evidence
            ],
            "exam_evidence": [
                {"index": e.index, "text": e.text, "weight": e.weight}
                for e in self.exam_evidence
            ],
            "total_weight": self.total_weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnnotatedSample":
        s = cls(sample_id=d["sample_id"], correct_diagnosis=d["correct_diagnosis"])
        s.patient_evidence = [
            EvidenceItem(**e) for e in d.get("patient_evidence", [])
        ]
        s.exam_evidence = [
            EvidenceItem(**e) for e in d.get("exam_evidence", [])
        ]
        return s


# =============================================================================
# Prompt templates
# =============================================================================

WECI_ANNOTATION_SYSTEM_PROMPT = (
    "You are a senior clinician evaluating the diagnostic importance of "
    "individual clinical evidence items. Your task is to assign an importance "
    "weight to each piece of evidence based on how critical it is for reaching "
    "the correct diagnosis."
)

WECI_ANNOTATION_INSTRUCTION_TEMPLATE = """Given the CORRECT diagnosis below, assign an importance weight (0.0 to 1.0) to each numbered evidence item.

Scoring guidelines:
- 1.0 = Pathognomonic / gold-standard — this evidence alone strongly confirms the diagnosis
- 0.7–0.9 = Highly discriminative — key finding that narrows the differential significantly
- 0.4–0.6 = Moderately helpful — supports the diagnosis but is non-specific
- 0.1–0.3 = Marginally relevant — background context, demographic, or weak signal
- 0.0 = Irrelevant — provides no diagnostic signal for this specific diagnosis

CORRECT DIAGNOSIS: {diagnosis}

{rag_context}

PATIENT EVIDENCE (history, symptoms, demographics):
{patient_evidence}

EXAM EVIDENCE (physical exam, labs, imaging):
{exam_evidence}

Return a JSON object with this exact structure:
{{
  "patient_weights": [
    {{"index": 1, "weight": <float>, "rationale": "<one short sentence>"}},
    ...
  ],
  "exam_weights": [
    {{"index": 1, "weight": <float>, "rationale": "<one short sentence>"}},
    ...
  ]
}}
"""


# =============================================================================
# Core logic
# =============================================================================


def _build_llm_agent(model_name: str, api_url: str = "", api_key: str = ""):
    """Build a CAMEL ChatAgent instance for evidence scoring.

    Parameters
    ----------
    model_name : str
        e.g. "gpt-5-mini", "qwen3-max"
    api_url : str
        Optional override for the API base URL.
    api_key : str
        Optional override for the API key.
    """
    import os

    from camel.agents import ChatAgent
    from camel.configs.openai_config import ChatGPTConfig
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    model_config_dict = ChatGPTConfig(
        temperature=0.0, max_tokens=4096
    ).as_dict()

    kwargs: dict[str, Any] = {
        "model_platform": ModelPlatformType.OPENAI,
        "model_type": model_name,
        "model_config_dict": model_config_dict,
    }
    if api_url:
        kwargs["url"] = api_url
    elif os.getenv("OPENAI_API_BASE_URL"):
        kwargs["url"] = os.getenv("OPENAI_API_BASE_URL")
    if api_key:
        kwargs["api_key"] = api_key
    elif os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    model = ModelFactory.create(**kwargs)
    return ChatAgent(system_message=WECI_ANNOTATION_SYSTEM_PROMPT, model=model)


def _parse_weights_response(text: str) -> tuple[list[dict], list[dict]] | None:
    """Parse LLM JSON response into patient and exam weight lists."""
    try:
        data = json.loads(text)
        return data.get("patient_weights", []), data.get("exam_weights", [])
    except json.JSONDecodeError:
        # Try to extract JSON block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("patient_weights", []), data.get("exam_weights", [])
            except json.JSONDecodeError:
                pass
    return None


def score_single_sample(
    raw: dict[str, Any],
    agent: Any,
    retriever: KnowledgeRetriever | None = None,
) -> AnnotatedSample | None:
    """Annotate one dataset sample with evidence importance weights.

    Parameters
    ----------
    raw : dict
        Raw record from a segmented JSONL dataset, containing
        ``atomic_facts`` and ``OSCE_Examination.Correct_Diagnosis``.
    agent : camel.agents.ChatAgent
        Pre-built LLM agent.
    retriever : KnowledgeRetriever or None
        Optional RAG backend.

    Returns
    -------
    AnnotatedSample or None
    """
    retriever = retriever or NoOpRetriever()

    sample_id = str(raw.get("id", raw.get("_id", "unknown")))
    osce = raw.get("OSCE_Examination", {})
    diagnosis = osce.get("Correct_Diagnosis", "")
    if not diagnosis:
        logger.warning("Sample %s has no Correct_Diagnosis — skipping", sample_id)
        return None

    atomic = raw.get("atomic_facts", {})
    patient_facts: list[str] = atomic.get("patient_facts", [])
    exam_facts: list[str] = atomic.get("exam_facts", [])

    if not patient_facts and not exam_facts:
        logger.warning("Sample %s has no atomic_facts — skipping", sample_id)
        return None

    # Build evidence text blocks
    patient_text = "\n".join(patient_facts) if patient_facts else "(none)"
    exam_text = "\n".join(exam_facts) if exam_facts else "(none)"

    # Retrieve RAG context (no-op by default)
    all_evidence_concise = "\n".join(patient_facts[:5] + exam_facts[:5])
    rag_context = retriever.retrieve(diagnosis, all_evidence_concise)
    rag_block = f"ADDITIONAL MEDICAL KNOWLEDGE:\n{rag_context}\n" if rag_context else ""

    instruction = WECI_ANNOTATION_INSTRUCTION_TEMPLATE.format(
        diagnosis=diagnosis,
        rag_context=rag_block,
        patient_evidence=patient_text,
        exam_evidence=exam_text,
    )

    try:
        response = agent.step(instruction)
        content = str(response.msg.content) if hasattr(response, "msg") else str(response.msgs[0].content)
    except Exception as exc:
        logger.error("LLM call failed for %s: %s", sample_id, exc)
        return None

    parsed = _parse_weights_response(content)
    if parsed is None:
        logger.warning("Could not parse LLM response for %s", sample_id)
        return None

    patient_weights, exam_weights = parsed

    # Build EvidenceItem lists, matching by index
    def _match(
        facts: list[str], weights: list[dict], category: str
    ) -> list[EvidenceItem]:
        items = []
        # Index fact strings by their leading number
        fact_map: dict[int, str] = {}
        for f in facts:
            m = re.match(r"\s*(\d+)\.\s*", f)
            if m:
                fact_map[int(m.group(1))] = f

        for w in weights:
            idx = int(w.get("index", 0))
            weight = float(w.get("weight", 0))
            rationale = w.get("rationale", "")
            text = fact_map.get(idx, f"Fact #{idx}")
            items.append(
                EvidenceItem(index=idx, text=text, category=category, weight=weight)
            )
        return items

    sample = AnnotatedSample(sample_id=sample_id, correct_diagnosis=diagnosis)
    sample.patient_evidence = _match(patient_facts, patient_weights, "patient")
    sample.exam_evidence = _match(exam_facts, exam_weights, "exam")
    return sample


# =============================================================================
# Evaluation Agent (ϕ_EA)  —  evidence scorer with optional RAG
# =============================================================================


class EvaluationAgent:
    """ϕ_EA — Evidence Evaluation Agent for WECI computation.

    Scores each evidence item's diagnostic importance (0–1 weight) given the
    ground-truth diagnosis.  Optionally augmented with a KnowledgeRetriever.

    This is the agent embodiment of the WECI annotation step — it sees the GT
    diagnosis and all atomic facts, then assigns w_e ∈ [0,1] to each evidence
    item.  The score_evidence() method returns an AnnotatedSample ready for
    WECI computation.

    Usage::

        agent = EvaluationAgent(model_name="gpt-4o-mini")
        annotated = agent.score_evidence(
            sample_id="case_0",
            gt_diagnosis="Myasthenia gravis",
            patient_facts=["1. Ptosis present", "2. Diplopia"],
            exam_facts=["1. Anti-AChR Ab positive", "2. Repetitive nerve stimulation abnormal"],
        )
        # annotated is an AnnotatedSample with weights filled in
    """

    def __init__(
        self,
        model_name: str = "",
        retriever: KnowledgeRetriever | None = None,
        api_url: str = "",
        api_key: str = "",
    ) -> None:
        import os

        self.model_name = model_name or os.getenv("ROUTER_MODEL", "gpt-4o-mini")
        self.retriever: KnowledgeRetriever = retriever or NoOpRetriever()
        self._api_url = api_url
        self._api_key = api_key

    def _build_agent(self) -> Any:
        """Build a fresh CAMEL ChatAgent (thread-safe — no caching)."""
        return _build_llm_agent(self.model_name, self._api_url, self._api_key)

    def score_evidence(
        self,
        sample_id: str,
        gt_diagnosis: str,
        patient_facts: list[str],
        exam_facts: list[str],
    ) -> AnnotatedSample | None:
        """Score all evidence items for a single case.

        Parameters
        ----------
        sample_id : str
            Case identifier (e.g. "agentclinic_medqa_0").
        gt_diagnosis : str
            Ground-truth diagnosis — the Evaluation Agent uses this to
            calibrate importance weights.
        patient_facts : list[str]
            Indexed patient fact strings (e.g. ["1. Age 45", "2. Chest pain"]).
        exam_facts : list[str]
            Indexed exam fact strings.

        Returns
        -------
        AnnotatedSample or None
            Sample with weights filled in, or None on LLM failure.
        """
        if not gt_diagnosis:
            logger.warning("Sample %s has no GT diagnosis — skipping scoring", sample_id)
            return None
        if not patient_facts and not exam_facts:
            logger.warning("Sample %s has no facts — skipping scoring", sample_id)
            return None

        patient_text = "\n".join(patient_facts) if patient_facts else "(none)"
        exam_text = "\n".join(exam_facts) if exam_facts else "(none)"

        # RAG context
        all_evidence_concise = "\n".join(patient_facts[:5] + exam_facts[:5])
        rag_context = self.retriever.retrieve(gt_diagnosis, all_evidence_concise)
        rag_block = (
            f"ADDITIONAL MEDICAL KNOWLEDGE:\n{rag_context}\n" if rag_context else ""
        )

        instruction = WECI_ANNOTATION_INSTRUCTION_TEMPLATE.format(
            diagnosis=gt_diagnosis,
            rag_context=rag_block,
            patient_evidence=patient_text,
            exam_evidence=exam_text,
        )

        agent = self._build_agent()
        try:
            response = agent.step(instruction)
            content = (
                str(response.msg.content)
                if hasattr(response, "msg")
                else str(response.msgs[0].content)
            )
        except Exception as exc:
            logger.error(
                "EvaluationAgent LLM call failed for %s: %s", sample_id, exc
            )
            return None

        parsed = _parse_weights_response(content)
        if parsed is None:
            logger.warning(
                "Could not parse EvaluationAgent response for %s", sample_id
            )
            return None

        patient_weights, exam_weights = parsed

        def _match(
            facts: list[str], weights: list[dict], category: str
        ) -> list[EvidenceItem]:
            items = []
            fact_map: dict[int, str] = {}
            for f in facts:
                m = re.match(r"\s*(\d+)\.\s*", f)
                if m:
                    fact_map[int(m.group(1))] = f

            for w in weights:
                idx = int(w.get("index", 0))
                weight = float(w.get("weight", 0))
                text = fact_map.get(idx, f"Fact #{idx}")
                items.append(
                    EvidenceItem(
                        index=idx, text=text, category=category, weight=weight
                    )
                )
            return items

        sample = AnnotatedSample(
            sample_id=sample_id, correct_diagnosis=gt_diagnosis
        )
        sample.patient_evidence = _match(patient_facts, patient_weights, "patient")
        sample.exam_evidence = _match(exam_facts, exam_weights, "exam")
        return sample


def compute_weci(
    annotated: AnnotatedSample,
    collected_patient_indices: set[int],
    collected_exam_indices: set[int],
) -> float:
    """Compute WECI for a single case.

    WECI = Σ_{e ∈ E∩E*} w_e  /  Σ_{e ∈ E*} w_e

    Parameters
    ----------
    annotated : AnnotatedSample
        Pre-annotated sample with evidence weights.
    collected_patient_indices : set[int]
        1-based indices of collected patient facts.
    collected_exam_indices : set[int]
        1-based indices of collected exam facts.

    Returns
    -------
    float
        WECI ∈ [0, 1].
    """
    numerator = 0.0
    denominator = annotated.total_weight

    if denominator == 0:
        return 0.0

    for e in annotated.patient_evidence:
        if e.index in collected_patient_indices:
            numerator += e.weight

    for e in annotated.exam_evidence:
        if e.index in collected_exam_indices:
            numerator += e.weight

    return numerator / denominator


def extract_collected_indices_from_trace(trace: list[dict[str, Any]]) -> tuple[set[int], set[int]]:
    """Extract collected fact indices from a scenario result trace.

    Parameters
    ----------
    trace : list[dict]
        Scenario trace entries, each having ``role_id`` and ``content``.

    Returns
    -------
    (patient_indices, exam_indices)
    """
    patient_ids: set[int] = set()
    exam_ids: set[int] = set()

    for entry in trace:
        role = entry.get("role_id", "")
        content = entry.get("content", "")

        if role == "patient":
            # Parse [REFERENCE] section for indices
            ref_match = re.search(
                r"\[REFERENCE\]\s*(.+?)(?=\n\[RESPONSE\]|\n\n|$)",
                content, re.DOTALL,
            )
            if ref_match:
                ref_content = ref_match.group(1).strip()
                if ref_content.upper() != "N/A":
                    patient_ids.update(
                        int(m) for m in re.findall(r"(\d+)\.", ref_content)
                    )
        elif role in {"reporter", "measurement"}:
            exam_ids.update(int(m) for m in re.findall(r"(\d+)\.", content))

    return patient_ids, exam_ids


# =============================================================================
# Annotate CLI
# =============================================================================


def cmd_annotate(args: argparse.Namespace) -> None:
    """Annotate a segmented JSONL dataset with evidence weights."""
    from dotenv import load_dotenv

    load_dotenv()

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("Input file not found: %s", in_path)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load samples
    samples: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.max_items:
        samples = samples[: args.max_items]

    logger.info("Annotating %d samples with model=%s ...", len(samples), args.model)

    # Build RAG retriever (placeholder – subclass to enable)
    retriever: KnowledgeRetriever = NoOpRetriever()
    if args.rag_backend:
        logger.warning(
            "RAG backend '%s' requested but no implementation found — "
            "using NoOpRetriever. Set up a KnowledgeRetriever subclass to enable.",
            args.rag_backend,
        )

    agent = _build_llm_agent(args.model, args.api_url, args.api_key)

    annotated_samples: list[AnnotatedSample] = []
    failures = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(score_single_sample, s, agent, retriever): i
            for i, s in enumerate(samples)
        }
        with tqdm(total=len(samples), desc="Scoring evidence", ncols=100) as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    annotated_samples.append(result)
                else:
                    failures += 1
                pbar.update(1)

    # Write output
    with out_path.open("w", encoding="utf-8") as f:
        for s in annotated_samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    logger.info(
        "Done: %d annotated, %d failed → %s",
        len(annotated_samples), failures, out_path,
    )


# =============================================================================
# Compute CLI
# =============================================================================


def cmd_compute(args: argparse.Namespace) -> None:
    """Compute WECI from scenario result traces using pre-annotated weights."""
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Weights file not found: %s", weights_path)
        sys.exit(1)

    traces_dir = Path(args.traces)
    if not traces_dir.is_dir():
        logger.error("Traces directory not found: %s", traces_dir)
        sys.exit(1)

    # Load annotated weights (indexed by sample_id)
    weight_map: dict[str, AnnotatedSample] = {}
    with weights_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s = AnnotatedSample.from_dict(json.loads(line))
                weight_map[s.sample_id] = s

    logger.info("Loaded weights for %d samples", len(weight_map))

    # Compute WECI for each trace file
    weci_values: list[float] = []
    patient_coverages: list[float] = []
    exam_coverages: list[float] = []

    for trace_file in sorted(traces_dir.glob("*.json")):
        with trace_file.open("r", encoding="utf-8") as f:
            result = json.load(f)

        case_id = result.get("case_id", trace_file.stem)

        # Map trace case_id to weight sample_id
        # (agentclinic_medqa_N → agentclinic_medqa_N)
        annotated = weight_map.get(case_id)
        if annotated is None:
            # Try common suffixes
            for key in weight_map:
                if case_id in key or key in case_id:
                    annotated = weight_map[key]
                    break

        if annotated is None:
            logger.warning("No weights for %s — skipping", case_id)
            continue

        trace = result.get("trace", [])
        p_ids, e_ids = extract_collected_indices_from_trace(trace)
        weci = compute_weci(annotated, p_ids, e_ids)

        # Also compute raw coverage for comparison
        p_total = len(annotated.patient_evidence)
        e_total = len(annotated.exam_evidence)
        pcov = len(p_ids) / p_total if p_total else 1.0
        ecov = len(e_ids) / e_total if e_total else 1.0

        weci_values.append(weci)
        patient_coverages.append(pcov)
        exam_coverages.append(ecov)

        if args.verbose:
            logger.info(
                "%s: WECI=%.4f  PC=%.2f  EC=%.2f  (p_collected=%d/%d  e_collected=%d/%d)",
                case_id, weci, pcov, ecov,
                len(p_ids), p_total, len(e_ids), e_total,
            )

    if not weci_values:
        logger.error("No matching trace-weight pairs found")
        sys.exit(1)

    avg_weci = sum(weci_values) / len(weci_values)
    avg_pc = sum(patient_coverages) / len(patient_coverages)
    avg_ec = sum(exam_coverages) / len(exam_coverages)

    print(json.dumps({
        "num_samples": len(weci_values),
        "avg_weci": round(avg_weci, 4),
        "avg_patient_coverage": round(avg_pc, 4),
        "avg_exam_coverage": round(avg_ec, 4),
    }, indent=2))


# =============================================================================
# CLI entry point
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WECI — Weighted Evidence Convergent Index (QCC Quantify Stage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- annotate ----------------------------------------------------------
    ap = sub.add_parser("annotate", help="Annotate dataset with evidence weights")
    ap.add_argument("--input", required=True, help="Path to segmented JSONL dataset")
    ap.add_argument("--output", required=True, help="Output path for annotated weights (JSONL)")
    ap.add_argument("--model", default="gpt-5-mini", help="LLM for evidence scoring (default: gpt-5-mini)")
    ap.add_argument("--api-url", default="", help="API base URL override")
    ap.add_argument("--api-key", default="", help="API key override")
    ap.add_argument("--max-items", type=int, default=None, help="Limit number of samples")
    ap.add_argument("--workers", type=int, default=5, help="Parallel workers (default: 5)")
    ap.add_argument("--rag-backend", default=None, help="RAG backend identifier (requires custom KnowledgeRetriever)")

    # ---- compute -----------------------------------------------------------
    cp = sub.add_parser("compute", help="Compute WECI from scenario traces")
    cp.add_argument("--weights", required=True, help="Path to annotated weights JSONL")
    cp.add_argument("--traces", required=True, help="Directory containing trace JSON files")
    cp.add_argument("--verbose", action="store_true", help="Print per-case WECI")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "compute":
        cmd_compute(args)


if __name__ == "__main__":
    main()
