"""Main Benchmark class for running evaluations."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from qcc.config import ModelConfig
from qcc.datasets import Dataset, DataItem, load_dataset
from qcc.metrics import Metric, get_metric, compute_coverage
from qcc.scenarios import get_scenario
from qcc.scenarios.base import BaseScenario, CaseInput, ScenarioResult
from qcc.weci import (
    EvaluationAgent,
    compute_weci,
    extract_collected_indices_from_trace,
)

logger = logging.getLogger(__name__)


class Benchmark:
    """Main benchmark runner for evaluating LLM diagnostic capabilities.

    Coordinates dataset loading, scenario execution, and result evaluation.
    """

    def __init__(
        self,
        dataset: Dataset,
        metric: Metric,
        output_dir: Path | str = "results",
        save_traces: bool = True,
        record_suffix: str = "",
        evaluation_agent: EvaluationAgent | None = None,
    ) -> None:
        """Initialize benchmark.

        Args:
            dataset: Dataset to evaluate on
            metric: Metric for evaluation
            output_dir: Directory for output files
            save_traces: Whether to save individual case traces
            record_suffix: Suffix for record directory (e.g., '_0222_143025')
            evaluation_agent: Optional ϕ_EA for WECI computation
        """
        self.dataset = dataset
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.save_traces = save_traces
        self.record_suffix = record_suffix
        self.evaluation_agent = evaluation_agent

    def evaluate(
        self,
        scenario: BaseScenario,
        max_workers: int = 10,
        summary_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run evaluation on the dataset.

        Args:
            scenario: Scenario to execute
            max_workers: Maximum parallel workers
            summary_path: Path to save summary JSON

        Returns:
            Summary statistics dictionary
        """
        logger.info(
            "Starting evaluation: %d items, %d workers",
            len(self.dataset),
            max_workers,
        )

        evaluation_results: list[dict[str, Any]] = []

        # Create trace output directory
        record_name = f"record{self.record_suffix}" if self.record_suffix else "record"
        trace_dir = self.output_dir / record_name
        if self.save_traces:
            trace_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_single_item, scenario, item): item
                for item in self.dataset
            }

            for future in tqdm(
                as_completed(futures),
                total=len(self.dataset),
                desc="Evaluating",
                ncols=100,
            ):
                item = futures[future]
                try:
                    full_result = future.result()
                    # Extract evaluation result (without trace and metadata)
                    eval_result = self._extract_evaluation_result(full_result)
                    evaluation_results.append(eval_result)

                    # Save trace if enabled
                    if self.save_traces:
                        trace_path = trace_dir / f"{item.case_id}.json"
                        self._save_trace(trace_path, full_result)

                except Exception as e:
                    logger.error("Failed to evaluate %s: %s", item.case_id, e)
                    evaluation_results.append({
                        "case_id": item.case_id,
                        "error": str(e),
                        "is_correct": False,
                    })

        # Summarize results
        summary = self.metric.summarize(evaluation_results)
        summary["dataset"] = self.dataset.name
        summary["scenario"] = scenario.__class__.__name__

        # Save summary
        if summary_path:
            summary_path = Path(summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info("Summary saved to %s", summary_path)

        return summary

    def _run_single_item(
        self,
        scenario: BaseScenario,
        item: DataItem,
    ) -> dict[str, Any]:
        """Run scenario on a single item and evaluate.

        Args:
            scenario: Scenario to execute
            item: Data item to evaluate

        Returns:
            Full result dictionary including trace and metadata
        """
        # Build case input
        case_input = CaseInput(
            case_id=item.case_id,
            task=item.task,
            patient_facts=item.patient_facts,
            exam_facts=item.exam_facts,
            ground_truth=item.answer,
        )

        # Run scenario
        result = scenario.run(case_input)

        # Evaluate result
        eval_result, is_correct = self.metric.compare(result.answer, item.answer)

        # Compute information coverage
        coverage_result = compute_coverage(
            trace=result.trace,
            patient_facts=item.patient_facts,
            exam_facts=item.exam_facts,
        )

        # Compute WECI if EvaluationAgent is configured
        weci: float | None = None
        evidence_weights: dict[str, Any] | None = None
        if self.evaluation_agent is not None:
            weci, evidence_weights = self._compute_weci_for_case(
                case_id=item.case_id,
                ground_truth=item.answer,
                patient_facts=item.patient_facts,
                exam_facts=item.exam_facts,
                trace=result.trace,
            )

        return {
            "case_id": item.case_id,
            "prediction": result.answer,
            "ground_truth": item.answer,
            "is_correct": is_correct,
            "weci": weci,
            "evidence_weights": evidence_weights,
            "trace": result.trace,
            "metadata": result.metadata,
            "coverage": {
                "information_coverage_rate": coverage_result.information_coverage_rate,
                "patient_coverage": coverage_result.patient_coverage,
                "exam_coverage": coverage_result.exam_coverage,
                "ground_truth_patient_facts": coverage_result.ground_truth_patient_facts,
                "ground_truth_exam_facts": coverage_result.ground_truth_exam_facts,
                "collected_patient_facts": coverage_result.collected_patient_facts,
                "collected_exam_facts": coverage_result.collected_exam_facts,
            },
            **eval_result,
        }

    def _compute_weci_for_case(
        self,
        case_id: str,
        ground_truth: str,
        patient_facts: list[str],
        exam_facts: list[str],
        trace: list[dict[str, Any]],
    ) -> tuple[float | None, dict[str, Any] | None]:
        """Score evidence weights and compute WECI for a single case.

        1. EvaluationAgent scores all evidence items given GT diagnosis
        2. Extract collected indices from trace
        3. Compute WECI = Σ_{e∈E∩E*} w_e / Σ_{e∈E*} w_e

        Returns:
            (weci, evidence_weights_dict) — weci is the scalar metric;
            evidence_weights_dict contains per-item weights, collected status,
            numerator, denominator, and the GT diagnosis used for scoring.
        """
        agent = self.evaluation_agent
        if agent is None:
            return None, None

        annotated = agent.score_evidence(
            sample_id=case_id,
            gt_diagnosis=ground_truth,
            patient_facts=patient_facts,
            exam_facts=exam_facts,
        )
        if annotated is None:
            return None, None

        p_ids, e_ids = extract_collected_indices_from_trace(trace)
        weci = compute_weci(annotated, p_ids, e_ids)

        # Build detailed evidence weights dict for inspection
        def _build_items(items: list, collected_ids: set[int]) -> list[dict]:
            return [
                {
                    "index": e.index,
                    "text": e.text,
                    "weight": e.weight,
                    "collected": e.index in collected_ids,
                    "category": e.category,
                }
                for e in items
            ]

        patient_items = _build_items(annotated.patient_evidence, p_ids)
        exam_items = _build_items(annotated.exam_evidence, e_ids)

        collected_weight = sum(
            e.weight for e in annotated.patient_evidence if e.index in p_ids
        ) + sum(
            e.weight for e in annotated.exam_evidence if e.index in e_ids
        )

        evidence_weights = {
            "gt_diagnosis": ground_truth,
            "total_weight": annotated.total_weight,
            "collected_weight": collected_weight,
            "weci": weci,
            "patient_evidence": patient_items,
            "exam_evidence": exam_items,
        }

        return weci, evidence_weights

    def _extract_evaluation_result(self, full_result: dict[str, Any]) -> dict[str, Any]:
        """Extract evaluation result without trace and metadata.

        Args:
            full_result: Full result dictionary with trace and metadata

        Returns:
            Evaluation result dictionary without trace and metadata.
            Includes evidence_weights (per-item WECI scoring details) when present.
        """
        result = {
            k: v
            for k, v in full_result.items()
            if k not in ("trace", "metadata", "coverage")
        }
        coverage = full_result.get("coverage", {})
        result["information_coverage_rate"] = coverage.get(
            "information_coverage_rate", 0.0
        )
        result["patient_coverage"] = coverage.get("patient_coverage", 0.0)
        result["exam_coverage"] = coverage.get("exam_coverage", 0.0)
        # evidence_weights already in result via the dict comprehension above
        return result

    def _save_trace(self, path: Path, result: dict[str, Any]) -> None:
        """Save trace to JSON file."""
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def run_evaluation(
    dataset_name: str,
    mode: str,
    doctor_model: str,
    patient_model: str | None = None,
    reporter_model: str | None = None,
    annotator_model: str | None = None,
    summarizer_model: str | None = None,
    diagnostician_model: str | None = None,
    verifier_model: str | None = None,
    max_items: int | None = None,
    max_turns: int = 16,
    max_workers: int = 10,
    output_dir: str = "results",
    dataset_path: str | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """Convenience function to run a complete evaluation.

    Args:
        dataset_name: Name of the dataset
        mode: Evaluation mode (cot, roleplay, react, sc, refine)
        doctor_model: Model name for doctor
        patient_model: Model name for patient (optional)
        reporter_model: Model name for reporter (optional)
        annotator_model: Model name for annotator (optional)
        summarizer_model: Model name for summarizer (optional, SC/REFINE)
        diagnostician_model: Model name for diagnostician (optional, SC/REFINE)
        verifier_model: Model name for verifier (optional, REFINE)
        max_items: Maximum items to evaluate (optional)
        max_turns: Maximum interaction turns (default: 16)
        max_workers: Maximum parallel workers (default: 10)
        output_dir: Output directory (default: results)
        dataset_path: Custom dataset path (optional)
    """
    from qcc.config import load_config, ModelConfig

    # Load environment configuration
    load_config()

    # Load dataset
    dataset = load_dataset(dataset_name, path=dataset_path, max_items=max_items)

    # Create model configs
    doctor_config = ModelConfig.from_string(doctor_model)

    patient_config = (
        ModelConfig.from_string(patient_model) if patient_model else doctor_config
    )
    reporter_config = (
        ModelConfig.from_string(reporter_model) if reporter_model else doctor_config
    )
    annotator_config = (
        ModelConfig.from_string(annotator_model) if annotator_model else doctor_config
    )
    summarizer_config = (
        ModelConfig.from_string(summarizer_model) if summarizer_model else doctor_config
    )
    diagnostician_config = (
        ModelConfig.from_string(diagnostician_model) if diagnostician_model else doctor_config
    )
    verifier_config_obj = (
        ModelConfig.from_string(verifier_model) if verifier_model else doctor_config
    )

    # Create scenario
    scenario = get_scenario(
        mode=mode,
        dataset_name=dataset_name,
        doctor_config=doctor_config,
        patient_config=patient_config,
        reporter_config=reporter_config,
        max_turns=max_turns,
        summarizer_config=summarizer_config,
        diagnostician_config=diagnostician_config,
        verifier_config=verifier_config_obj,
    )

    # Create metric
    metric = get_metric(dataset_name, judge_config=annotator_config)

    # Create EvaluationAgent for WECI computation
    evaluation_agent = EvaluationAgent(model_name=doctor_model)

    # Build output path
    model_name = doctor_model.replace("/", "_")
    output_path = Path(output_dir) / dataset_name / mode / model_name
    if mode != "cot":
        output_path = output_path / f"{max_turns}_turns"

    # Run benchmark
    record_suffix = f"_{run_id}" if run_id else ""
    benchmark = Benchmark(
        dataset=dataset,
        metric=metric,
        output_dir=output_path,
        save_traces=True,
        record_suffix=record_suffix,
        evaluation_agent=evaluation_agent,
    )

    summary = benchmark.evaluate(
        scenario=scenario,
        max_workers=max_workers,
        summary_path=output_path / "result.json",
    )

    return summary
