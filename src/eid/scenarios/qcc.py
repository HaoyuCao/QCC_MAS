"""QCC (Quality Controlled Consultation) scenario.

Extends REFINE with a Planner agent that generates candidate diagnoses
and an evidence checklist to guide systematic evidence collection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eid.agents import create_agent
from eid.prompts import PromptManager
from eid.scenarios.base import BaseScenario, CaseInput, ScenarioResult

if TYPE_CHECKING:
    from eid.config import ModelConfig
    from eid.agents.chat_agent import AgentWrapper


@dataclass
class CandidateDiagnosis:
    """A candidate diagnosis with its gold standard evidence."""
    
    diagnosis: str
    gold_standard: str
    is_pruned: bool = False
    prune_reason: str = ""


@dataclass
class EvidenceItem:
    """An evidence item in the checklist."""
    
    description: str
    collected: bool = False
    result: str = ""
    associated_candidate: int | None = None  # Index of associated candidate, None for differential


@dataclass
class DiagnosticPlan:
    """Container for the diagnostic plan from Planner."""
    
    candidates: list[CandidateDiagnosis] = field(default_factory=list)
    differential_evidence: list[EvidenceItem] = field(default_factory=list)
    
    def get_active_candidates(self) -> list[tuple[int, CandidateDiagnosis]]:
        """Return list of (index, candidate) for non-pruned candidates."""
        return [(i, c) for i, c in enumerate(self.candidates) if not c.is_pruned]
    
    def get_all_candidates_str(self) -> str:
        """Format all candidates with pruned status."""
        lines = []
        for i, c in enumerate(self.candidates):
            status = " [PRUNED]" if c.is_pruned else ""
            lines.append(f"{i+1}. {c.diagnosis}{status}")
            lines.append(f"   - Gold Standard: {c.gold_standard}")
            if c.is_pruned:
                lines.append(f"   - Prune Reason: {c.prune_reason}")
        return "\n".join(lines)
    
    def get_active_candidates_str(self) -> str:
        """Format active (non-pruned) candidates."""
        lines = []
        for i, c in self.get_active_candidates():
            lines.append(f"{i+1}. {c.diagnosis}")
            lines.append(f"   - Gold Standard: {c.gold_standard}")
        return "\n".join(lines)
    
    def get_checklist_str(self) -> str:
        """Format evidence checklist with completion status."""
        lines = []
        
        # Gold standard evidence for active candidates
        lines.append("=== Gold Standard Evidence ===")
        for i, c in enumerate(self.candidates):
            if c.is_pruned:
                status = "[PRUNED - skipped]"
            elif c.gold_standard in [e.description for e in self.differential_evidence if e.collected]:
                status = "[✓ Collected]"
            else:
                # Check if explicitly collected
                status = "[○ Pending]"
            lines.append(f"{i+1}. ({c.diagnosis}) {c.gold_standard} {status}")
        
        # Differential evidence
        lines.append("\n=== Differential Evidence ===")
        for i, e in enumerate(self.differential_evidence):
            if e.associated_candidate is not None and self.candidates[e.associated_candidate].is_pruned:
                status = "[PRUNED - skipped]"
            elif e.collected:
                status = f"[✓ {e.result[:50]}...]" if len(e.result) > 50 else f"[✓ {e.result}]"
            else:
                status = "[○ Pending]"
            lines.append(f"{i+1}. {e.description} {status}")
        
        return "\n".join(lines)
    
    def prune_candidate(self, index: int, reason: str) -> bool:
        """Mark a candidate as pruned."""
        if 0 <= index < len(self.candidates) and not self.candidates[index].is_pruned:
            self.candidates[index].is_pruned = True
            self.candidates[index].prune_reason = reason
            return True
        return False


class QCCScenario(BaseScenario):
    """QCC (Quality Controlled Consultation) evaluation scenario.

    Extends REFINE with:
    - Planner agent: generates candidate diagnoses and evidence checklist
    - Guided evidence collection: Doctor follows checklist
    - Dynamic pruning: Doctor can prune candidates during collection
    - Re-planning: If candidates don't fit evidence, generate new plan
    """

    def __init__(
        self,
        dataset_name: str,
        doctor_config: "ModelConfig",
        patient_config: "ModelConfig",
        reporter_config: "ModelConfig",
        max_turns: int = 16,
        summarizer_config: "ModelConfig | None" = None,
        diagnostician_config: "ModelConfig | None" = None,
        verifier_config: "ModelConfig | None" = None,
        planner_config: "ModelConfig | None" = None,
        initial_turns: int = 5,
    ) -> None:
        """Initialize QCC scenario.

        Args:
            dataset_name: Name of the dataset
            doctor_config: Model configuration for doctor role
            patient_config: Model configuration for patient simulator
            reporter_config: Model configuration for reporter simulator
            max_turns: Maximum interaction turns
            summarizer_config: Model configuration for summarizer role (default: doctor_config)
            diagnostician_config: Model configuration for diagnostician role (default: doctor_config)
            verifier_config: Model configuration for verifier role (default: doctor_config)
            planner_config: Model configuration for planner role (default: doctor_config)
            initial_turns: Number of initial turns before planning (default: 2)
        """
        super().__init__(dataset_name)
        self.doctor_config = doctor_config
        self.patient_config = patient_config
        self.reporter_config = reporter_config
        self.max_turns = max_turns
        self.summarizer_config = summarizer_config or doctor_config
        self.diagnostician_config = diagnostician_config or doctor_config
        self.verifier_config = verifier_config or doctor_config
        self.planner_config = planner_config or doctor_config
        self.initial_turns = initial_turns
        self.prompts = PromptManager(dataset_name)

    # =========================================================================
    # Parsing Methods
    # =========================================================================

    @staticmethod
    def parse_planner_response(text: str) -> DiagnosticPlan:
        """Parse planner response into DiagnosticPlan.

        Args:
            text: Planner agent response

        Returns:
            DiagnosticPlan with candidates and differential evidence
        """
        plan = DiagnosticPlan()
        
        # Extract candidates section
        candidates_match = re.search(
            r"\[CANDIDATES\]\s*([\s\S]*?)(?=\[DIFFERENTIAL_EVIDENCE\]|\Z)",
            text
        )
        if candidates_match:
            candidates_text = candidates_match.group(1)
            # Parse each candidate: "1. Diagnosis\n   - Gold Standard: ..."
            candidate_pattern = re.compile(
                r"(\d+)\.\s*([^\n]+)\n\s*-\s*Gold Standard:\s*([^\n]+)",
                re.IGNORECASE
            )
            for match in candidate_pattern.finditer(candidates_text):
                plan.candidates.append(CandidateDiagnosis(
                    diagnosis=match.group(2).strip(),
                    gold_standard=match.group(3).strip(),
                ))
        
        # Extract differential evidence section
        diff_match = re.search(
            r"\[DIFFERENTIAL_EVIDENCE\]\s*([\s\S]*)",
            text
        )
        if diff_match:
            diff_text = diff_match.group(1)
            # Parse each evidence item: "1. Evidence description"
            evidence_pattern = re.compile(r"(\d+)\.\s*([^\n]+)")
            for match in evidence_pattern.finditer(diff_text):
                plan.differential_evidence.append(EvidenceItem(
                    description=match.group(2).strip(),
                ))
        
        return plan

    @staticmethod
    def extract_prune_action(text: str) -> tuple[int | None, str]:
        """Extract prune action from doctor response.

        Args:
            text: Doctor response text

        Returns:
            Tuple of (candidate_index or None, prune_reason)
        """
        prune_match = re.search(r"\[PRUNE\]\s*(\d+)\s*([\s\S]*?)(?=\[|$)", text)
        if prune_match:
            try:
                index = int(prune_match.group(1)) - 1  # Convert to 0-indexed
                reason = prune_match.group(2).strip()
                return index, reason
            except ValueError:
                pass
        return None, ""

    @staticmethod
    def extract_confidence(text: str) -> str:
        """Extract confidence level from diagnostician response.

        Args:
            text: Diagnostician response text

        Returns:
            Confidence level: 'CONFIDENT' or 'UNCERTAIN'
        """
        confidence_match = re.search(r"\[CONFIDENCE\]\s*(\w+)", text)
        if confidence_match:
            confidence = confidence_match.group(1).upper()
            if confidence in ("CONFIDENT", "UNCERTAIN"):
                return confidence
        return "UNCERTAIN"  # Default to uncertain

    @staticmethod
    def extract_verifier_decision_qcc(text: str) -> tuple[str, str]:
        """Extract decision and feedback from QCC verifier response.

        Args:
            text: Verifier agent response

        Returns:
            Tuple of (decision, feedback)
            decision is one of: 'PASS', 'INCOMPLETE', 'REPLAN'
        """
        decision_match = re.search(r"\[DECISION\]\s*(\w+)", text)
        feedback_match = re.search(r"\[FEEDBACK\]\s*([\s\S]*)", text)

        decision = decision_match.group(1).upper() if decision_match else "PASS"
        if decision not in ("PASS", "INCOMPLETE", "REPLAN"):
            decision = "PASS"
        feedback = feedback_match.group(1).strip() if feedback_match else ""

        return decision, feedback

    # =========================================================================
    # Main Run Method
    # =========================================================================

    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute QCC scenario on a single case.

        Args:
            case_input: Input data for the case

        Returns:
            ScenarioResult with diagnosis and trace
        """
        # Format facts for simulators
        patient_facts_str = "\n".join(case_input.patient_facts)
        exam_facts_str = "\n".join(case_input.exam_facts)

        # Create agents
        initial_doctor = create_agent(
            role_id="doctor_initial",
            system_prompt=self.prompts.get_base_doctor_system_prompt().format(
                max_turns=self.max_turns
            ),
            config=self.doctor_config,
            message_window_size=24,
            summarize_threshold=80,
        )

        patient = create_agent(
            role_id="patient",
            system_prompt=self.prompts.get_patient_system_prompt().format(
                patient_facts=patient_facts_str
            ),
            config=self.patient_config,
            message_window_size=1,
            summarize_threshold=95,
        )

        reporter = create_agent(
            role_id="reporter",
            system_prompt=self.prompts.get_reporter_system_prompt().format(
                exam_facts=exam_facts_str
            ),
            config=self.reporter_config,
            message_window_size=1,
            summarize_threshold=80,
        )

        planner = create_agent(
            role_id="planner",
            system_prompt=self.prompts.get_qcc_planner_system_prompt(),
            config=self.planner_config,
            message_window_size=3,
            summarize_threshold=80,
        )

        summarizer = create_agent(
            role_id="summarizer",
            system_prompt=self.prompts.get_summarizer_system_prompt(),
            config=self.summarizer_config,
            message_window_size=3,
            summarize_threshold=80,
        )

        diagnostician = create_agent(
            role_id="diagnostician",
            system_prompt=self.prompts.get_qcc_diagnostician_system_prompt(),
            config=self.diagnostician_config,
            message_window_size=3,
            summarize_threshold=90,
        )

        verifier = create_agent(
            role_id="verifier",
            system_prompt=self.prompts.get_qcc_verifier_system_prompt(),
            config=self.verifier_config,
            message_window_size=3,
            summarize_threshold=80,
        )

        # Initialize state
        trace: list[dict] = []
        dialogue_history = ""
        last_reply = ""
        answer = ""
        current_turn = 0
        diagnostic_plan: DiagnosticPlan | None = None
        qcc_doctor: "AgentWrapper | None" = None

        # =====================================================================
        # Phase 1: Initial Evidence Collection (First 2 turns)
        # =====================================================================
        while current_turn < self.initial_turns and current_turn < self.max_turns:
            instruction = self.prompts.get_doctor_turn_instruction(
                current_turns=current_turn,
                max_turns=self.max_turns,
                last_reply=last_reply,
            )

            doctor_response = initial_doctor.step(instruction)
            duration = initial_doctor.get_last_duration()

            trace.append({
                "role_id": "doctor_initial",
                "content": doctor_response,
                "duration": duration,
            })

            action_type, action_content = self.extract_action(doctor_response)
            dialogue_history += f"\nDoctor: {action_content}\n"

            # Early finish during initial phase
            if action_type == "finish" or action_type == "diagnosis":
                answer = action_content if action_type == "diagnosis" else ""
                return self._finalize_result(
                    trace=trace,
                    answer=answer,
                    current_turn=current_turn,
                    diagnostic_plan=None,
                    agents=[
                        ("doctor_initial", initial_doctor),
                        ("patient", patient),
                        ("reporter", reporter),
                        ("planner", planner),
                        ("summarizer", summarizer),
                        ("diagnostician", diagnostician),
                        ("verifier", verifier),
                    ],
                )

            current_turn += 1

            # Route to simulator
            if action_type == "test":
                m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
                m_response = reporter.step(m_instruction)
                m_duration = reporter.get_last_duration()

                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": m_duration,
                })
                last_reply = m_response
                dialogue_history += f"\nMeasurement: {m_response}\n"
            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()

                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })
                last_reply = self.extract_patient_response(p_response)
                dialogue_history += f"\nPatient: {last_reply}\n"

        # =====================================================================
        # Planning Phase: Generate Diagnostic Plan
        # =====================================================================
        planner_instruction = self.prompts.get_planner_instruction(dialogue_history)
        planner_response = planner.step(planner_instruction)
        planner_duration = planner.get_last_duration()

        trace.append({
            "role_id": "planner",
            "content": planner_response,
            "duration": planner_duration,
        })

        diagnostic_plan = self.parse_planner_response(planner_response)

        # Create QCC doctor with checklist awareness
        qcc_doctor = create_agent(
            role_id="doctor_qcc",
            system_prompt=self.prompts.get_qcc_doctor_system_prompt().format(
                max_turns=self.max_turns
            ),
            config=self.doctor_config,
            message_window_size=24,
            summarize_threshold=80,
        )

        # =====================================================================
        # Phase 2: Guided Evidence Collection
        # =====================================================================
        while current_turn < self.max_turns:
            instruction = self.prompts.get_qcc_doctor_turn_instruction(
                current_turns=current_turn,
                max_turns=self.max_turns,
                last_reply=last_reply,
                checklist=diagnostic_plan.get_checklist_str(),
                active_candidates=diagnostic_plan.get_active_candidates_str(),
            )

            doctor_response = qcc_doctor.step(instruction)
            duration = qcc_doctor.get_last_duration()

            trace.append({
                "role_id": "doctor_qcc",
                "content": doctor_response,
                "duration": duration,
            })

            # Check for prune action
            prune_index, prune_reason = self.extract_prune_action(doctor_response)
            if prune_index is not None:
                diagnostic_plan.prune_candidate(prune_index, prune_reason)
                trace.append({
                    "role_id": "system",
                    "content": f"Candidate {prune_index + 1} pruned: {prune_reason}",
                    "duration": 0,
                })

            action_type, action_content = self.extract_action(doctor_response)
            dialogue_history += f"\nDoctor: {action_content}\n"

            # Check for finish
            if action_type == "finish":
                answer, should_continue, diagnostic_plan = self._run_qcc_verification_pipeline(
                    trace=trace,
                    dialogue_history=dialogue_history,
                    diagnostic_plan=diagnostic_plan,
                    summarizer=summarizer,
                    diagnostician=diagnostician,
                    verifier=verifier,
                    planner=planner,
                    qcc_doctor=qcc_doctor,
                    patient=patient,
                    reporter=reporter,
                    current_turn=current_turn,
                )

                if not should_continue or current_turn >= self.max_turns - 1:
                    break

                current_turn += 1
                continue

            current_turn += 1

            # Route to simulator
            if action_type == "test":
                m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
                m_response = reporter.step(m_instruction)
                m_duration = reporter.get_last_duration()

                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": m_duration,
                })
                last_reply = m_response
                dialogue_history += f"\nMeasurement: {m_response}\n"
            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()

                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })
                last_reply = self.extract_patient_response(p_response)
                dialogue_history += f"\nPatient: {last_reply}\n"

        # Final verification if we hit turn limit
        if current_turn >= self.max_turns and not answer:
            answer, _, _ = self._run_qcc_verification_pipeline(
                trace=trace,
                dialogue_history=dialogue_history,
                diagnostic_plan=diagnostic_plan,
                summarizer=summarizer,
                diagnostician=diagnostician,
                verifier=verifier,
                planner=planner,
                qcc_doctor=qcc_doctor,
                patient=patient,
                reporter=reporter,
                current_turn=current_turn,
            )

        # Collect all agents for final result
        all_agents = [
            ("doctor_initial", initial_doctor),
            ("patient", patient),
            ("reporter", reporter),
            ("planner", planner),
            ("summarizer", summarizer),
            ("diagnostician", diagnostician),
            ("verifier", verifier),
        ]
        if qcc_doctor:
            all_agents.append(("doctor_qcc", qcc_doctor))

        return self._finalize_result(
            trace=trace,
            answer=answer,
            current_turn=current_turn,
            diagnostic_plan=diagnostic_plan,
            agents=all_agents,
        )

    # =========================================================================
    # Verification Pipeline
    # =========================================================================

    def _run_qcc_verification_pipeline(
        self,
        trace: list[dict],
        dialogue_history: str,
        diagnostic_plan: DiagnosticPlan,
        summarizer: "AgentWrapper",
        diagnostician: "AgentWrapper",
        verifier: "AgentWrapper",
        planner: "AgentWrapper",
        qcc_doctor: "AgentWrapper",
        patient: "AgentWrapper",
        reporter: "AgentWrapper",
        current_turn: int,
    ) -> tuple[str, bool, DiagnosticPlan]:
        """Run the QCC verification pipeline.

        Args:
            trace: Interaction trace list
            dialogue_history: Accumulated dialogue
            diagnostic_plan: Current diagnostic plan
            summarizer: Summarizer agent
            diagnostician: Diagnostician agent
            verifier: Verifier agent
            planner: Planner agent (for re-planning)
            qcc_doctor: QCC Doctor agent
            patient: Patient agent
            reporter: Reporter agent
            current_turn: Current turn count

        Returns:
            Tuple of (answer, should_continue, updated_plan)
        """
        # Summarizer phase
        s_instruction = self.prompts.get_summarizer_instruction(dialogue_history)
        s_response = summarizer.step(s_instruction)
        s_duration = summarizer.get_last_duration()

        trace.append({
            "role_id": "summarizer",
            "content": s_response,
            "duration": s_duration,
        })

        summary = self.extract_summary(s_response)

        # Diagnostician phase
        d_instruction = self.prompts.get_qcc_diagnostician_instruction(
            summary=summary,
            all_candidates=diagnostic_plan.get_all_candidates_str(),
            checklist=diagnostic_plan.get_checklist_str(),
        )
        d_response = diagnostician.step(d_instruction)
        d_duration = diagnostician.get_last_duration()

        trace.append({
            "role_id": "diagnostician",
            "content": d_response,
            "duration": d_duration,
        })

        _, answer = self.extract_action(d_response)
        if not answer:
            answer = d_response
        confidence = self.extract_confidence(d_response)

        # Verifier phase
        v_instruction = self.prompts.get_qcc_verifier_instruction(
            current_turns=current_turn,
            max_turns=self.max_turns,
            summary=summary,
            diagnosis=answer,
            confidence=confidence,
            checklist=diagnostic_plan.get_checklist_str(),
        )
        v_response = verifier.step(v_instruction)
        v_duration = verifier.get_last_duration()

        trace.append({
            "role_id": "verifier",
            "content": v_response,
            "duration": v_duration,
        })

        decision, feedback = self.extract_verifier_decision_qcc(v_response)
        turns_remaining = self.max_turns - current_turn

        # Handle different decisions
        if decision == "PASS":
            return answer, False, diagnostic_plan

        elif decision == "REPLAN" and turns_remaining > 3 and confidence == "UNCERTAIN":
            # Re-planning: generate new diagnostic plan
            replan_instruction = self.prompts.get_replan_instruction(
                original_candidates=diagnostic_plan.get_all_candidates_str(),
                collected_evidence=dialogue_history,
                feedback=feedback,
            )
            replan_response = planner.step(replan_instruction)
            replan_duration = planner.get_last_duration()

            trace.append({
                "role_id": "planner",
                "content": replan_response,
                "duration": replan_duration,
            })

            # Parse new plan
            new_plan = self.parse_planner_response(replan_response)
            if new_plan.candidates:
                diagnostic_plan = new_plan

            return answer, True, diagnostic_plan

        elif decision == "INCOMPLETE" and turns_remaining > 1:
            # Continue evidence collection with feedback
            fb_instruction = self.prompts.get_doctor_feedback_instruction(feedback)
            fb_response = qcc_doctor.step(fb_instruction)
            fb_duration = qcc_doctor.get_last_duration()

            trace.append({
                "role_id": "doctor_qcc",
                "content": fb_response,
                "duration": fb_duration,
            })

            action_type, action_content = self.extract_action(fb_response)

            # Route to simulator
            if action_type == "test":
                m_instruction = self.prompts.get_reporter_turn_instruction(action_content)
                m_response = reporter.step(m_instruction)
                m_duration = reporter.get_last_duration()

                trace.append({
                    "role_id": "reporter",
                    "content": m_response,
                    "duration": m_duration,
                })
            elif action_type == "query":
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()

                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })

            return answer, True, diagnostic_plan

        # Default: accept diagnosis (at turn limit or forced)
        return answer, False, diagnostic_plan

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _finalize_result(
        self,
        trace: list[dict],
        answer: str,
        current_turn: int,
        diagnostic_plan: DiagnosticPlan | None,
        agents: list[tuple[str, "AgentWrapper"]],
    ) -> ScenarioResult:
        """Finalize and return the scenario result.

        Args:
            trace: Interaction trace
            answer: Final diagnosis answer
            current_turn: Final turn count
            diagnostic_plan: Final diagnostic plan (if any)
            agents: List of (role_id, agent) tuples

        Returns:
            ScenarioResult with all metadata
        """
        role_records = self._collect_role_records(agents)

        metadata = {
            "mode": "qcc",
            "turns": current_turn,
            "max_turns": self.max_turns,
            "role_records": role_records,
        }

        if diagnostic_plan:
            metadata["diagnostic_plan"] = {
                "candidates": [
                    {
                        "diagnosis": c.diagnosis,
                        "gold_standard": c.gold_standard,
                        "is_pruned": c.is_pruned,
                        "prune_reason": c.prune_reason,
                    }
                    for c in diagnostic_plan.candidates
                ],
                "differential_evidence": [
                    {
                        "description": e.description,
                        "collected": e.collected,
                        "result": e.result,
                    }
                    for e in diagnostic_plan.differential_evidence
                ],
            }

        return ScenarioResult(
            answer=answer,
            trace=trace,
            metadata=metadata,
        )

    def _collect_role_records(
        self, agents: list[tuple[str, "AgentWrapper"]]
    ) -> list[dict]:
        """Collect conversation history from all agents."""
        records = []
        for role_id, agent in agents:
            history = agent.get_history()
            token_usage = agent.get_usage_stats()
            records.append({
                role_id: history,
                "token_usage": token_usage,
            })
        return records
