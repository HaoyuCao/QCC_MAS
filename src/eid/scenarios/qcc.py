"""Progressive QCC (Quality Controlled Consultation) scenario.

A progressive planning approach where a Planner agent provides system-level
diagnostic directions (not specific diagnoses) to guide the Doctor's
investigation. Extends REFINE with periodic Planner intervention, significant
finding triggers, and extended verification with Planner feedback.
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

# =============================================================================
# Progressive QCC Data Structures
# =============================================================================

@dataclass
class SystemDirection:
    """A system-level diagnostic direction from Planner.
    
    Unlike CandidateDiagnosis which specifies exact diagnoses,
    this represents a body system or organ system to investigate.
    """
    
    system: str  # e.g., "Cardiovascular", "Respiratory", "Neurological"
    priority: str  # "HIGH", "MEDIUM", "LOW"
    basis: str  # Why this direction is suggested
    is_deprioritized: bool = False
    deprioritize_reason: str = ""


@dataclass
class PriorityEvidence:
    """An evidence type that Planner suggests Doctor should collect."""
    
    evidence_type: str  # e.g., "cardiac enzymes", "chest imaging"
    reason: str  # Why this evidence is important
    associated_system: str = ""  # Which system direction this relates to


@dataclass
class PlannerState:
    """Container for Progressive Planner's state.
    
    Key differences from DiagnosticPlan:
    - system_directions: Coarse system-level directions (not specific diagnoses)
    - priority_evidence: Evidence types to collect (not a checklist to complete)
    - internal_reasoning: Planner's private reasoning (visible only to Verifier)
    - planning_history: Record of all planning updates for context
    """
    
    system_directions: list[SystemDirection] = field(default_factory=list)
    priority_evidence: list[PriorityEvidence] = field(default_factory=list)
    internal_reasoning: str = ""  # Planner's suspected diagnosis - NOT visible to Doctor
    internal_excluded: str = ""  # What Planner has deprioritized and why
    planning_history: list[dict] = field(default_factory=list)  # [{turn: int, content: str}]
    
    def get_active_directions(self) -> list[tuple[int, SystemDirection]]:
        """Return list of (index, direction) for non-deprioritized directions."""
        return [(i, d) for i, d in enumerate(self.system_directions) if not d.is_deprioritized]
    
    def get_directions_for_doctor(self) -> str:
        """Format system directions visible to Doctor."""
        lines = ["[SYSTEM_DIRECTIONS]"]
        for i, d in enumerate(self.system_directions):
            if d.is_deprioritized:
                status = " [DEPRIORITIZED]"
                lines.append(f"{i+1}. {d.system} (Priority: {d.priority}){status}")
                lines.append(f"   - Reason: {d.deprioritize_reason}")
            else:
                lines.append(f"{i+1}. {d.system} (Priority: {d.priority})")
                lines.append(f"   - Basis: {d.basis}")
        
        lines.append("")
        lines.append("[PRIORITY_EVIDENCE]")
        for i, e in enumerate(self.priority_evidence):
            lines.append(f"{i+1}. {e.evidence_type}")
            lines.append(f"   - Why: {e.reason}")
        
        return "\n".join(lines)
    
    def get_full_state_for_verifier(self) -> str:
        """Format full state including internal reasoning for Verifier."""
        lines = [self.get_directions_for_doctor()]
        lines.append("")
        lines.append("[INTERNAL_REASONING]")
        lines.append(f"Internal suspicion: {self.internal_reasoning}")
        lines.append(f"Excluded/deprioritized: {self.internal_excluded}")
        return "\n".join(lines)
    
    def deprioritize_direction(self, index: int, reason: str) -> bool:
        """Mark a direction as deprioritized (done by Planner, not Doctor)."""
        if 0 <= index < len(self.system_directions) and not self.system_directions[index].is_deprioritized:
            self.system_directions[index].is_deprioritized = True
            self.system_directions[index].deprioritize_reason = reason
            return True
        return False
    
    def add_planning_record(self, turn: int, content: str) -> None:
        """Record a planning update for history tracking."""
        self.planning_history.append({"turn": turn, "content": content})


# =============================================================================
# Progressive QCC Scenario
# =============================================================================

class ProgressiveQCCScenario(BaseScenario):
    """Progressive QCC (Quality Controlled Consultation) evaluation scenario.

    A progressive planning approach with:
    - Planner agent: provides system-level directions (not specific diagnoses)
    - Periodic intervention: Planner intervenes every ~3 turns
    - Significant finding triggers: Doctor can mark [SIGNIFICANT_FINDING] to trigger extra planning
    - Extended verification: Verifier checks both evidence and planning quality
    - Independent diagnosis: Diagnostician generates final diagnosis same as REFINE
    
    Key differences from original QCC:
    - Planner does NOT generate candidate diagnoses visible to Doctor
    - Doctor does NOT prune candidates (Planner handles direction changes)
    - Planner intervenes multiple times, not just once after initial turns
    - Verifier provides feedback to both Doctor AND Planner
    """
    
    # Planner intervention schedule: every N turns
    PLANNER_INTERVAL = 3
    
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
        free_exploration_turns: int = 3,
    ) -> None:
        """Initialize Progressive QCC scenario.

        Args:
            dataset_name: Name of the dataset
            doctor_config: Model configuration for doctor role
            patient_config: Model configuration for patient simulator
            reporter_config: Model configuration for reporter simulator
            max_turns: Maximum interaction turns
            summarizer_config: Model configuration for summarizer role
            diagnostician_config: Model configuration for diagnostician role
            verifier_config: Model configuration for verifier role
            planner_config: Model configuration for planner role
            free_exploration_turns: Number of free exploration turns before first planning
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
        self.free_exploration_turns = free_exploration_turns
        self.prompts = PromptManager(dataset_name)

    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _should_planner_intervene(self, current_turn: int, has_significant_finding: bool) -> bool:
        """Determine if Planner should intervene at this turn.
        
        Planner intervenes:
        - Every PLANNER_INTERVAL turns (3, 6, 9, 12, 15...)
        - When Doctor marks [SIGNIFICANT_FINDING]
        - When Verifier provides feedback (handled separately)
        """
        if current_turn < self.free_exploration_turns:
            return False
        
        if has_significant_finding:
            return True
        
        if current_turn > 0 and current_turn % self.PLANNER_INTERVAL == 0:
            return True
        
        return False

    # =========================================================================
    # Parsing Methods
    # =========================================================================

    @staticmethod
    def parse_progressive_planner_response(text: str) -> PlannerState:
        """Parse Progressive Planner response into PlannerState.
        
        Expected format:
        [SYSTEM_DIRECTIONS]
        1. {System} (Priority: HIGH/MEDIUM/LOW) - {basis}
        ...
        
        [PRIORITY_EVIDENCE]
        1. {Evidence type} - {why important}
        ...
        
        [INTERNAL_REASONING]
        Internal suspicion: {specific disease Planner suspects}
        Excluded/deprioritized: {directions and reasons}
        """
        state = PlannerState()
        
        # Extract system directions section
        directions_match = re.search(
            r"\[SYSTEM_DIRECTIONS\]\s*([\s\S]*?)(?=\[PRIORITY_EVIDENCE\]|\[INTERNAL_REASONING\]|\Z)",
            text
        )
        if directions_match:
            directions_text = directions_match.group(1)
            direction_pattern = re.compile(
                r"(\d+)\.\s*([^(\\n]+)\s*\(Priority:\s*(HIGH|MEDIUM|LOW)\)\s*[-:]?\s*(.*)",
                re.IGNORECASE
            )
            for match in direction_pattern.finditer(directions_text):
                state.system_directions.append(SystemDirection(
                    system=match.group(2).strip(),
                    priority=match.group(3).upper().strip(),
                    basis=match.group(4).strip(),
                ))
        
        # Extract priority evidence section
        evidence_match = re.search(
            r"\[PRIORITY_EVIDENCE\]\s*([\s\S]*?)(?=\[INTERNAL_REASONING\]|\Z)",
            text
        )
        if evidence_match:
            evidence_text = evidence_match.group(1)
            evidence_pattern = re.compile(r"(\d+)\.\s*([^-\\n]+)\s*[-:]?\s*(.*)")
            for match in evidence_pattern.finditer(evidence_text):
                state.priority_evidence.append(PriorityEvidence(
                    evidence_type=match.group(2).strip(),
                    reason=match.group(3).strip(),
                ))
        
        # Extract internal reasoning section
        internal_match = re.search(
            r"\[INTERNAL_REASONING\]\s*([\s\S]*)",
            text
        )
        if internal_match:
            internal_text = internal_match.group(1)
            suspicion_match = re.search(r"Internal suspicion:\s*(.+?)(?=Basis:|Excluded|$)", internal_text, re.IGNORECASE | re.DOTALL)
            if suspicion_match:
                state.internal_reasoning = suspicion_match.group(1).strip()
            excluded_match = re.search(r"Excluded[^:]*:\s*(.+)", internal_text, re.IGNORECASE | re.DOTALL)
            if excluded_match:
                state.internal_excluded = excluded_match.group(1).strip()
        
        return state

    @staticmethod
    def extract_significant_finding(text: str) -> tuple[bool, str]:
        """Extract [SIGNIFICANT_FINDING] marker from Doctor response."""
        finding_match = re.search(
            r"\[SIGNIFICANT_FINDING\]\s*(.+?)(?=\[|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if finding_match:
            return True, finding_match.group(1).strip()
        return False, ""

    @staticmethod
    def extract_verifier_decision_progressive(text: str) -> tuple[str, str, str]:
        """Extract decision and feedback from Progressive QCC verifier response.
        
        Returns:
            Tuple of (decision, doctor_feedback, planner_feedback)
        """
        decision_match = re.search(r"\[DECISION\]\s*(\w+)", text)
        doctor_fb_match = re.search(
            r"\[DOCTOR_FEEDBACK\]\s*([\s\S]*?)(?=\[PLANNER_FEEDBACK\]|$)",
            text
        )
        planner_fb_match = re.search(
            r"\[PLANNER_FEEDBACK\]\s*([\s\S]*)",
            text
        )

        decision = decision_match.group(1).upper() if decision_match else "PASS"
        if decision not in ("PASS", "INCOMPLETE"):
            decision = "PASS"
        
        doctor_feedback = doctor_fb_match.group(1).strip() if doctor_fb_match else ""
        planner_feedback = planner_fb_match.group(1).strip() if planner_fb_match else ""

        return decision, doctor_feedback, planner_feedback

    # =========================================================================
    # Confidence Extraction
    # =========================================================================

    @staticmethod
    def extract_confidence(text: str) -> str:
        """Extract confidence level from diagnostician response.

        Returns:
            'CONFIDENT' or 'UNCERTAIN'
        """
        match = re.search(r"\[CONFIDENCE\]\s*(\w+)", text, re.IGNORECASE)
        if match:
            level = match.group(1).upper()
            if level in ("CONFIDENT", "UNCERTAIN"):
                return level
        return "UNCERTAIN"

    # =========================================================================
    # Core Run Method
    # =========================================================================

    def run(self, case_input: CaseInput) -> ScenarioResult:
        """Execute Progressive QCC scenario on a single case.

        Flow:
        1. Free exploration phase (first N turns with no Planner)
        2. Planner intervenes periodically, on [SIGNIFICANT_FINDING],
           or on Verifier feedback
        3. Doctor acts with Planner's system directions as guidance
        4. Verification pipeline at [FINISH] or turn limit

        Args:
            case_input: Input data for the case

        Returns:
            ScenarioResult with diagnosis and trace
        """
        # Format facts for simulators
        patient_facts_str = "\n".join(case_input.patient_facts)
        exam_facts_str = "\n".join(case_input.exam_facts)

        # Create agents -------------------------------------------------------
        doctor = create_agent(
            role_id="doctor",
            system_prompt=self.prompts.get_progressive_doctor_system_prompt().format(
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
            system_prompt=self.prompts.get_progressive_planner_system_prompt(),
            config=self.planner_config,
            message_window_size=8,
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
            system_prompt=self.prompts.get_diagnostician_system_prompt(),
            config=self.diagnostician_config,
            message_window_size=3,
            summarize_threshold=90,
        )

        verifier = create_agent(
            role_id="diagnostician_verifier",
            system_prompt=self.prompts.get_progressive_verifier_system_prompt(),
            config=self.verifier_config,
            message_window_size=3,
            summarize_threshold=80,
        )

        # State ----------------------------------------------------------------
        trace: list[dict] = []
        dialogue_history = ""
        last_reply = ""
        answer = ""
        current_turn = 0
        planner_state: PlannerState | None = None

        # Main interaction loop ------------------------------------------------
        while current_turn <= self.max_turns:

            # --- Planner phase (before Doctor acts) --------------------------
            # Check last Doctor response for significant finding
            has_significant_finding = False
            significant_finding_text = ""
            if trace and trace[-1].get("role_id") == "doctor":
                has_significant_finding, significant_finding_text = (
                    self.extract_significant_finding(trace[-1]["content"])
                )

            if self._should_planner_intervene(current_turn, has_significant_finding):
                p_instruction = self.prompts.get_progressive_planner_instruction(
                    dialogue_history=dialogue_history,
                    current_turn=current_turn,
                    max_turns=self.max_turns,
                    previous_state=planner_state,
                    significant_finding=significant_finding_text,
                )
                p_response = planner.step(p_instruction)
                p_duration = planner.get_last_duration()

                trace.append({
                    "role_id": "planner",
                    "content": p_response,
                    "duration": p_duration,
                })

                planner_state = self.parse_progressive_planner_response(p_response)
                planner_state.add_planning_record(current_turn, p_response)

            # --- Doctor phase ------------------------------------------------
            system_directions = ""
            if planner_state is not None:
                system_directions = planner_state.get_directions_for_doctor()

            instruction = self.prompts.get_progressive_doctor_turn_instruction(
                current_turns=current_turn,
                max_turns=self.max_turns,
                last_reply=last_reply,
                system_directions=system_directions,
            )

            doctor_response = doctor.step(instruction)
            duration = doctor.get_last_duration()

            trace.append({
                "role_id": "doctor",
                "content": doctor_response,
                "duration": duration,
            })

            # Parse doctor action
            action_type, action_content = self.extract_action(doctor_response)

            # Update dialogue history
            dialogue_history += f"\nDoctor: {action_content}\n"

            # --- Finish / verification ----------------------------------------
            if action_type == "finish" or current_turn >= self.max_turns:
                answer, verified = self._run_progressive_verification_pipeline(
                    trace=trace,
                    dialogue_history=dialogue_history,
                    summarizer=summarizer,
                    diagnostician=diagnostician,
                    verifier=verifier,
                    doctor=doctor,
                    patient=patient,
                    reporter=reporter,
                    planner=planner,
                    planner_state=planner_state,
                    current_turn=current_turn,
                )

                if verified or current_turn >= self.max_turns - 1:
                    break

                # Verifier requested more evidence – continue
                current_turn += 1
                continue

            current_turn += 1

            # --- Route to simulator ------------------------------------------
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

        # Collect role records and return result
        return self._finalize_progressive_result(
            answer=answer,
            trace=trace,
            current_turn=current_turn,
            planner_state=planner_state,
            agents=[
                ("doctor", doctor),
                ("patient", patient),
                ("reporter", reporter),
                ("planner", planner),
                ("summarizer", summarizer),
                ("diagnostician", diagnostician),
                ("verifier", verifier),
            ],
        )

    # =========================================================================
    # Verification Pipeline
    # =========================================================================

    def _run_progressive_verification_pipeline(
        self,
        trace: list[dict],
        dialogue_history: str,
        summarizer: "AgentWrapper",
        diagnostician: "AgentWrapper",
        verifier: "AgentWrapper",
        doctor: "AgentWrapper",
        patient: "AgentWrapper",
        reporter: "AgentWrapper",
        planner: "AgentWrapper",
        planner_state: "PlannerState | None",
        current_turn: int,
    ) -> tuple[str, bool]:
        """Run summarizer -> diagnostician -> verifier pipeline.

        Extended from REFINE:
        - Passes planner state (including internal reasoning) to Verifier
        - Handles [PLANNER_FEEDBACK] from Verifier -> triggers Planner update

        Returns:
            (answer, verified) – verified=True means PASS or at turn limit
        """
        # --- Summarizer phase ------------------------------------------------
        s_instruction = self.prompts.get_summarizer_instruction(dialogue_history)
        s_response = summarizer.step(s_instruction)
        s_duration = summarizer.get_last_duration()

        trace.append({
            "role_id": "summarizer",
            "content": s_response,
            "duration": s_duration,
        })

        summary = self.extract_summary(s_response)

        # --- Diagnostician phase ---------------------------------------------
        d_instruction = self.prompts.get_diagnostician_instruction(summary)
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

        # --- Verifier phase --------------------------------------------------
        planner_full_state = ""
        if planner_state is not None:
            planner_full_state = planner_state.get_full_state_for_verifier()

        v_instruction = self.prompts.get_progressive_verifier_instruction(
            current_turns=current_turn,
            max_turns=self.max_turns,
            summary=summary,
            diagnosis=answer,
            confidence=confidence,
            planner_state=planner_full_state,
        )
        v_response = verifier.step(v_instruction)
        v_duration = verifier.get_last_duration()

        trace.append({
            "role_id": "diagnostician_verifier",
            "content": v_response,
            "duration": v_duration,
        })

        decision, doctor_feedback, planner_feedback = (
            self.extract_verifier_decision_progressive(v_response)
        )

        # --- Handle Planner feedback from Verifier ---------------------------
        if planner_feedback and planner_state is not None:
            pf_instruction = self.prompts.get_planner_feedback_instruction(
                dialogue_history=dialogue_history,
                current_turn=current_turn,
                max_turns=self.max_turns,
                previous_state=planner_state,
                verifier_feedback=planner_feedback,
            )
            pf_response = planner.step(pf_instruction)
            pf_duration = planner.get_last_duration()

            trace.append({
                "role_id": "planner",
                "content": pf_response,
                "duration": pf_duration,
            })

            # Update planner state with revised planning
            planner_state = self.parse_progressive_planner_response(pf_response)
            planner_state.add_planning_record(current_turn, pf_response)

        # --- Handle INCOMPLETE decision --------------------------------------
        if decision == "INCOMPLETE" and current_turn < self.max_turns - 1:
            # Doctor receives feedback and gathers more evidence
            fb_instruction = self.prompts.get_doctor_feedback_instruction(doctor_feedback)
            fb_response = doctor.step(fb_instruction)
            fb_duration = doctor.get_last_duration()

            trace.append({
                "role_id": "doctor",
                "content": fb_response,
                "duration": fb_duration,
            })

            # Parse doctor response to feedback
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
            else:
                p_instruction = self.prompts.get_patient_turn_instruction(action_content)
                p_response = patient.step(p_instruction)
                p_duration = patient.get_last_duration()

                trace.append({
                    "role_id": "patient",
                    "content": p_response,
                    "duration": p_duration,
                })

            return answer, False  # Not verified, continue gathering

        return answer, True  # PASS or at turn limit

    # =========================================================================
    # Finalization Helpers
    # =========================================================================

    def _finalize_progressive_result(
        self,
        answer: str,
        trace: list[dict],
        current_turn: int,
        planner_state: "PlannerState | None",
        agents: list[tuple[str, "AgentWrapper"]],
    ) -> ScenarioResult:
        """Build the final ScenarioResult with progressive metadata."""
        role_records = self._collect_role_records(agents)

        metadata: dict = {
            "mode": "qcc",
            "framework": "progressive",
            "turns": current_turn,
            "max_turns": self.max_turns,
            "role_records": role_records,
        }

        if planner_state is not None:
            metadata["planner_state"] = {
                "system_directions": [
                    {
                        "system": d.system,
                        "priority": d.priority,
                        "basis": d.basis,
                        "is_deprioritized": d.is_deprioritized,
                        "deprioritize_reason": d.deprioritize_reason,
                    }
                    for d in planner_state.system_directions
                ],
                "priority_evidence": [
                    {
                        "evidence_type": e.evidence_type,
                        "reason": e.reason,
                        "associated_system": e.associated_system,
                    }
                    for e in planner_state.priority_evidence
                ],
                "internal_reasoning": planner_state.internal_reasoning,
                "internal_excluded": planner_state.internal_excluded,
                "planning_history": planner_state.planning_history,
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
