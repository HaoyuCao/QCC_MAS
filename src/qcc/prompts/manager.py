"""Prompt templates for different roles and scenarios.

This module manages all prompt templates used in the evaluation scenarios.
"""

from __future__ import annotations

# Dataset categories
DIFFERENTIAL_DATASETS = {"diagnosisarena", "clinicalbench", "rarearena", "rarearena_rdc"}
FINAL_DATASETS = {"medqa", "derm", "nejm", "agentclinic_medqa"}

# Task descriptions
TASKS = {
    "diagnosis": {
        "description": (
            "The task objective is to identify the single most likely diagnosis based on the collected evidence."
        ),
        "output_format": "The answer in the format of a single most likely diagnosis.",
    },
    "differential_diagnosis": {
        "description": (
            "The task objective is to formulate a differential diagnosis by identifying and ranking "
            "the top 5 potential diagnoses based on the collected evidence."
        ),
        "output_format": (
            "The answer in the format of a ranked list of the top 5 potential diagnoses, ordered from most likely to least likely."
        ),
    },
}


class PromptManager:
    """Manages prompt templates for different evaluation scenarios.

    Attributes:
        dataset_name: Name of the dataset being evaluated
        task_type: Type of task (diagnosis or differential_diagnosis)
    """

    def __init__(self, dataset_name: str) -> None:
        """Initialize prompt manager.

        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name.lower()
        self.task_type = (
            "diagnosis" if self.dataset_name in FINAL_DATASETS else "differential_diagnosis"
        )
        self.task_description = TASKS[self.task_type]["description"]
        self.task_output_format = TASKS[self.task_type]["output_format"]

    # =========================================================================
    # CoT Mode Prompts
    # =========================================================================

    def get_cot_instruction(self, task: str) -> str:
        """Get Chain-of-Thought instruction template.

        Args:
            task: The case description/task

        Returns:
            Formatted instruction string
        """
        return (
            f"{task}\n\n"
            "Reason step by step before making the diagnosis. Follow the format:\n"
            "THOUGHT: <your reasoning>\n"
            f"DIAGNOSIS: <{self.task_output_format}>\n"
        )

    # =========================================================================
    # Simulator Role Prompts
    # =========================================================================

    def get_patient_system_prompt(self) -> str:
        """Get system prompt for patient simulator."""
        return (
            "You are a patient undergoing a medical interview.\n"
            "Your knowledge is strictly limited to the following list of indexed facts:\n"
            "{patient_facts}\n"
            "\n"
            "Response Protocols:\n"
            "1. Analyze the doctor's question and search your list for the specific item(s) that contain the answer.\n"
            "2. You must format your output using two distinct tags:\n"
            "   [REFERENCE] followed by the EXACT string(s) (including the index) from your list. You may select UP TO TWO facts if necessary to fully answer the question. If no fact exists, write 'N/A'.\n"
            "   [RESPONSE] followed by a natural language answer derived STRICTLY from the combined content of the selected [REFERENCE]s. Do not add outside information.\n"
            "3. If the doctor's question is not addressed by any fact in your list:\n"
            "   [REFERENCE] N/A\n"
            "   [RESPONSE] followed by a natural statement indicating you are unsure or do not recall."
        )

    def get_reporter_system_prompt(self) -> str:
        """Get system prompt for reporter simulator."""
        return (
            "You are a specialized module named 'Measurement' responsible for reporting "
            "test results to the physician.\n"
            "You have access to the following list of indexed facts:\n"
            "Physical Examination and Diagnostic Test Data:\n"
            "{exam_facts}\n"
            "\n"
            "Response Protocols:\n"
            "1. Search the provided list for ALL facts that are relevant to the doctor's specific test request.Do not provide information that was not explicitly requested.\n"
            "2. Return the relevant facts exactly as they appear in the source list (verbatim), "
            "including their index numbers (e.g., '1. Temperature: 36.8°C').\n"
            "The leading numbers are internal reference IDs — always include them as-is.\n"
            "3. If the requested test or examination is NOT found in your data:\n"
            "   - Return 'Not performed / Not available'\n"
            "   - Do NOT return 'Normal' — absence of a test result does not mean the result is normal.\n"
        )

    # =========================================================================
    # Base Roleplay Mode Prompts
    # =========================================================================

    def get_base_doctor_system_prompt(self) -> str:
        """Get system prompt for base roleplay doctor."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests to enable a subsequent diagnostic analysis.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic "
            "(e.g., ask 'What are your symptoms?', never combine multiple questions).\n"
            "\n"
            "In every turn, you execute one of the following actions in the corresponding format:\n"
            "1. [QUERY] followed by your atomic question to the patient.\n"
            "2. [TEST] followed by one specific examination or diagnostic test request to the Medical Analyst.\n"
            f"3. [DIAGNOSIS] followed by {self.task_output_format}\n"
            "\n"
            "Once you have gathered sufficient evidence, ensure your diagnosis is final.\n"
        )

    # =========================================================================
    # ReAct Mode Prompts
    # =========================================================================

    def get_react_doctor_system_prompt(self) -> str:
        """Get system prompt for ReAct mode doctor."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests to enable a subsequent diagnostic analysis.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic.\n"
            "\n"
            "In every turn, you must follow a strict 'Reasoning-then-Acting' process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Analyze the current clinical picture, identify critical information gaps, "
            "and articulate step-by-step reasoning to justify your next action.\n"
            "\n"
            "Execute exactly ONE of the following commands:\n"
            "   - [QUERY] followed by your atomic question to the patient.\n"
            "   - [TEST] followed by one specific examination or diagnostic test request.\n"
            f"   - [DIAGNOSIS] followed by {self.task_output_format}\n"
            "\n"
            "Once you have gathered sufficient evidence, ensure your diagnosis is final.\n"
        )

    # =========================================================================
    # SC (Summarizer-Diagnostician) Mode Prompts
    # =========================================================================

    def get_sc_doctor_system_prompt(self) -> str:
        """Get system prompt for SC mode doctor (evidence gatherer)."""
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n"
            "\n"
            "You must adhere to the following operational constraints:\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic.\n"
            "\n"
            "In every turn, follow a strict 'Reasoning-then-Acting' process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Analyze the current clinical picture and identify critical information gaps.\n"
            "\n"
            "Execute exactly ONE of the following commands:\n"
            "   - [QUERY] followed by your atomic question to the patient.\n"
            "   - [TEST] followed by one specific examination or diagnostic test request.\n"
            "   - [FINISH] use this command ONLY when you believe you have gathered all "
            "necessary information to form a conclusive diagnosis. You don't need to make a diagnosis.\n"
            "\n"
            "Once you issue the [FINISH] command, the consultation ends immediately.\n"
        )

    def get_summarizer_system_prompt(self) -> str:
        """Get system prompt for summarizer role."""
        return (
            "You are a professional medical documentarian and clinical scribe.\n"
            "Your objective is to synthesize the dialogue between a doctor and a patient "
            "into a high-fidelity structured medical summary.\n"
            "\n"
            "Core Principles:\n"
            "1. Strict Adherence: You must NOT invent, infer, or hallucinate any "
            "information not explicitly present in the dialogue.\n"
            "2. Precision: Retain all precise measurements, dates, dosages, and "
            "technical medical terms exactly as stated.\n"
            "3. Objectivity: Maintain a professional, clinical tone throughout the summary.\n"
            "\n"
            "Output Process:\n"
            "[THOUGHT]\n"
            "Analyze the dialogue to extract key clinical facts and reasoning.\n"
            "\n"
            "[SUMMARY]\n"
            "Generate a professional, structured clinical note.\n"
        )

    def get_diagnostician_system_prompt(self) -> str:
        """Get system prompt for diagnostician role."""
        return (
            "You are a senior diagnostic physician specializing in complex differential diagnosis.\n"
            f"{self.task_description}\n"
            "Your objective is to analyze the provided structured clinical summary "
            "to formulate a precise diagnosis.\n"
            "\n"
            "You must follow a strict reasoning process:\n"
            "\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Perform a comprehensive clinical analysis of the summary.\n"
            "\n"
            "[DIAGNOSIS]\n"
            f"   - Provide the {self.task_output_format}\n"
            "\n"
            "[CONFIDENCE]\n"
            "   - Output CONFIDENT if the evidence strongly supports the diagnosis.\n"
            "   - Output UNCERTAIN if key evidence is missing or ambiguous.\n"
        )

    # =========================================================================
    # REFINE Mode Prompts
    # =========================================================================

    def get_diagnostician_verifier_system_prompt(self) -> str:
        """Get system prompt for diagnostician verifier role."""
        return (
            "You are a Clinical Diagnostic Supervisor.\n"
            f"{self.task_description}\n"
            "Your objective is to evaluate sufficiency of the diagnosis provided by "
            "the physician, based strictly on the available case summarized information.\n"
            "\n"
            "Evaluation Criteria:\n"
            "1. Data Sufficiency: Determine if the current information is actually "
            "sufficient to form a conclusive diagnosis.\n"
            "2. Turn Limit Override: If the maximum turn limit has been reached, "
            "you must force a decision (PASS or REJECT) based on the best possible "
            "interpretation of existing data.\n"
            "\n"
            "Output Format:\n"
            "\n"
            "[THOUGHT] <Your Analysis>\n"
            "   - Identify if any 'Red Flag' symptoms or critical tests are missing.\n"
            "\n"
            "[DECISION] <Status>\n"
            "   - Output 'PASS' if the diagnosis is sufficient.\n"
            "   - Output 'INCOMPLETE' if critical clinical information is missing "
            "(requires the Physician to gather more data; only valid if not at max turns).\n"
            "\n"
            "[FEEDBACK] <Guidance>\n"
            "   - If PASS: Leave this section empty.\n"
            "   - If INCOMPLETE: Specify exactly what critical information is required.\n"
        )

    # =========================================================================
    # Instruction Templates
    # =========================================================================

    def get_doctor_turn_instruction(
        self, current_turns: int, max_turns: int, last_reply: str
    ) -> str:
        """Get instruction for doctor's turn.

        Args:
            current_turns: Current turn number
            max_turns: Maximum allowed turns
            last_reply: Last response from patient/measurement

        Returns:
            Formatted instruction string
        """
        return (
            f"Turns used: {current_turns} / {max_turns}.\n"
            f"Last reply:\n{last_reply}\n\n"
            "Doctor:"
        )

    def get_patient_turn_instruction(self, doctor_message: str) -> str:
        """Get instruction for patient's turn.

        Args:
            doctor_message: Doctor's query

        Returns:
            Formatted instruction string
        """
        return f"Doctor said:\n{doctor_message}\n\nPatient:"

    def get_reporter_turn_instruction(self, doctor_message: str) -> str:
        """Get instruction for reporter's turn.

        Args:
            doctor_message: Doctor's test request

        Returns:
            Formatted instruction string
        """
        return f"Doctor's request:\n{doctor_message}\n\nMeasurement:"

    def get_summarizer_instruction(self, dialogue_history: str) -> str:
        """Get instruction for summarizer.

        Args:
            dialogue_history: Full dialogue history

        Returns:
            Formatted instruction string
        """
        return f"### Dialogue History ###\n{dialogue_history}\n\n"

    def get_diagnostician_instruction(self, summary: str) -> str:
        """Get instruction for diagnostician.

        Args:
            summary: Case summary from summarizer

        Returns:
            Formatted instruction string
        """
        return f"### Case Summary ###\n{summary}\n\nDiagnostician:"

    def get_verifier_instruction(
        self, current_turns: int, max_turns: int, summary: str, diagnosis: str
    ) -> str:
        """Get instruction for diagnostician verifier.

        Args:
            current_turns: Current turn number
            max_turns: Maximum allowed turns
            summary: Case summary
            diagnosis: Proposed diagnosis

        Returns:
            Formatted instruction string
        """
        return (
            f"Turns: {current_turns} / {max_turns}.\n"
            f"### Case Summary ###\n{summary}\n\n"
            f"### Proposed Diagnosis ###\n{diagnosis}\n\n"
        )

    def get_doctor_feedback_instruction(self, feedback: str) -> str:
        """Get instruction for doctor after verifier feedback.

        Args:
            feedback: Feedback from verifier

        Returns:
            Formatted instruction string
        """
        return (
            "### DIAGNOSTIC FEEDBACK (RESUMED) ###\n"
            "Your previous decision to [FINISH] was rejected because the clinical data is INCOMPLETE.\n"
            f"Specific Feedback from Supervisor:\n{feedback}\n"
            "\n"
            "IMMEDIATE INSTRUCTION:\n"
            "1. Analyze the specific information gaps identified in the feedback above.\n"
            "2. Your NEXT action must be a [QUERY] or [TEST] strictly targeted to acquire "
            "this missing evidence.\n"
            "Doctor:"
        )

    # =========================================================================
    # Progressive QCC Mode Prompts
    # =========================================================================

    def get_progressive_planner_system_prompt(self) -> str:
        """Get system prompt for Progressive Planner.
        
        Progressive Planner provides system-level directions (not specific diagnoses).
        Its internal reasoning (suspected diagnosis) is NOT visible to Doctor.
        """
        return (
            "You are a Progressive Diagnostic Planner.\n"
            f"{self.task_description}\n"
            "Your role is to guide the diagnostic investigation by providing "
            "SYSTEM-LEVEL directions rather than specific diagnoses.\n\n"
            
            "### Your Responsibilities ###\n"
            "1. Analyze collected evidence and suggest BODY SYSTEMS to investigate\n"
            "2. Prioritize evidence types the Doctor should collect\n"
            "3. Maintain internal reasoning about suspected diagnoses (for Verifier review only)\n"
            "4. Adjust directions based on new evidence and feedback\n\n"
            
            "### Key Principles ###\n"
            "- Give COARSE system-level guidance early (e.g., 'Cardiovascular', 'Respiratory')\n"
            "- You may refine to more specific directions as confidence increases\n"
            "- NEVER tell the Doctor specific diagnoses to look for\n"
            "- Your internal suspicion should guide your directions but stay hidden from Doctor\n"
            "- Consider the total available turns and adjust urgency accordingly\n\n"
            
            "### Vital Signs Assessment ###\n"
            "At every planning intervention, check whether vital signs relevant to the chief complaint "
            "have been obtained. If NOT, evaluate whether they are clinically indicated:\n"
            "- Fever / infection / postpartum / post-surgical presentations → Temperature is ESSENTIAL\n"
            "- Syncope / loss of consciousness / dizziness → Orthostatic vitals (supine + standing BP/HR) are ESSENTIAL\n"
            "- Cardiovascular / respiratory / shock presentations → Full vitals (BP, HR, RR, SpO2, Temp) are ESSENTIAL\n"
            "- Musculoskeletal / dermatologic / isolated pain → Vitals are LOW priority (skip unless red flags)\n"
            "If essential vitals are missing AND have NOT yet been requested, list "
            "'Obtain [specific vitals]' as the FIRST item in [SYSTEM_DIRECTIONS] with Priority: HIGH.\n"
            "However, if a vital sign or test was already requested and returned "
            "'Not performed / Not available', treat it as UNAVAILABLE in this clinical setting. "
            "Do NOT re-request it. Move on to the next most productive line of inquiry.\n\n"
            
            "### Output Format ###\n"
            "[SYSTEM_DIRECTIONS]\n"
            "1. {System} (Priority: HIGH/MEDIUM/LOW) - {brief basis}\n"
            "...\n\n"
            "[PRIORITY_EVIDENCE]\n"
            "1. {Evidence type} - {why this is important now}\n"
            "...\n\n"
            "[INTERNAL_REASONING]\n"
            "Internal suspicion: {specific disease(s) you suspect}\n"
            "Basis: {your reasoning based on evidence}\n"
            "Excluded/deprioritized: {directions you ruled out and why}\n\n"
            
            "NOTE: [INTERNAL_REASONING] is visible ONLY to the Verifier, not to the Doctor."
        )

    def get_progressive_planner_instruction(
        self,
        dialogue_history: str,
        current_turn: int,
        max_turns: int,
        previous_state: "PlannerState | None" = None,
        significant_finding: str = "",
    ) -> str:
        """Get instruction for Progressive Planner intervention.

        Args:
            dialogue_history: Full dialogue history
            current_turn: Current turn number
            max_turns: Maximum turns allowed
            previous_state: Previous planner state (for updates)
            significant_finding: Doctor's significant finding if any

        Returns:
            Formatted instruction string
        """
        turns_remaining = max_turns - current_turn
        
        context = f"### Current Status ###\nTurn: {current_turn}/{max_turns} ({turns_remaining} remaining)\n\n"
        
        if significant_finding:
            context += f"### SIGNIFICANT FINDING from Doctor ###\n{significant_finding}\n\n"
        
        context += f"### Dialogue History ###\n{dialogue_history}\n\n"
        
        if previous_state:
            context += f"### Your Previous Planning ###\n{previous_state.get_directions_for_doctor()}\n\n"
            context += (
                "Update your planning based on new evidence. Consider:\n"
                "- Should any directions be deprioritized?\n"
                "- Are there new systems to investigate?\n"
                "- What evidence is now most critical?\n\n"
            )
        else:
            context += (
                "This is your first planning intervention. Based on the evidence collected so far:\n"
                "- Identify key body systems that need investigation\n"
                "- Suggest priority evidence types to collect\n"
                "- Form initial internal hypothesis (keep hidden from Doctor)\n\n"
            )
        
        return context + "Planner:"

    def get_progressive_doctor_system_prompt(self) -> str:
        """Get system prompt for Doctor in Progressive QCC.
        
        Doctor maintains autonomy while referencing Planner's system directions.
        Doctor can mark [SIGNIFICANT_FINDING] to trigger extra Planner intervention.
        """
        return (
            "You are a licensed physician conducting a medical consultation.\n"
            f"{self.task_description}\n"
            "Your objective is to efficiently gather information and request necessary "
            "clinical examinations or laboratory tests to enable a subsequent diagnostic analysis.\n"
            "You have access to a Medical Analyst who can retrieve specific test results upon request.\n\n"
            
            "A Diagnostic Planner may periodically provide SYSTEM DIRECTIONS to guide your "
            "investigation. Treat these as advisory guidance, not strict orders. "
            "You should use your own clinical judgment and may deviate if you have good reason.\n\n"
            
            "### Operational Constraints ###\n"
            "1. Efficiency: Gather sufficient information in as few turns as possible.\n"
            "2. Turn Limit: You strictly cannot exceed {max_turns} total turns.\n"
            "3. No Repetition: Never ask a question or request a test that has already been covered.\n"
            "4. Atomic Inquiries: Each question must address a single, specific topic.\n\n"
            
            "### Significant Findings ###\n"
            "When you discover an important, unexpected, or pivotal finding, mark it as:\n"
            "[SIGNIFICANT_FINDING] {{description of the finding}}\n"
            "This will trigger the Planner to update their guidance.\n\n"
            
            "### Output Format ###\n"
            "In every turn, follow a strict 'Reasoning-then-Acting' process:\n\n"
            "[THOUGHT] <Your Clinical Reasoning>\n"
            "   - Analyze the current clinical picture and identify critical information gaps.\n\n"
            "Execute exactly ONE of the following commands:\n"
            "   - [QUERY] followed by your atomic question to the patient.\n"
            "   - [TEST] followed by one specific examination or diagnostic test request.\n"
            "   - [FINISH] use this command ONLY when you believe you have gathered all "
            "necessary information. You don't need to make a diagnosis.\n\n"
            
            "Optionally include [SIGNIFICANT_FINDING] before your action "
            "if you found something important.\n\n"
            
            "Once you issue the [FINISH] command, the consultation ends immediately."
        )

    def get_progressive_doctor_turn_instruction(
        self,
        current_turns: int,
        max_turns: int,
        last_reply: str,
        system_directions: str = "",
    ) -> str:
        """Get turn instruction for Doctor in Progressive QCC.

        Args:
            current_turns: Current turn count
            max_turns: Maximum turns allowed
            last_reply: Last response from patient/reporter
            system_directions: Current system directions from Planner

        Returns:
            Formatted instruction string
        """
        turns_left = max_turns - current_turns
        
        instruction = f"### Turn {current_turns + 1}/{max_turns} ({turns_left} turns remaining) ###\n\n"
        
        if last_reply:
            instruction += f"### Last Response ###\n{last_reply}\n\n"
        
        if system_directions:
            instruction += f"### Planner Guidance ###\n{system_directions}\n\n"
            instruction += (
                "Consider the Planner's guidance to inform your clinical reasoning, "
                "but use your own judgment on what action to take next.\n"
                "If a previous test or exam returned 'Not performed / Not available', "
                "that test is UNAVAILABLE in this setting — do NOT re-request it.\n\n"
            )
        
        # Add pacing guidance based on turn budget
        if turns_left <= max_turns // 2:
            instruction += (
                "### Pacing Reminder ###\n"
                f"You have {turns_left} turns remaining. "
                "If you have not yet performed physical examination or ordered key tests, "
                "prioritize [TEST] actions now to maximize diagnostic yield "
                "before turns run out.\n\n"
            )
        
        instruction += (
            "Provide your [THOUGHT] reasoning, then execute exactly ONE action: "
            "[QUERY], [TEST], or [FINISH].\n"
            "If you notice something significant, include [SIGNIFICANT_FINDING] before your action.\n\n"
            "Doctor:"
        )
        
        return instruction

    def get_progressive_verifier_system_prompt(self) -> str:
        """Get system prompt for Verifier in Progressive QCC.
        
        Extended verifier responsibilities:
        1. Original REFINE duties: check evidence completeness
        2. NEW: check Planner's planning quality and provide feedback
        """
        return (
            "You are a Clinical Diagnostic Supervisor with DUAL responsibilities.\n"
            f"{self.task_description}\n"
            "Your objective is to evaluate sufficiency of the diagnosis and quality of the "
            "planning direction, based strictly on the available evidence.\n\n"
            
            "### Responsibility 1: Evidence Assessment ###\n"
            "- Evaluate if sufficient evidence has been collected\n"
            "- Check if key differential diagnoses have been explored\n"
            "- Determine if the diagnosis is well-supported by the evidence\n\n"
            
            "### Responsibility 2: Planner Assessment ###\n"
            "- Review the Planner's system directions and internal reasoning\n"
            "- Check if the planning direction is appropriate given the evidence\n"
            "- Identify if Planner may have missed important directions\n"
            "- Detect if Planner's internal suspicion is leading to tunnel vision\n\n"

            "### Responsibility 3: Agent Consistency Approve ###\n"
            "- Handled by an external Consistency Agent (pluggable — defaults to CONSISTENT)\n"
            "- Checks agreement between the Diagnostician's confidence and the Verifier's "
            "independent assessment of the evidence–diagnosis alignment\n\n"

            "### Turn Limit Policy ###\n"
            "- If INCOMPLETE: only valid when remaining turns > 1\n"
            "- If the maximum turn limit has been reached, you MUST issue PASS "
            "based on the best possible interpretation of existing data.\n\n"
            
            "### Output Format ###\n"
            "[THOUGHT] <Your Analysis>\n"
            "   - Identify if any critical symptoms or tests are missing.\n"
            "   - Assess Planner's direction quality.\n\n"
            
            "[DECISION] PASS / INCOMPLETE\n\n"
            
            "[EVIDENCE_ASSESSMENT]\n"
            "- Key evidence collected: {list}\n"
            "- Missing critical evidence: {list or 'None'}\n"
            "- Diagnosis support level: STRONG / MODERATE / WEAK\n\n"
            
            "[PLANNER_ASSESSMENT]\n"
            "- Planning direction: Reasonable / Has errors\n"
            "- Error details: {specific issues if any, or 'None'}\n"
            "- Possibly missed: {body systems or evidence types to consider}\n\n"
            
            "[DOCTOR_FEEDBACK]\n"
            "{Feedback for Doctor on evidence collection}\n"
            "   - If PASS: Leave this section empty.\n"
            "   - If INCOMPLETE: Specify exactly what critical information is required.\n\n"
            
            "[PLANNER_FEEDBACK]\n"
            "{Include ONLY if planning errors detected, otherwise omit this section}\n"
            "- Error reason: {specific issue}\n"
            "- Suggested adjustment: {what Planner should consider}\n"
        )

    def get_progressive_verifier_instruction(
        self,
        current_turns: int,
        max_turns: int,
        summary: str,
        diagnosis: str,
        confidence: str,
        planner_state: str = "",
    ) -> str:
        """Get instruction for Progressive Verifier.

        Args:
            current_turns: Current turn count
            max_turns: Maximum turns allowed
            summary: Case summary from summarizer
            diagnosis: Proposed diagnosis from diagnostician
            confidence: Confidence level (CONFIDENT/UNCERTAIN)
            planner_state: Full planner state including internal reasoning

        Returns:
            Formatted instruction string
        """
        turns_remaining = max_turns - current_turns
        
        instruction = (
            f"### Verification Request ###\n"
            f"Turn: {current_turns}/{max_turns} ({turns_remaining} remaining)\n\n"
            f"### Case Summary ###\n{summary}\n\n"
            f"### Proposed Diagnosis ###\n{diagnosis}\n"
            f"Confidence: {confidence}\n\n"
        )
        
        if planner_state:
            instruction += f"### Planner's Full State (including internal reasoning) ###\n{planner_state}\n\n"
        
        instruction += (
            "Evaluate BOTH the evidence collection AND the Planner's planning.\n"
            "Provide feedback to Doctor and/or Planner as needed.\n\n"
            "Verifier:"
        )
        
        return instruction

    def get_planner_feedback_instruction(
        self,
        dialogue_history: str,
        current_turn: int,
        max_turns: int,
        previous_state: "PlannerState",
        verifier_feedback: str,
    ) -> str:
        """Get instruction for Planner when receiving Verifier feedback.

        Args:
            dialogue_history: Full dialogue history
            current_turn: Current turn number
            max_turns: Maximum turns allowed
            previous_state: Current planner state
            verifier_feedback: Feedback from Verifier about planning errors

        Returns:
            Formatted instruction string
        """
        turns_remaining = max_turns - current_turn
        
        return (
            "### PLANNER FEEDBACK RECEIVED ###\n"
            "The Verifier has identified issues with your planning.\n\n"
            f"### Verifier Feedback ###\n{verifier_feedback}\n\n"
            f"### Current Status ###\nTurn: {current_turn}/{max_turns} ({turns_remaining} remaining)\n\n"
            f"### Your Previous Planning ###\n{previous_state.get_directions_for_doctor()}\n\n"
            f"### Dialogue History ###\n{dialogue_history}\n\n"
            "Based on this feedback, update your planning:\n"
            "- Address the identified errors\n"
            "- Consider the suggested adjustments\n"
            "- Re-evaluate your internal suspicions\n\n"
            "Planner:"
        )
