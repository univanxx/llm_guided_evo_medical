from helper import Expert, log_info_expert, log_info, get_responses, parse_confidence_score, expert_response_choice, question_generation, run_mediq
from typing import List, Tuple

PROGRAM_PARAMS = {
    "self_consistency": 3,
    "max_questions": 8,
}


PROB_THRESHOLD = 0.2


expert_system = {
    "system_msg": "You are a medical doctor trying to reason through a real-life clinical case. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, respond according to the task specified by the user. Base your response on the current and standard practices referenced in medical guidelines.",

    "question_word": "Doctor Question",
    "answer_word": "Patient Response",

    "abstention_prompt": "Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. What is your confidence score to pick the correct option to the inquiry factually using the conversation log? Answer strictly in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nSCORE: your confidence score written as a float from 0.0 to 1.0.",

    "question_prompt": "If there are missing features that prevent you from picking a confident and factual answer to the inquiry, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision. You can ask about any relevant information about the patient's case, such as family history, tests and exams results, treatments already done, etc. Consider what are the common questions asked in the specific subject relating to the patient's known symptoms, and what the best and most intuitive doctor would ask. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. Make sure to NOT repeat any questions from the above conversation log. Answer in the following format:\nATOMIC QUESTION: the atomic question and NOTHING ELSE.\nATOMIC QUESTION: ",

    "answer": "Assume that you already have enough information from the above question-answer pairs to answer the patient inquiry, use the above information to produce a factual conclusion. Respond with the correct letter choice (A, B, C, or D) and NOTHING ELSE.\nLETTER CHOICE: ",

    "curr_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:

PATIENT INFORMATION: {0}
QUESTION: {2}
OPTIONS: {3}
YOUR TASK: {4}

CONVERSATION LOG:
{1}"""
    
}


def expert_response_confidence_score(messages, **kwargs):
    """
    Numerical Abstain
    """
    log_info(f"++++++++++++++++++++ Start of Numerical Confidence Score [py:expert_response_confidence_score()] ++++++++++++++++++++")
    log_info(f"[<CONF SCORE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

    conf_scores, response_texts = [], {}
    n = PROGRAM_PARAMS["self_consistency"]
    responses = get_responses(messages, n=n, **kwargs)
    for i, response_text in enumerate(responses):
        log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
        if not response_text:
            log_info("[<CONF SCORE LM RES>]: " + "No response.")
            continue
        log_info("[<CONF SCORE LM RES>]: " + response_text)

        conf_score = parse_confidence_score(response_text)
        conf_scores.append(conf_score)
        response_texts[conf_score] = response_text
        log_info(f"[<CONF SCORE PARSED>]: {conf_score}")
    
    if len(conf_scores) > 0:
        avg_conf_score = sum(conf_scores) / len(conf_scores)
        # response_text = "CONFIDENCE SCORE: " + str(avg_conf_score)
        temp = [abs(r - avg_conf_score) for r in conf_scores]
        response_text = response_texts[conf_scores[temp.index(min(temp))]]
    else:
        avg_conf_score, response_text = 0, "No response."
    log_info(f"[<CONF SCORE RETURN>] (average conf score): {avg_conf_score}")
    return response_text, avg_conf_score


def numcutoff_abstention_decision(patient_state, inquiry, options_dict, abstain_threshold, **kwargs):
    """
    Numcutoff abstention strategy based on the current patient state.
    This function prompts the model to produce a numerical confidence score of how confident it is in its decision, then decide abstention based on arbitrarily set threshold
    """
    if abstain_threshold is None:
        abstain_threshold = PROB_THRESHOLD

    abstain_task_prompt = expert_system["abstention_prompt"]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{expert_system['question_word']}: {qa['question']}\n{expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": expert_system["system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score = expert_response_confidence_score(
        messages, abstain_threshold=abstain_threshold, **kwargs
    )
    abstain_decision = conf_score < abstain_threshold
    log_info_expert(f"[ABSTENTION PROMPT]: {messages}")
    log_info_expert(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second phase: only get a final answer if we choose NOT to abstain
    if not abstain_decision:
        prompt_answer = expert_system["curr_template"].format(
            patient_info,
            conv_log if conv_log != '' else 'None',
            inquiry,
            options_text,
            expert_system["answer"],
        )
        messages_answer = [
            {"role": "system", "content": expert_system["system_msg"]},
            {"role": "user", "content": prompt_answer},
        ]
        response_text, letter_choice = expert_response_choice(
            messages_answer, options_dict, **kwargs
        )
        log_info_expert(
            f"[NUMCUTOFF RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, "
            f"letter_choice: {letter_choice}\n"
        )
        return {
            "abstain": abstain_decision,
            "confidence": conf_score,
            "messages": messages,
            "letter_choice": letter_choice,
        }

    # If abstaining, do not compute an intermediate answer
    log_info_expert(
        f"[NUMCUTOFF RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, "
        f"letter_choice: None\n"
    )
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "messages": messages,
        "letter_choice": None,
    }


def _numcutoff_force_final_choice(patient_state, inquiry, options_dict, **kwargs):
    """One call to get letter when max_questions reached and letter was None.
       NOTE: The global question budget (max_questions) is fixed by the benchmark environment.
       Do NOT introduce your own question-count limits.
    """
    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{expert_system['question_word']}: {qa['question']}\n{expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    prompt_answer = expert_system["curr_template"].format(
        patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, expert_system["answer"],
    )
    messages = [
        {"role": "system", "content": expert_system["system_msg"]},
        {"role": "user", "content": prompt_answer},
    ]
    _, letter_choice = expert_response_choice(messages, options_dict, **kwargs)
    return letter_choice


class CustomExpert(Expert):
    def respond(self, patient_state):
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = numcutoff_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"], expert_system["question_prompt"], question_generation)
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
        }

    def force_final_choice(self, patient_state):
        kwargs = self.get_abstain_kwargs(patient_state)
        kwargs = {k: v for k, v in kwargs.items() if k not in ("patient_state", "inquiry", "options_dict")}
        return _numcutoff_force_final_choice(patient_state, self.inquiry, self.options, **kwargs)


def entrypoint(context=None) -> Tuple[List, List[str], List[int], List[str], dict]:
    seed = context.get("seed") if context else None
    return run_mediq(CustomExpert, seed=seed, program_params=PROGRAM_PARAMS)
