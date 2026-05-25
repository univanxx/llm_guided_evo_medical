from helper import log_info, log_info_expert, get_responses, Expert, parse_atomic_question, run_mediq, parse_choice, expert_response_choice
from typing import List, Tuple
import random

PROGRAM_PARAMS = {
    "self_consistency": 3,
    "max_questions": 8,
}


expert_system = {
    "system_msg": "You are a medical doctor trying to reason through a real-life clinical case. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, respond according to the task specified by the user. Base your response on the current and standard practices referenced in medical guidelines.",

    "question_word": "Doctor Question",
    "answer_word": "Patient Response",
    
    "abstention_prompt": "Given the information so far, if you are confident to pick an option correctly and factually, respond in the format:\nREASON: a one-sentence explanation of why you are choosing a particular option.\nANSWER: the letter choice and NOTHING ELSE. Otherwise, if you are not confident to pick an option and need more information, ask ONE SPECIFIC ATOMIC QUESTION to the patient. The question should be bite-sized, NOT ask for too much at once, and NOT repeat what has already been asked. In this case, respond in the format:\nREASON: a one-sentence explanation of why you should ask the particular question.\nQUESTION: the atomic question and NOTHING ELSE.",

    "answer": "Assume that you already have enough information from the above question-answer pairs to answer the patient inquiry, use the above information to produce a factual conclusion. Respond with the correct letter choice (A, B, C, or D) and NOTHING ELSE.\nLETTER CHOICE: ",

    "curr_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:

PATIENT INFORMATION: {0}
QUESTION: {2}
OPTIONS: {3}
YOUR TASK: {4}

CONVERSATION LOG:
{1}"""
    
}   


def expert_response_choice_or_question(messages, options_dict, **kwargs):
    """
    Implicit Abstain
    """
    log_info(f"++++++++++++++++++++ Start of Implicit Abstention [py:expert_response_choice_or_question()] ++++++++++++++++++++")
    log_info(f"[<IMPLICIT ABSTAIN PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
    answers, questions, response_texts = [], [], {}
    n = PROGRAM_PARAMS["self_consistency"]
    responses = get_responses(messages, n=n, **kwargs)
    for i, response_text in enumerate(responses):
        log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
        if not response_text:
            log_info("[<IMPLICIT ABSTAIN LM RES>]: " + "No response --> Re-prompt")
            continue
        log_info("[<IMPLICIT ABSTAIN LM RES>]: " + response_text)
        response_text = response_text.replace("Confident --> Answer: ", "").replace("Not confident --> Doctor Question: ", "")

        if "?" not in response_text:
            letter_choice = parse_choice(response_text, options_dict)
            if letter_choice:
                log_info("[<IMPLICIT ABSTAIN PARSED>]: " + letter_choice)
                answers.append(letter_choice)
                response_texts[letter_choice] = response_text
        else:
            # not a choice, parse as question
            atomic_question = parse_atomic_question(response_text)
            if atomic_question:
                log_info("[<IMPLICIT ABSTAIN PARSED>]: " + atomic_question)
                questions.append(atomic_question)
                response_texts[atomic_question] = response_text
            
            else:
                log_info("[<IMPLICIT ABSTAIN PARSED>]: " + "FAILED TO PARSE --> Re-prompt")

    if len(answers) + len(questions) == 0:
        log_info("[<IMPLICIT ABSTAIN SC-PARSED>]: " + "No response.")
        return "No response.", None, None, 0.0

    conf_score = len(answers) / (len(answers) + len(questions))
    if len(answers) > len(questions): 
        final_answer = max(set(answers), key = answers.count)
        response_text = response_texts[final_answer]
        atomic_question = None
    else:
        final_answer = None
        rand_id = random.choice(range(len(questions)))
        atomic_question = questions[rand_id]
        response_text = response_texts[atomic_question]
    log_info(f"[<IMPLICIT ABSTAIN RETURN>]: atomic_question: {atomic_question}, final_answer: {final_answer}, conf_score: {conf_score} ([{len(answers)} : {len(questions)}])")
    return response_text, atomic_question, final_answer, conf_score


def implicit_abstention_decision(patient_state, inquiry, options_dict, **kwargs):
    """
    Implicit abstention strategy based on the current patient state.
    This function uses the expert system to make a decision on whether to abstain or not based on the current patient state.
    """
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
    response_text, atomic_question, letter_choice, conf_score = expert_response_choice_or_question(messages, options_dict, **kwargs)
    log_info_expert(f"[ABSTENTION PROMPT]: {messages}")
    log_info_expert(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    if atomic_question != None: abstain_decision = True  # if the model generates a question, it is abstaining from answering, therefore abstain decision is True
    elif letter_choice != None: abstain_decision = False  # if the model generates an answer, it is not abstaining from answering, therefore abstain decision is False
    else: abstain_decision = True  # if the model generates neither an answer nor a question, it is abstaining from answering, therefore abstain decision is True

    log_info_expert(
        f"[IMPLICIT ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, "
        f"letter_choice: {letter_choice}, atomic_question: {atomic_question}\n"
    )
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "messages": messages,
        "letter_choice": letter_choice,
        "atomic_question": atomic_question,
    }


def _implicit_force_final_choice(patient_state, inquiry, options_dict, **kwargs):
    """One call to get letter when max_questions reached and letter was None."""
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
        abstain_response_dict = implicit_abstention_decision(**kwargs)
        return {
            "type": "question" if abstain_response_dict["abstain"] else "choice",
            "question": abstain_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
        }

    def force_final_choice(self, patient_state):
        kwargs = self.get_abstain_kwargs(patient_state)
        kwargs = {k: v for k, v in kwargs.items() if k not in ("patient_state", "inquiry", "options_dict")}
        return _implicit_force_final_choice(patient_state, self.inquiry, self.options, **kwargs)


def entrypoint(context=None) -> Tuple[List, List[str], List[int], List[str], dict]:
    seed = context.get("seed") if context else None
    return run_mediq(CustomExpert, seed=seed, program_params=PROGRAM_PARAMS)