import json
import os
import sys
import logging
import re
import time
from typing import Dict, List, Optional, Tuple
import random


# BatchComposer lives in shared/ at the repository root
_SHARED_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
from batch_composer import BatchComposer

# Add mediQ src to path — set MEDIQ_SRC_PATH env var to the mediQ/src directory
_mediq_src = os.environ.get("MEDIQ_SRC_PATH")
if _mediq_src:
    sys.path.append(_mediq_src)

# Default data path — set MEDIQ_DATA_PATH env var to the all_train_convo.jsonl file
DEFAULT_DATA_PATH = os.environ.get(
    "MEDIQ_DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "all_train_convo.jsonl"),
)

# Models specification (must match served model names). Set by run.sh.
# Four env variables: two models and two URLs.
#   MEDIQ_EXPERT_MODEL_NAME  – doctor model (also used for question generation)
#   MEDIQ_PATIENT_MODEL_NAME – patient model
#   MEDIQ_EXPERT_URL         – base_url for doctor/QG
#   MEDIQ_PATIENT_URL        – base_url for patient
MEDIQ_EXPERT_MODEL = os.environ.get("MEDIQ_EXPERT_MODEL_NAME")
MEDIQ_EXPERT_MODEL_QG = MEDIQ_EXPERT_MODEL
MEDIQ_PATIENT_MODEL = os.environ.get("MEDIQ_PATIENT_MODEL_NAME", MEDIQ_EXPERT_MODEL)

MEDIQ_EXPERT_URL = os.environ.get("MEDIQ_EXPERT_URL")
MEDIQ_PATIENT_URL = os.environ.get("MEDIQ_PATIENT_URL", MEDIQ_EXPERT_URL)

# Other fixed params
MAX_CASES = 50

PROGRAM_PARAM_RANGES = {
    "self_consistency": (1, 3),
    "max_questions": (1, 8),
}

# ---------------------------------------------------------------------------
# Three-part batch composition config (passed to BatchComposer from shared/)
#   MEDIQ_BATCH_MODE       = "composed" | "random"
#   MEDIQ_ANCHOR_SIZE      = number of stable anchor samples per batch
#   MEDIQ_HARD_BUFFER_SIZE = max hard-buffer slots (filled by weighted sampling)
#   MEDIQ_ANCHOR_REFRESH   = anchor set lifetime in generations
# ---------------------------------------------------------------------------
BATCH_MODE = os.environ.get("MEDIQ_BATCH_MODE", "composed")
BATCH_STATE_FILE = os.environ.get(
    "MEDIQ_BATCH_STATE_FILE",
    os.path.join(os.path.dirname(DEFAULT_DATA_PATH), "batch_state.json"),
)
ANCHOR_SIZE = int(os.environ.get("MEDIQ_ANCHOR_SIZE", "10"))
HARD_BUFFER_MAX_SIZE = int(os.environ.get("MEDIQ_HARD_BUFFER_SIZE", "15"))
ANCHOR_REFRESH_INTERVAL = int(os.environ.get("MEDIQ_ANCHOR_REFRESH", "5"))

def log_info(message, logger_name="detail_logger", print_to_std=False, type="info"):
    logger = logging.getLogger(logger_name)
    if type == "error": return logger.error(message)
    if logger: logger.info(message)
    if print_to_std: print(message + "\n")


def log_info_expert(message, logger="detail_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if logger: logger.info(message)
    if print_to_std: print(message + "\n")


global models
models = {}

_expert_responses: List[str] = []

# ---------------------------------------------------------------------------
# Per-request timing (TTFT, tok/s) instrumentation
# ---------------------------------------------------------------------------
_CURRENT_CASE_ID: Optional[int] = None
_CURRENT_CASE_CALL_IDX: int = 0
_TIMING_EVENTS_BY_CASE: dict[int, list[dict]] = {}


def timing_set_current_case(case_id: Optional[int]) -> None:
    """Set active case id for subsequent model calls."""
    global _CURRENT_CASE_ID, _CURRENT_CASE_CALL_IDX
    _CURRENT_CASE_ID = case_id
    _CURRENT_CASE_CALL_IDX = 0


def timing_get_and_clear_case_events(case_id: int) -> list[dict]:
    """Return and clear accumulated timing events for a case."""
    return _TIMING_EVENTS_BY_CASE.pop(case_id, [])


def _timing_enabled() -> bool:
    return os.environ.get("MEDIQ_TIMING", "0") in ("1", "true", "True", "yes", "YES")


class ModelCacheHTTP:
    """ModelCache that uses vLLM server via OpenAI-compatible HTTP API."""
    
    def __init__(self, model_name, base_url, **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.args = kwargs
        self._init_client()
    
    def _init_client(self):
        import httpx
        from openai import OpenAI
        proxy_url = self.args.get("proxy_url")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.args.get("api_key"),
            http_client=httpx.Client(proxy=proxy_url) if proxy_url else None,
        )
        log_info(f"[ModelCacheHTTP] Initialized client for {self.model_name} at {self.base_url}"
                 + (f" via proxy {proxy_url}" if proxy_url else ""))
    
    def generate(self, messages):
        log_info(f"[{self.model_name}][INPUT]: {messages}")

        temperature = self.args.get("temperature")
        max_tokens = self.args.get("max_tokens")
        top_p = self.args.get("top_p")
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presence_penalty", 0)
        extra_body = self.args.get("extra_body")

        try:
            timing = _timing_enabled()
            if timing:
                # Streaming path: implement create_with_ttft_tps() logic (simplified).
                start_time = time.perf_counter()
                first_token_time: Optional[float] = None
                end_time: Optional[float] = None

                text_parts: list[str] = []
                usage = None
                finish_reason = None
                system_fingerprint = None
                response_id = None
                model_name = None

                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=extra_body,
                )
                for chunk in stream:
                    end_time = time.perf_counter()

                    if response_id is None:
                        response_id = getattr(chunk, "id", None)
                    if model_name is None:
                        model_name = getattr(chunk, "model", None)
                    if system_fingerprint is None:
                        system_fingerprint = getattr(chunk, "system_fingerprint", None)

                    # Usage may arrive even when choices are empty (provider-dependent).
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        usage = chunk_usage

                    if not getattr(chunk, "choices", None):
                        continue

                    choice = chunk.choices[0]
                    if getattr(choice, "finish_reason", None) is not None:
                        finish_reason = choice.finish_reason

                    delta = getattr(choice, "delta", None)
                    piece = getattr(delta, "content", None) if delta is not None else None
                    if not piece:
                        continue

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    text_parts.append(piece)

                if end_time is None:
                    end_time = time.perf_counter()

                response_text = "".join(text_parts)

                if first_token_time is None:
                    ttft_s = None
                    tok_per_s = None
                    completion_tokens = 0
                    usage_json = None
                    generation_s = None
                    total_s = end_time - start_time
                else:
                    ttft_s = first_token_time - start_time
                    generation_time = end_time - first_token_time
                    completion_tokens = getattr(usage, "completion_tokens", None)
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    total_tokens = getattr(usage, "total_tokens", None)
                    usage_json = {
                        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
                        "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, (int, float)) else None,
                        "total_tokens": int(total_tokens) if isinstance(total_tokens, (int, float)) else None,
                    }
                    generation_s = generation_time
                    total_s = end_time - start_time
                    tok_per_s = (
                        (completion_tokens / generation_time)
                        if isinstance(completion_tokens, (int, float))
                        else None
                    )

                # Save per-call timing event, attached to current case id if set.
                global _CURRENT_CASE_CALL_IDX
                event = {
                    "ts": time.time(),
                    "case_id": _CURRENT_CASE_ID,
                    "call_idx": _CURRENT_CASE_CALL_IDX,
                    "model": self.model_name,
                    "is_patient": self.model_name == MEDIQ_PATIENT_MODEL,
                    "ttft_s": ttft_s,
                    "generation_s": generation_s,
                    "total_s": total_s,
                    "prompt_tokens": None if usage_json is None else usage_json.get("prompt_tokens"),
                    "completion_tokens": None if usage_json is None else usage_json.get("completion_tokens"),
                    "total_tokens": None if usage_json is None else usage_json.get("total_tokens"),
                    "tok_per_s": tok_per_s,
                }
                _CURRENT_CASE_CALL_IDX += 1
                if _CURRENT_CASE_ID is not None:
                    _TIMING_EVENTS_BY_CASE.setdefault(int(_CURRENT_CASE_ID), []).append(event)
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    extra_body=extra_body,
                )
                if not response.choices:
                    raise RuntimeError(f"[PROVIDER ERROR, not a mutant bug] Empty choices in response: {response}")
                response_text = response.choices[0].message.content
        except Exception as e:
            log_info(f"[{self.model_name}][ERROR]: {e}", type="error")
            raise

        log_info(
            f"[{self.model_name}][OUTPUT]: "
            f"{{'response_text': '{response_text[:100]}...'}}"
        )
        return response_text

    def generate_n(self, messages, n):
        """Shared-prefill n-way generation; returns a list[str] of length n.

        Sends a single OpenAI-compatible request with ``n=n`` so vLLM computes
        the prompt prefill once and runs ``n`` decode branches in a shared
        batch. This is strictly faster than sending ``n`` sequential requests
        even with prefix-caching, because there is exactly one prefill,
        one round-trip, and decode is co-batched at the scheduler level.

        Non-streaming is used regardless of ``MEDIQ_TIMING`` — per-sequence
        TTFT is not meaningful when multiple choices share a single prefill.
        A single aggregated timing event is recorded.
        """
        if n <= 1:
            return [self.generate(messages)]

        log_info(f"[{self.model_name}][INPUT n={n}]: {messages}")

        temperature = self.args.get("temperature")
        max_tokens = self.args.get("max_tokens")
        top_p = self.args.get("top_p")
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presence_penalty", 0)
        extra_body = self.args.get("extra_body")

        timing = _timing_enabled()
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=n,
                extra_body=extra_body,
            )
        except Exception as e:
            log_info(f"[{self.model_name}][ERROR n={n}]: {e}", type="error")
            raise

        if not response.choices:
            raise RuntimeError(
                f"[PROVIDER ERROR, not a mutant bug] Empty choices in response: {response}"
            )

        # vLLM returns choices in index order, but sort defensively.
        texts_by_idx: dict[int, str] = {}
        for c in response.choices:
            idx = getattr(c, "index", len(texts_by_idx))
            texts_by_idx[idx] = c.message.content or ""
        texts = [texts_by_idx.get(i, "") for i in range(n)]

        if timing:
            end_time = time.perf_counter()
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = (
                getattr(usage, "completion_tokens", None) if usage else None
            )
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            total_s = end_time - start_time
            tok_per_s = (
                (completion_tokens / total_s)
                if isinstance(completion_tokens, (int, float)) and total_s > 0
                else None
            )

            global _CURRENT_CASE_CALL_IDX
            event = {
                "ts": time.time(),
                "case_id": _CURRENT_CASE_ID,
                "call_idx": _CURRENT_CASE_CALL_IDX,
                "model": self.model_name,
                "is_patient": self.model_name == MEDIQ_PATIENT_MODEL,
                "ttft_s": None,
                "generation_s": None,
                "total_s": total_s,
                "prompt_tokens": int(prompt_tokens)
                if isinstance(prompt_tokens, (int, float))
                else None,
                "completion_tokens": int(completion_tokens)
                if isinstance(completion_tokens, (int, float))
                else None,
                "total_tokens": int(total_tokens)
                if isinstance(total_tokens, (int, float))
                else None,
                "tok_per_s": tok_per_s,
                "n": n,
            }
            _CURRENT_CASE_CALL_IDX += 1
            if _CURRENT_CASE_ID is not None:
                _TIMING_EVENTS_BY_CASE.setdefault(int(_CURRENT_CASE_ID), []).append(
                    event
                )

        log_info(
            f"[{self.model_name}][OUTPUT n={n}]: "
            + str([(t[:80] + "...") if len(t) > 80 else t for t in texts])
        )
        return texts


def _ensure_model_cache(model_name, **kwargs):
    """Initialise (or reuse) a ModelCacheHTTP for the given model name."""
    is_patient = model_name == MEDIQ_PATIENT_MODEL
    if is_patient:
        base_url = MEDIQ_PATIENT_URL
        api_key = os.environ.get("MEDIQ_PATIENT_API_KEY")
        proxy_url = os.environ.get("MEDIQ_PATIENT_PROXY_URL")
    else:
        base_url = MEDIQ_EXPERT_URL
        api_key = os.environ.get("MEDIQ_EXPERT_API_KEY")
        proxy_url = None

    model_cache = models.get(model_name, None)
    if model_cache is None:
        log_info(
            f"[model_cache] Initializing HTTP client for {model_name} at {base_url}"
        )
        chat_template_kwargs_raw = os.environ.get("MEDIQ_CHAT_TEMPLATE_KWARGS")
        extra_body = None
        if chat_template_kwargs_raw:
            try:
                extra_body = {
                    "chat_template_kwargs": json.loads(chat_template_kwargs_raw)
                }
            except Exception as e:
                log_info(
                    f"[model_cache] Invalid MEDIQ_CHAT_TEMPLATE_KWARGS: {e}",
                    type="error",
                )

        model_cache = ModelCacheHTTP(
            model_name,
            base_url=base_url,
            api_key=api_key,
            proxy_url=proxy_url,
            extra_body=extra_body,
            **kwargs,
        )
        models[model_name] = model_cache
    return model_cache, is_patient


def get_response(messages, model_name, **kwargs):
    model_cache, is_patient = _ensure_model_cache(model_name, **kwargs)
    result = model_cache.generate(messages)
    if not is_patient and isinstance(result, str):
        _expert_responses.append(result)
    return result


def get_responses(messages, model_name, n, **kwargs):
    """Get ``n`` completions via a single shared-prefill API call.

    Prefer this over calling ``get_response`` in a loop when doing
    self-consistency: vLLM runs a single prefill for the prompt and
    ``n`` decode branches in a shared batch, which is consistently
    faster than sequential calls even with prefix-caching enabled.
    """
    model_cache, is_patient = _ensure_model_cache(model_name, **kwargs)
    results = model_cache.generate_n(messages, n)
    if not is_patient:
        for r in results:
            if isinstance(r, str):
                _expert_responses.append(r)
    return results


def shutdown_models():
    global models
    for mc in models.values():
        mc.model = None
    models.clear()

    
def load_data(filename):
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    data = [json.loads(line) for line in json_list]

    # Normalize any precomputed facts / atomic_facts:
    # - strip leading numbering like "1.", "2)", "3 -", etc.
    # - strip leading dash markers
    for sample in data:
        for key in ("facts", "atomic_facts"):
            facts = sample.get(key)
            if isinstance(facts, list):
                cleaned_facts: list[str] = []
                for s in facts:
                    if not isinstance(s, str):
                        continue
                    s = s.strip()
                    if not s:
                        continue
                    # Remove leading dash markers
                    s = s.lstrip("-").strip()
                    # Remove leading numbering like "1.", "2)", "3 -", etc.
                    s = re.sub(r"^[0-9]+\s*[\.\-\):]*\s*", "", s)
                    if s:
                        cleaned_facts.append(s)
                sample[key] = cleaned_facts
    return data


class Patient:
    def __init__(self, args, sample):
        self.args = args
        # Assuming 'context' is a list or a long string of historical or background information
        if isinstance(sample['context'], list) and len(sample['context']) > 0:
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = sample['context'][0]  # Taking the first item if it's a list
            self.context_list = sample['context']
            self.context_para = " ".join(sample['context'])
        elif isinstance(sample['context'], str):
            # Assuming sentences are separated by periods, taking the first sentence
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = sample['context'].split(". ")[0]
            temp = sample['context'].split(". ")
            self.context_list = [temp[i]+'.' if i!=len(temp)-1 and not temp[i].endswith('.') else temp[i] for i in range(len(temp))]
            self.context_para = sample['context']
        else:
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = ""  # Default fallback
            self.context_list = []
            self.context_para = 'None'
        
        self.model_name = MEDIQ_PATIENT_MODEL
        self.history = []  # To track the interaction history of questions and answers
        # To store atomic facts after initial processing, you can choose to
        # store this somewhere locally to avoid repeated processing
        self.facts = sample.get('facts') or sample.get('atomic_facts')

        # Use the same token budget as the expert unless explicitly overridden
        # in respond() (e.g., for long fact extraction).
        self.max_length = args.max_tokens

    def update_state(self, question, answer):
        # Update the internal history with the new question and the corresponding answer
        self.history.append({"question": question, "answer": answer})

    def get_state(self):
        # Return the initial context and the history of interactions
        return {
            "initial_info": self.initial_info,
            "interaction_history": self.history
        }
    
    def get_questions(self):
        # Return the list of questions asked so far
        return [qa["question"] for qa in self.history]
    
    def get_answers(self):
        # Return the list of answers provided so far
        return [qa["answer"] for qa in self.history]
    
    def get_response(self, messages, max_length=None):
        """
        Call the MedIQ LLM for the patient.

        max_length controls the completion token budget for this call.
        If not provided, falls back to self.max_length, which is tied to
        args.max_tokens.
        """
        if max_length is None:
            max_length = self.max_length

        # Important: ModelCacheHTTP expects `max_tokens`, not `max_length`.
        # We also forward temperature / top_p / top_logprobs so that patient
        # and expert use consistent generation settings.
        return get_response(
            messages,
            self.model_name,
            max_tokens=max_length,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_logprobs=self.args.top_logprobs,
        )
    
    def respond(self, question):
        raise NotImplementedError


class FactSelectPatient(Patient):
    def respond(self, question):
        if not self.facts:
            system_prompt = "You are a truthful medical assistant that understands the patient's information."
            user_prompt = f"Break the following patient information into a list of independent atomic facts, with one piece of information in each statement. Each fact should only include the smallest unit of information, but should be self-contained.\n\"{self.context_para}\"\nResponse with the list of atomic facts and nothing else. Write each fact on a new line starting with a dash (-). No numbered lists allowed."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response_text = self.get_response(messages, max_length=self.max_length)
            response_text = [s.strip().lstrip('- ').strip() for s in response_text.splitlines() if s.strip()]
            self.facts = response_text
        
        facts_prompt = "\n".join(self.facts)
        system_prompt = "You are a truthful medical assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient given a list of factual statements describing the patient. Please return the facts that answer the doctor's question verbatim without any additional information. Do not prefix the answer with numbers. If none of the facts answer the question, simply say \"The patient cannot answer this question, please do not ask this question again.\""
        prompt = f"List of facts:\n{facts_prompt}\n\nDoctor's question: \"{question}\"\n\nStatements that answer the question:"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = self.get_response(messages)
        
        self.update_state(question, response)
        return response


def run_patient_interaction(expert_class, patient_class, args, sample):
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = patient_class(args, sample)  # Assuming the patient_system is initialized with the sample which includes necessary context
    temp_additional_info = []  # To store optional data like confidence scores

    while len(patient_system.get_questions()) < args.max_questions:
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state)
        
        # Optional return values for analysis, e.g., confidence score, logprobs
        temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "letter_choice", "question"]})

        if response_dict["type"] == "question":
            # Patient generates an answer based on the last question asked, and add to memory
            patient_response = patient_system.respond(response_dict["question"])
            log_info(f"[Patient System]: {patient_response}")

        elif response_dict["type"] == "choice":
            expert_decision = response_dict["letter_choice"]
            sample_info = {
                "initial_info": patient_system.initial_info,
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                "context": sample["context"],
                "facts": patient_system.facts, # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
                "forced_final": False,
            }
            return expert_decision, patient_system.get_questions(), patient_system.get_answers(), temp_additional_info, sample_info
        
        else:
            raise ValueError("Invalid response type from expert_system.")
    
    # If max questions are reached and no final decision has been made
    log_info(f"==================== Max Interaction Length ({args.max_questions} turns) Reached --> Force Final Answer ====================")
    patient_state = patient_system.get_state()
    log_info("[Force final choice] requesting final letter from model (one call).")
    stuck_response = expert_system.force_final_choice(patient_state)

    sample_info = {
        "initial_info": patient_system.initial_info,
        "correct_answer": sample["answer"],
        "correct_answer_idx": sample["answer_idx"],
        "question": sample["question"],
        "options": sample["options"],
        "context": sample["context"],
        "facts": patient_system.facts,  # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
        "forced_final": True,
    }

    return (
        stuck_response,
        patient_system.get_questions(),
        patient_system.get_answers(),
        temp_additional_info,
        sample_info,
    )

class Expert:
    """
    Expert system skeleton
    """
    def __init__(self, args, inquiry, options):
        # Initialize the expert with necessary parameters and the initial context or inquiry
        self.args = args
        self.inquiry = inquiry
        self.options = options

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        raise NotImplementedError

    def force_final_choice(self, patient_state):
        """When max_questions reached and letter_choice is None, get a final letter in one extra call."""
        raise NotImplementedError

    def ask_question(self, patient_state, prev_messages, task_prompt, question_generation_function):
        # Generate a question based on the current patient state
        kwargs = {
            "patient_state": patient_state,
            "task_prompt": task_prompt,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "messages": prev_messages,
            "model_name": MEDIQ_EXPERT_MODEL_QG,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
        }
        return question_generation_function(**kwargs)
    
    def get_abstain_kwargs(self, patient_state):
        kwargs = {
            "max_depth": self.args.max_questions,
            "patient_state": patient_state,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "abstain_threshold": self.args.abstain_threshold,
            "self_consistency": self.args.self_consistency,
            "model_name": MEDIQ_EXPERT_MODEL,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
        }
        return kwargs


def parse_choice(response_text, options_dict):
    if response_text.strip() in ["A", "B", "C", "D"]:
        return response_text.strip()
    for response_line in response_text.split("\n"):
        for op_letter, op_text in options_dict.items():
            if op_text.lower() in response_line.lower():
                log_info(f"....Found {op_text} in response line: {response_line}")
                return op_letter
        for op_letter in options_dict.keys():
            if op_letter in [token for token in re.sub(r"[,.;@#()?!'/&:$]+\ *", " ", response_line).split(' ')]:
                log_info(f"....Found {op_letter} in response line: {response_line}")
                return op_letter
    log_info("can't parse choice: {}".format(response_text), type="error")
    return None


def expert_response_choice(messages, options_dict, **kwargs):
    """
    Get intermediate answer choice regardless of abstention decision
    """
    log_info(f"++++++++++++++++++++ Start of Multiple Chocie Decision [py:expert_response_choice()] ++++++++++++++++++++")
    log_info(f"[<CHOICE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
    response_text = get_response(messages, **kwargs)
    if not response_text: 
        log_info("[<CHOICE LM RES>]: " + "No response.")
        return "No response.", None
    log_info("[<CHOICE LM RES>]: " + response_text)

    letter_choice = parse_choice(response_text, options_dict)
    if letter_choice:
        log_info("[<CHOICE PARSED>]: " + letter_choice)
    else:
        log_info("[<CHOICE PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, letter_choice


def parse_confidence_score(response_text):
    # parse the probability
    float_regex = re.compile(r'\d+\.\d+')
    scores = re.findall(float_regex, response_text)

    if len(scores) == 0:
        log_info("can't parse confidence score - answer: {}".format(response_text), type="error")
        score = round(0.2 + (random.random() - random.random()) * 0.2, 4)
        return score
    
    prob = float(scores[-1])
    if len(scores) > 1: logging.warning("more than one confidence score - using last: {}".format(response_text))
    if prob > 1: logging.warning("confidence score > 1: {}".format(response_text))
    return prob 


def parse_yes_no(response_text):
    temp_processed_response = response_text.lower().replace('.','').replace(',','').replace(';','').replace(':','').split("DECISION:")[-1].strip()
    yes_answer = "yes" in temp_processed_response
    no_answer = "no" in temp_processed_response
    if yes_answer == no_answer:
        yes_choice = "NO"
        log_info("can't parse yes/no abstain answer: {}".format(response_text), type="error")
    if yes_answer: yes_choice = "YES"
    elif no_answer: yes_choice = "NO"
    return yes_choice


def parse_atomic_question(response_text):
    questions = []
    for line in response_text.split("\n"):
        if '?' in line:
            questions.append(line.split(":")[-1].strip())
        
    if len(questions) == 0:
        log_info("can't find question in answer: {}".format(response_text), type="error")
        return None
            
    atomic_question = questions[-1].replace("'", "").replace('"', "").strip()
    return atomic_question


def expert_response_question(messages, **kwargs):
    """
    Get follow-up question
    """
    log_info(f"++++++++++++++++++++ Start of Question Generator [py:expert_response_question()] ++++++++++++++++++++")
    log_info(f"[<QUESTION GENERATOR PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
    response_text = get_response(messages, **kwargs)
    if not response_text: 
        log_info("[<QUESTION GENERATOR LM RES>]: " + "No response.")
        return "No response.", None
    log_info("[<QUESTION GENERATOR LM RES>]: " + response_text)

    atomic_question = parse_atomic_question(response_text)
    if atomic_question:
        log_info("[<QUESTION GENERATOR PARSED>]: " + atomic_question)
    else:
        log_info("[<QUESTION GENERATOR PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, atomic_question


def question_generation(messages, task_prompt, **kwargs):
    messages.append({"role": "user", "content": task_prompt})

    response_text, atomic_question = expert_response_question(messages, **kwargs)
    log_info_expert(f"[ATOMIC QUESTION PROMPT]: {messages}")
    log_info_expert(f"[ATOMIC QUESTION RESPONSE]: {atomic_question}\n")
    messages.append({"role": "assistant", "content": atomic_question})

    log_info_expert(f"[ATOMIC QUESTION RETURN]: {atomic_question}\n")
    return {
        "atomic_question": atomic_question,
        "messages": messages,
    }


class Args:
    def __init__(self, program_params):
        # Benchmark settings
        self.max_questions = int(program_params["max_questions"])
        self.self_consistency = int(program_params["self_consistency"])
        
        # Abstention settings
        # If abstain_threshold is None, each expert uses its own internal default
        # (e.g. PROB_THRESHOLD in numerical_cutoff, SCALE_THRESHOLD in scale_abstain).
        # self.self_consistency = 1
        self.abstain_threshold = None
        
        # Model inference settings
        self.temperature = 0.6
        self.top_p = 0.9
        self.max_tokens = 256
        self.top_logprobs = 0


def run_mediq(expert_class, seed=None, program_params=None) -> Tuple[List, List[str], List[int], List[str], Dict[str, Dict[str, int]]]:
    if not isinstance(program_params, dict):
        raise ValueError("PROGRAM_PARAMS must be a dict")
    for name, (min_v, max_v) in PROGRAM_PARAM_RANGES.items():
        val = int(program_params.get(name, -1))
        if val < min_v or val > max_v:
            raise ValueError(f"PROGRAM_PARAMS['{name}']={val} out of range [{min_v}, {max_v}]")

    data_path = DEFAULT_DATA_PATH
    max_cases = MAX_CASES
    print(f"max_cases: {max_cases}")
    
    args = Args(program_params=program_params)
    patient_data = load_data(data_path)

    batch_indices: Optional[List[int]] = None
    composer: Optional[BatchComposer] = None

    if max_cases and max_cases < len(patient_data):
        if seed is not None and BATCH_MODE == "composed":
            composer = BatchComposer(
                len(patient_data),
                batch_size=max_cases,
                state_path=BATCH_STATE_FILE,
                anchor_size=ANCHOR_SIZE,
                hard_buffer_max_size=HARD_BUFFER_MAX_SIZE,
                anchor_refresh_interval=ANCHOR_REFRESH_INTERVAL,
            )
            batch_indices = composer.compose_batch(seed)
            patient_data = [patient_data[i] for i in batch_indices]
        elif seed is not None:
            rng = random.Random(seed)
            patient_data = rng.sample(patient_data, k=max_cases)
        else:
            patient_data = random.sample(patient_data, k=max_cases)
    
    dialogues = []
    diagnoses = []
    case_ids = []
    ground_truth = []
    all_expert_responses: List[List[str]] = []
    forced_final_flags: List[bool] = []
    for i, sample in enumerate(patient_data):
        _expert_responses.clear()
        letter_choice, questions, answers, temp_additional_info, sample_info = run_patient_interaction(
            expert_class=expert_class,
            patient_class=FactSelectPatient,
            args=args,
            sample=sample
        )
        
        dialogue = []
        for q, a in zip(questions, answers):
            dialogue.append(("doctor", q))
            dialogue.append(("patient", a))
        dialogue.append(("doctor", f"Final answer: {letter_choice}"))
        
        dialogues.append(dialogue)
        diagnoses.append(letter_choice)
        case_ids.append(i)
        ground_truth.append(sample["answer_idx"])
        all_expert_responses.append(list(_expert_responses))
        forced_final_flags.append(sample_info["forced_final"])

    if composer is not None and batch_indices is not None and seed is not None:
        correct_mask = [d == gt for d, gt in zip(diagnoses, ground_truth)]
        composer.update_results(batch_indices, correct_mask, seed)

    run_metadata = {
        "program_params": {"max_questions": args.max_questions, "self_consistency": args.self_consistency},
        "expert_responses": all_expert_responses,
        "forced_final": forced_final_flags,
    }
    return dialogues, diagnoses, case_ids, ground_truth, run_metadata