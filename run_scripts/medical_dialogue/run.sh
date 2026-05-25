#!/bin/bash
# =============================================================================
# GigaEvo run script for medical_dialogue
# Connects to two LLMs (evolution + MedIQ) via API. Redis runs locally.
# Fill in the variables below with your endpoints and API keys.
# =============================================================================

set -e

# Redis (local)
REDIS_PORT=44004

# Evolution LLM (API)
EVOLUTION_LLM_URL="<your-evolution-llm-base-url>/v1"
EVOLUTION_MODEL_NAME="<your-evolution-model-name>"
OPENAI_API_KEY="${OPENAI_API_KEY:-<your-evolution-api-key>}"

# MedIQ LLM — expert (doctor + question generation)
MEDIQ_EXPERT_URL="${MEDIQ_EXPERT_URL:-http://localhost:44005/v1}"
MEDIQ_EXPERT_MODEL_NAME="${MEDIQ_EXPERT_MODEL_NAME:-google/gemma-4-31b-it}"
MEDIQ_EXPERT_API_KEY="${MEDIQ_EXPERT_API_KEY:-not-needed}"

# HuggingFace model name for tokenizer (repo_id, may differ from served name)
MEDIQ_HF_MODEL_NAME="${MEDIQ_HF_MODEL_NAME:-${MEDIQ_EXPERT_MODEL_NAME}}"
MEDIQ_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
MEDIQ_LOG_VLLM_METRICS=0

# MedIQ LLM — patient
MEDIQ_PATIENT_URL="${MEDIQ_PATIENT_URL:-<your-patient-llm-base-url>}"
MEDIQ_PATIENT_MODEL_NAME="${MEDIQ_PATIENT_MODEL_NAME:-openai/gpt-3.5-turbo}"
MEDIQ_PATIENT_API_KEY="${MEDIQ_PATIENT_API_KEY:-<your-patient-api-key>}"
MEDIQ_PATIENT_PROXY_URL="${MEDIQ_PATIENT_PROXY_URL:-}"

# HuggingFace token (for gated models such as Meta-Llama)
HF_TOKEN="${HF_TOKEN:-<your-hf-token>}"

# Batch composition: "composed" (anchor + hard buffer + random) | "random"
MEDIQ_BATCH_MODE="${MEDIQ_BATCH_MODE:-composed}"
MEDIQ_ANCHOR_SIZE="${MEDIQ_ANCHOR_SIZE:-10}"
MEDIQ_HARD_BUFFER_SIZE="${MEDIQ_HARD_BUFFER_SIZE:-15}"
MEDIQ_ANCHOR_REFRESH="${MEDIQ_ANCHOR_REFRESH:-5}"

export OPENAI_API_KEY
export MEDIQ_MODEL_NAME="${MEDIQ_HF_MODEL_NAME}"
export MEDIQ_EXPERT_URL MEDIQ_EXPERT_MODEL_NAME MEDIQ_EXPERT_API_KEY
export MEDIQ_PATIENT_URL MEDIQ_PATIENT_MODEL_NAME MEDIQ_PATIENT_API_KEY MEDIQ_PATIENT_PROXY_URL
export HF_TOKEN
export MEDIQ_CHAT_TEMPLATE_KWARGS MEDIQ_LOG_VLLM_METRICS
export MEDIQ_BATCH_MODE MEDIQ_ANCHOR_SIZE MEDIQ_HARD_BUFFER_SIZE MEDIQ_ANCHOR_REFRESH

# Optional: flush Redis before run
echo "Flushing Redis on port ${REDIS_PORT}..."
redis-cli -p "${REDIS_PORT}" FLUSHDB || echo "Redis not running or already empty"

echo ""
echo "=============================================="
echo "Starting GigaEvo for medical_dialogue"
echo "  Redis:          port ${REDIS_PORT}"
echo "  Evolution LLM:  ${EVOLUTION_MODEL_NAME} @ ${EVOLUTION_LLM_URL}"
echo "  MedIQ expert:   ${MEDIQ_EXPERT_MODEL_NAME} @ ${MEDIQ_EXPERT_URL}"
echo "  MedIQ patient:  ${MEDIQ_PATIENT_MODEL_NAME} @ ${MEDIQ_PATIENT_URL}"
echo "=============================================="
echo ""

# Health-check: evolution LLM
echo "Checking LLM availability..."
CURL_AUTH=()
[[ -n "${OPENAI_API_KEY:-}" ]] && CURL_AUTH=(-H "Authorization: Bearer ${OPENAI_API_KEY}")

if curl -sf --connect-timeout 15 --max-time 20 "${CURL_AUTH[@]}" "${EVOLUTION_LLM_URL}/models" > /dev/null; then
  echo "  OK: Evolution LLM (${EVOLUTION_LLM_URL})"
else
  echo "  FAIL: Evolution LLM unavailable: ${EVOLUTION_LLM_URL}"
  exit 1
fi

MEDIQ_EXPERT_AUTH=()
[[ -n "${MEDIQ_EXPERT_API_KEY:-}" ]] && MEDIQ_EXPERT_AUTH=(-H "Authorization: Bearer ${MEDIQ_EXPERT_API_KEY}")
if curl -sf --connect-timeout 15 --max-time 20 "${MEDIQ_EXPERT_AUTH[@]}" "${MEDIQ_EXPERT_URL}/models" > /dev/null; then
  echo "  OK: MedIQ expert (${MEDIQ_EXPERT_URL})"
else
  echo "  FAIL: MedIQ expert unavailable: ${MEDIQ_EXPERT_URL}"
  exit 1
fi

MEDIQ_PATIENT_AUTH=()
[[ -n "${MEDIQ_PATIENT_API_KEY:-}" ]] && MEDIQ_PATIENT_AUTH=(-H "Authorization: Bearer ${MEDIQ_PATIENT_API_KEY}")
if curl -sf --connect-timeout 15 --max-time 20 "${MEDIQ_PATIENT_AUTH[@]}" "${MEDIQ_PATIENT_URL}/models" > /dev/null; then
  echo "  OK: MedIQ patient (${MEDIQ_PATIENT_URL})"
else
  echo "  FAIL: MedIQ patient unavailable: ${MEDIQ_PATIENT_URL}"
  exit 1
fi

echo "All LLMs available."
echo ""

cd ../gigaevo-core

# reasoning_effort: low | medium | high (for reasoning models; ignored otherwise)
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
export REASONING_EFFORT

# Best (exploitation): FitnessProportionalEliteSelector
python run.py \
    problem.name=gigaevo_mediq_task \
    algorithm=single_island_2d \
    redis.port="${REDIS_PORT}" \
    llm_base_url="${EVOLUTION_LLM_URL}" \
    model_name="${EVOLUTION_MODEL_NAME}" \
    reasoning_effort="${REASONING_EFFORT}" \
    max_tokens=8192 \
    stage_timeout=3000 \
    dag_timeout=3750 \
    max_mutations_per_generation=4 \
    dag_concurrency=4 \
    pipeline=evolution_aware