#!/usr/bin/env bash
# One-command reproduction of the H100 results on a freshly rented GPU box.
#
#   git clone https://github.com/shujaraina/emojimodel.git && cd EmojiModel
#   bash scripts/run_repro_h100.sh
#
# Stages run in order and can be selected individually:
#
#   STAGES=setup                bash scripts/run_repro_h100.sh
#   STAGES=train,save           bash scripts/run_repro_h100.sh
#   STAGES=eval                 bash scripts/run_repro_h100.sh
#
# Design notes, so you are not debugging this while the meter runs:
#
#   * flash-attn / mamba-ssm / causal-conv1d are NOT installed. models/dit.py
#     falls back to F.scaled_dot_product_attention and this project uses the
#     `dit` backbone, so mamba is dead weight. Those three packages compile
#     from source and can cost 30-60 minutes for no benefit here.
#   * numpy is pinned <2. transformers 4.38.2 / lightning 2.2.1 predate the
#     numpy 2 ABI break and fail confusingly against it.
#   * Checkpoints are archived immediately after training and BEFORE the eval
#     stage. The original H100 instance was released with the only copy of the
#     weights on it; do not repeat that.
#
# Wall-clock on a single H100, approximate:
#   setup ~15 min | semantic ~5 min | train ~30 min | eval ~25 min
set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RUN_NAME=${RUN_NAME:-repro_hard1}
RUN_ROOT=${RUN_ROOT:-outputs/${RUN_NAME}}
DATA_CACHE=${DATA_CACHE:-/tmp/emoji_repro_${RUN_NAME}}
VENV=${VENV:-.venv-repro}

# Training shape. These reproduce outputs/emoji_two_phase_h100_hard1, read back
# from its committed .hydra/config.yaml.
MODEL=${MODEL:-medium}          # hidden 1024 x 24 blocks, ~328M params
MODEL_LENGTH=${MODEL_LENGTH:-64}
PHASE1_STEPS=${PHASE1_STEPS:-1000}
PHASE2_STEPS=${PHASE2_STEPS:-3000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-512}

# Semantic-table shape. hard1's semantic_pairs.jsonl has exactly 100,000 rows
# and pairs = limit x top_k, so limit=100000 / top_k=1 is the combination that
# reproduces it. NOTE: the build arguments were never recorded in the hydra
# config (it is a separate script), so this is inferred from the artifact's
# line count, not read from the run. full3's 15,000 = 5,000 x 3 corroborates
# the relationship.
SEMANTIC_LIMIT=${SEMANTIC_LIMIT:-100000}
SEMANTIC_TOP_K=${SEMANTIC_TOP_K:-1}
TEACHER_MODEL=${TEACHER_MODEL:-BAAI/bge-small-en-v1.5}

# Evaluation shape.
EVAL_SAMPLES=${EVAL_SAMPLES:-7}
EVAL_STEPS=${EVAL_STEPS:-32}
EVAL_BATCH=${EVAL_BATCH:-16}
NUM_GROUPS=${NUM_GROUPS:-25}

ARCHIVE_DIR=${ARCHIVE_DIR:-/tmp/emoji_artifacts}
STAGES=${STAGES:-setup,semantic,train,save,eval,save}

export DISABLE_FLASH_ATTN=${DISABLE_FLASH_ATTN:-1}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/tmp/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/wandb/cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/wandb/config}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export HF_HOME=${HF_HOME:-/tmp/hf}

LOG_DIR="${RUN_ROOT}/logs"
PHASE1_DIR="${RUN_ROOT}/phase1_semantic"
PHASE2_DIR="${RUN_ROOT}/phase2_reply"
EVAL_DIR="${RUN_ROOT}/eval"
SEMANTIC_TABLE="${RUN_ROOT}/semantic_table.pt"
SEMANTIC_PAIRS="${RUN_ROOT}/semantic_pairs.jsonl"
VOCAB_CACHE="${DATA_CACHE}/atomic_emoji_vocab.json"
REPLY_DATA="data/emoji_reply/emoji_reply.jsonl"
HELDOUT="data/emoji_reply/heldout_validation.jsonl"
BENCHMARK="data/emoji_reply/benchmark.jsonl"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
c_blue=$'\033[1;34m'; c_grn=$'\033[1;32m'; c_yel=$'\033[1;33m'
c_red=$'\033[1;31m'; c_off=$'\033[0m'

say()  { echo "${c_blue}==>${c_off} $*"; }
ok()   { echo "${c_grn}  ok${c_off} $*"; }
warn() { echo "${c_yel}  !!${c_off} $*" >&2; }
die()  { echo "${c_red}ERROR${c_off} $*" >&2; exit 1; }

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

STAGE_START=0
begin() { STAGE_START=$SECONDS; say "$1"; }
finish() {
  local mins=$(( (SECONDS - STAGE_START) / 60 ))
  local secs=$(( (SECONDS - STAGE_START) % 60 ))
  ok "$1 finished in ${mins}m${secs}s"
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
[[ -f main.py && -d configs ]] || die "run this from the repository root"
mkdir -p "${RUN_ROOT}" "${DATA_CACHE}" "${LOG_DIR}" "${EVAL_DIR}" \
         "${ARCHIVE_DIR}" "${WANDB_DIR}" "${MPLCONFIGDIR}"

say "Reproduction run: ${RUN_NAME}"
echo "    stages       ${STAGES}"
echo "    model        ${MODEL} (length ${MODEL_LENGTH})"
echo "    steps        phase1 ${PHASE1_STEPS} / phase2 ${PHASE2_STEPS}"
echo "    batch        ${GLOBAL_BATCH_SIZE}"
echo "    run root     ${RUN_ROOT}"
echo "    archive      ${ARCHIVE_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version \
             --format=csv,noheader | sed 's/^/    gpu          /'
else
  warn "nvidia-smi not found — training will be extremely slow or fail"
fi

avail_gb=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
[[ "${avail_gb}" -ge 20 ]] || warn "only ${avail_gb}GB free; checkpoints need ~10GB"

# --------------------------------------------------------------------------
# Stage: setup
# --------------------------------------------------------------------------
if has_stage setup; then
  begin "setup — python environment (no flash-attn / mamba / causal-conv1d)"

  if [[ ! -d "${VENV}" ]]; then
    python3 -m venv "${VENV}"
    ok "created ${VENV}"
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python -m pip install -q --upgrade pip wheel

  if ! python -c "import torch" 2>/dev/null; then
    say "installing torch (cu121 wheel)"
    python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121
  fi

  # Pinned to match requirements.yaml, minus the three source-built CUDA
  # packages. numpy<2 is load-bearing for this transformers/lightning pair.
  python -m pip install -q \
    "numpy<2" \
    datasets==2.18.0 einops==0.7.0 "fsspec==2024.2.0" h5py==3.10.0 \
    hydra-core==1.3.2 lightning==2.2.1 omegaconf==2.3.0 pandas==2.2.1 \
    rich==13.7.1 scikit-learn==1.4.0 timm==0.9.16 transformers==4.38.2 \
    regex wandb fastapi pydantic uvicorn pyyaml

  python - <<'PY'
import torch
print(f"    torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device: {torch.cuda.get_device_name(0)}")
PY
  finish "setup"
else
  [[ -d "${VENV}" ]] && source "${VENV}/bin/activate" || true
fi

[[ -d "${VENV}" ]] && source "${VENV}/bin/activate" 2>/dev/null || true

# --------------------------------------------------------------------------
# Stage: semantic — build the Phase-1 teacher targets
# --------------------------------------------------------------------------
if has_stage semantic; then
  if [[ -f "${SEMANTIC_PAIRS}" ]]; then
    ok "semantic pairs already present ($(wc -l < "${SEMANTIC_PAIRS}") rows) — skipping"
  else
    begin "semantic — embedding Text2Emoji with ${TEACHER_MODEL}"
    python scripts/build_emoji_semantic_table.py \
      --cache-dir "${DATA_CACHE}" \
      --output "${SEMANTIC_TABLE}" \
      --pairs-output "${SEMANTIC_PAIRS}" \
      --teacher-model "${TEACHER_MODEL}" \
      --batch-size 512 \
      --top-k "${SEMANTIC_TOP_K}" \
      --limit "${SEMANTIC_LIMIT}" \
      --device cuda 2>&1 | tee "${LOG_DIR}/semantic.log"
    ok "wrote $(wc -l < "${SEMANTIC_PAIRS}") pairs"
    finish "semantic"
  fi
fi

# --------------------------------------------------------------------------
# Stage: train — two-phase
# --------------------------------------------------------------------------
COMMON_ARGS=(
  mode=train
  data=emoji_reply
  data.cache_dir="${DATA_CACHE}"
  data.emoji_vocab_cache="${VOCAB_CACHE}"
  'data.emoji_vocab_sources=[text2emoji,common]'
  "data.emoji_vocab_extra_files=[$(pwd)/${REPLY_DATA}]"
  model="${MODEL}"
  model.length="${MODEL_LENGTH}"
  parameterization=subs
  backbone=dit
  trainer.accelerator=cuda
  trainer.devices=1
  trainer.precision=bf16-mixed
  +trainer.check_val_every_n_epoch=null
  trainer.limit_val_batches=2
  trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=5
  loader.global_batch_size="${GLOBAL_BATCH_SIZE}"
  loader.batch_size="${BATCH_SIZE}"
  loader.eval_batch_size="${BATCH_SIZE}"
  loader.num_workers=8
  loader.pin_memory=true
  training.ema=0.9999
  eval.compute_generative_perplexity=false
  wandb.project=emoji-model
)

if has_stage train; then
  [[ -f "${SEMANTIC_PAIRS}" ]] || die "missing ${SEMANTIC_PAIRS} — run STAGES=semantic first"

  begin "train phase 1 — semantic grounding (${PHASE1_STEPS} steps)"
  python -u -m main \
    "${COMMON_ARGS[@]}" \
    data.data_file="$(pwd)/${SEMANTIC_PAIRS}" \
    data.emoji_benchmark_file=null \
    data.emoji_include_challenge_in_train=false \
    data.permutation_augment_prob=0.0 \
    trainer.max_steps="${PHASE1_STEPS}" \
    trainer.val_check_interval=50 \
    wandb.name="${RUN_NAME}_phase1" \
    wandb.id="${RUN_NAME}_phase1" \
    hydra.run.dir="${PHASE1_DIR}" \
    checkpointing.resume_from_ckpt=false \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=50 \
    2>&1 | tee "${LOG_DIR}/phase1.log"
  finish "phase 1"

  [[ -f "${PHASE1_DIR}/checkpoints/last.ckpt" ]] \
    || die "phase 1 produced no checkpoint at ${PHASE1_DIR}/checkpoints/last.ckpt"

  begin "train phase 2 — emoji reply (${PHASE2_STEPS} steps)"
  python -u -m main \
    "${COMMON_ARGS[@]}" \
    data.data_file="$(pwd)/${REPLY_DATA}" \
    data.emoji_benchmark_file="$(pwd)/${BENCHMARK}" \
    data.emoji_include_challenge_in_train=false \
    data.emoji_challenge_repeat=1 \
    data.emoji_split_strategy=random \
    data.emoji_validation_size=0.05 \
    data.emoji_split_seed=42 \
    data.emoji_max_response_tokens=null \
    data.emoji_supervised_pad_tokens=0 \
    data.permutation_augment_prob=0.6 \
    data.permutation_augment_prompt=true \
    data.permutation_augment_response=true \
    data.emoji_infill_train_prob=0.0 \
    trainer.max_steps="${PHASE2_STEPS}" \
    trainer.val_check_interval=50 \
    wandb.name="${RUN_NAME}_phase2" \
    wandb.id="${RUN_NAME}_phase2" \
    hydra.run.dir="${PHASE2_DIR}" \
    checkpointing.resume_from_ckpt=false \
    +checkpointing.init_from_ckpt_path="${PHASE1_DIR}/checkpoints/last.ckpt" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=50 \
    2>&1 | tee "${LOG_DIR}/phase2.log"
  finish "phase 2"
fi

CKPT="${PHASE2_DIR}/checkpoints/best.ckpt"
[[ -f "${CKPT}" ]] || CKPT="${PHASE2_DIR}/checkpoints/last.ckpt"

# --------------------------------------------------------------------------
# Stage: save — get the weights off this box
# --------------------------------------------------------------------------
# do_save <label> — archive weights + configs + whatever eval artifacts exist.
# Called twice: right after training (so the weights are safe before the long
# eval stage) and again after eval (to capture the results). Both are cheap
# relative to losing the run.
LAST_ARCHIVE=""
do_save() {
  local label="$1"
  if [[ ! -f "${CKPT}" ]]; then
    warn "no checkpoint yet — nothing to archive"
    return 0
  fi
  begin "save (${label}) — archiving checkpoints"
  local archive="${ARCHIVE_DIR}/${RUN_NAME}_${label}.tar"
  rm -f "${archive}" "${archive}.gz"
  tar -cf "${archive}" \
    "${PHASE2_DIR}/checkpoints" \
    "${PHASE1_DIR}/checkpoints/last.ckpt" \
    "${PHASE1_DIR}/.hydra" "${PHASE2_DIR}/.hydra" \
    "${SEMANTIC_TABLE}" 2>/dev/null || true
  # Only add eval artifacts if any exist; an empty dir just bloats the tar.
  if compgen -G "${EVAL_DIR}/*.jsonl" >/dev/null 2>&1; then
    tar -rf "${archive}" "${EVAL_DIR}" 2>/dev/null || true
    ok "included eval artifacts"
  fi
  [[ -d "${LOG_DIR}" ]] && tar -rf "${archive}" "${LOG_DIR}" 2>/dev/null || true
  gzip -f "${archive}"
  LAST_ARCHIVE="${archive}.gz"
  ok "archive: ${LAST_ARCHIVE} ($(du -h "${LAST_ARCHIVE}" | cut -f1))"

  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    say "uploading to W&B artifacts"
    WANDB_MODE=online python - "$LAST_ARCHIVE" "$RUN_NAME" "$label" <<'PY' || warn "W&B upload failed — the local archive is still there"
import sys, wandb
archive, run_name, label = sys.argv[1], sys.argv[2], sys.argv[3]
run = wandb.init(project="emoji-model", name=f"{run_name}_{label}", job_type="upload")
art = wandb.Artifact(f"{run_name}-checkpoints", type="model")
art.add_file(archive)
run.log_artifact(art)
run.finish()
print("    uploaded")
PY
  else
    warn "WANDB_API_KEY unset — archive is LOCAL ONLY on this rented box"
  fi

  cat <<EOF

  ${c_yel}Pull this to your laptop BEFORE releasing the instance:${c_off}

      scp <this-box>:$(readlink -f "${LAST_ARCHIVE}" 2>/dev/null || echo "${LAST_ARCHIVE}") ~/Downloads/

EOF
  finish "save (${label})"
}

if has_stage save; then
  do_save post_train
fi

# --------------------------------------------------------------------------
# Stage: eval
#
# Deliberately evaluated on the HELD-OUT split, not the original 24 groups.
# eval/HELDOUT_ORDER_EFFECT_RESULTS.md has frontier numbers on this split but
# no MDLM row, because the checkpoint was lost. These runs fill that gap, which
# is the open question; re-deriving the original-split number is not.
# --------------------------------------------------------------------------
if has_stage eval; then
  [[ -f "${CKPT}" ]] || die "no checkpoint at ${CKPT} — run STAGES=train first"
  say "evaluating ${CKPT}"

  begin "eval — order effect on held-out (uncapped)"
  python scripts/eval_order_effect.py \
    --checkpoint "${CKPT}" \
    --label "${RUN_NAME}_uncapped" \
    --reply-data-file "${HELDOUT}" \
    --data-cache "${DATA_CACHE}" \
    --vocab-cache "${VOCAB_CACHE}" \
    --num-groups "${NUM_GROUPS}" \
    --permutations-per-group 3 \
    --steps "${EVAL_STEPS}" \
    --length "${MODEL_LENGTH}" \
    --model "${MODEL}" \
    --device cuda \
    --jsonl-out "${EVAL_DIR}/order_effect_heldout_uncapped.jsonl" \
    2>&1 | tee "${LOG_DIR}/eval_order_uncapped.log"
  finish "order effect (uncapped)"

  begin "eval — order effect on held-out (capped-3)"
  python scripts/eval_order_effect.py \
    --checkpoint "${CKPT}" \
    --label "${RUN_NAME}_capped3" \
    --reply-data-file "${HELDOUT}" \
    --data-cache "${DATA_CACHE}" \
    --vocab-cache "${VOCAB_CACHE}" \
    --num-groups "${NUM_GROUPS}" \
    --permutations-per-group 3 \
    --steps "${EVAL_STEPS}" \
    --length "${MODEL_LENGTH}" \
    --model "${MODEL}" \
    --max-response-tokens 3 \
    --device cuda \
    --jsonl-out "${EVAL_DIR}/order_effect_heldout_capped3.jsonl" \
    2>&1 | tee "${LOG_DIR}/eval_order_capped3.log"
  finish "order effect (capped-3)"

  begin "eval — infilling on held-out (${EVAL_SAMPLES} samples)"
  python scripts/eval_infilling.py \
    --checkpoint "${CKPT}" \
    --eval-file "${HELDOUT}" \
    --data-cache "${DATA_CACHE}" \
    --vocab-cache "${VOCAB_CACHE}" \
    --samples "${EVAL_SAMPLES}" \
    --batch-size "${EVAL_BATCH}" \
    --steps "${EVAL_STEPS}" \
    --length "${MODEL_LENGTH}" \
    --model "${MODEL}" \
    --device cuda \
    --jsonl-out "${EVAL_DIR}/infill_heldout_samples${EVAL_SAMPLES}.jsonl" \
    2>&1 | tee "${LOG_DIR}/eval_infill.log"
  finish "infilling"

  cat <<EOF

  ${c_grn}Evaluation artifacts${c_off}
      ${EVAL_DIR}/order_effect_heldout_uncapped.jsonl
      ${EVAL_DIR}/order_effect_heldout_capped3.jsonl
      ${EVAL_DIR}/infill_heldout_samples${EVAL_SAMPLES}.jsonl

  These are the MDLM rows missing from eval/HELDOUT_ORDER_EFFECT_RESULTS.md.
  Compare order_effect against the frontier numbers already in that file.

  The frontier baselines are OpenRouter API calls — run them from your laptop,
  not this box:

      OPENROUTER_API_KEY=... python eval/order_eval.py \\
          --data ${HELDOUT} --num-groups ${NUM_GROUPS}

EOF

  # Re-archive so the eval artifacts travel with the weights.
  if has_stage save; then
    do_save final
  fi
fi

say "done — total elapsed $(( SECONDS / 60 ))m$(( SECONDS % 60 ))s"
if [[ -f "${CKPT}" ]]; then
  cat <<EOF

  ${c_red}Before you release this instance${c_off}, confirm you have pulled:
      ${ARCHIVE_DIR}/   (checkpoints + configs + eval artifacts)
EOF
fi
