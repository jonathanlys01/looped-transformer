#!/bin/bash

set -e

export HF_HUB_DISABLE_PROGRESS_BARS=1
MODEL_ID="meta-llama/Meta-Llama-3-8B" # "google/gemma-2-2b", "google/gemma-2-9b"

INTERPOLATION_MODE="uniform" # uniform, ema, auto-alignment, no
PLAN="10 13 3"


python main_accelerate.py $MODEL_ID "leaderboard|winogrande|5|0" \
   --use-looped-model --interpolation-mode $INTERPOLATION_MODE --plan $PLAN

python main_accelerate.py $MODEL_ID "lighteval|arc:easy|0|0" \
    --use-looped-model --interpolation-mode $INTERPOLATION_MODE --plan $PLAN

python main_accelerate.py $MODEL_ID "leaderboard|arc:challenge|25|0" \
    --use-looped-model --interpolation-mode $INTERPOLATION_MODE --plan $PLAN

python main_accelerate.py $MODEL_ID "leaderboard|hellaswag|10|0" \
    --use-looped-model --interpolation-mode $INTERPOLATION_MODE --plan $PLAN

python main_accelerate.py $MODEL_ID "original|mmlu|5|0" \
    --use-looped-model --interpolation-mode $INTERPOLATION_MODE --plan $PLAN
