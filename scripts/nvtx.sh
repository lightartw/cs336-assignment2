#!/bin/bash

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

MODEL_SIZE=$1
CONTEXT_LENGTH=$2
FORWARD_MODE=$3 

OUTPUT_PATH="${ROOT_DIR}/result/profile_${MODEL_SIZE}_ctx${CONTEXT_LENGTH}"
FORWARD_ARG=""


if [ "$FORWARD_MODE" == "forward-only" ]; then
    OUTPUT_PATH="${OUTPUT_PATH}_forward"
    FORWARD_ARG="--only-forward" 
elif [ "$FORWARD_MODE" == "--no-forward-only" ]; then
    FORWARD_ARG=""
else
    echo "Warning: Third argument should be 'forward-only' or '--no-forward-only'. Proceeding with default."
fi

mkdir -p "${ROOT_DIR}/result"

set -x 
nsys profile -w true -t cuda,nvtx,osrt \
    -o "$OUTPUT_PATH" \
    python "${ROOT_DIR}/main.py" sweep \
    --model-sizes "$MODEL_SIZE" \
    --context-lengths "$CONTEXT_LENGTH" \
    --use-nvtx $FORWARD_ARG
{ set +x; } 2>/dev/null