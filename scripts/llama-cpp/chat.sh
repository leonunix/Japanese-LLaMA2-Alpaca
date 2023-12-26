#!/bin/bash

# temporary script to chat with Japanese Alpaca-2 model
# usage: ./chat.sh alpaca2-ggml-model-path your-first-instruction

SYSTEM_PROMPT='You are a helpful assistant. あなたは助けを惜しまないアシスタントです。'
# SYSTEM_PROMPT='You are a helpful assistant. あなたは助けることを喜びとするアシスタントです。専門的であり、論理的で、内容が真実でありながら価値のある詳細な返答を提供してください。' # Try this one, if you prefer longer response.
MODEL_PATH=$1
FIRST_INSTRUCTION=$2

./main -m "$MODEL_PATH" \
--color -i -c 4096 -t 8 --temp 0.5 --top_k 40 --top_p 0.9 --repeat_penalty 1.1 \
--in-prefix-bos --in-prefix ' [INST] ' --in-suffix ' [/INST]' -p \
"[INST] <<SYS>>
$SYSTEM_PROMPT
<</SYS>>

$FIRST_INSTRUCTION [/INST]"
