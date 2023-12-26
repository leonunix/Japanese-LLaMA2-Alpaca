#!/bin/bash

# NOTE: start the server first before running this script.
# usage: ./server_curl_example.sh your-instruction

SYSTEM_PROMPT='You are a helpful assistant. あなたは助けを惜しまないアシスタントです。'
# SYSTEM_PROMPT='You are a helpful assistant. あなたは助けることを喜びとするアシスタントです。専門的であり、論理的で、内容が真実でありながら価値のある詳細な返答を提供してください。'  # Try this one, if you prefer longer response.
INSTRUCTION=$1
ALL_PROMPT="[INST] <<SYS>>\n$SYSTEM_PROMPT\n<</SYS>>\n\n$INSTRUCTION [/INST]"
CURL_DATA="{\"prompt\": \"$ALL_PROMPT\",\"n_predict\": 128}"

curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data "$CURL_DATA"
