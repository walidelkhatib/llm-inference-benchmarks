#!/usr/bin/env bash
# Runs vllm bench serve at four concurrency levels against a running server.
set -euo pipefail
mkdir -p results

run() {
  local conc=$1 prompts=$2
  vllm bench serve --model Qwen/Qwen3-8B --dataset-name random \
    --random-input-len 512 --random-output-len 256 \
    --num-prompts "$prompts" --max-concurrency "$conc" | tee "results/bench_c${conc}.txt"
}

run 1 20
run 5 50
run 20 100
run 40 200
