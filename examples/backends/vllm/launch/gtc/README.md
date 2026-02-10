# Spec

Qwen3 VL
https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor
Default patch size: 16

# Scenario A

## Input

System prompts
8,600 = 5,500 + 2,600 + 500

User input
4,300 = 3,000 + 300 + 1,000

Output token
500

3 512*512 images per request;
Say x requests have y unique images which appear z total times.
There is 20% apperance are duplicates.
In other word, if we have an infinity large embedding cache, there will be 20% cache rate.

## Generate

```
python examples/backends/vllm/launch/gtc/generate_requests.py \
  -n 100 --cache-hit-rate 0.2
```

## Run

```
aiperf profile \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --input-file examples/backends/vllm/launch/gtc/requests.jsonl \
  --custom-dataset-type single_turn \
  --shared-system-prompt-length 8600 \
  --extra-inputs "max_tokens:500" \
  --extra-inputs "min_tokens:500" \
  --extra-inputs "ignore_eos:true"
```

# Scenario B

## Input

9K Text input tokens
- 5K fixed System prompt
- 4K User input

500 Output tokens

Images
- 20 images per request
  - "The number of images is between 10 and 50 per request"
- choose 256 tokens per image
  - "There are two image token configurations (128 and 256)"
- There is 27% apperance are duplicates.
  - "within a 24-hour period, the duplication rate of images was 27%."
  - Say x requests have y unique images which appear z total times. 27% of z are seen before. In other word, if we have an infinity large embedding cache, there will be 27% cache rate.


## Generate

```
python examples/backends/vllm/launch/gtc/generate_requests.py \
  -n 100 \
  --images-per-request 20 \
  --cache-hit-rate 0.27 \
  --user-text-tokens 4000
```

## Run

```
aiperf profile \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --input-file examples/backends/vllm/launch/gtc/100req_20img_27pct_4000word.jsonl \
  --custom-dataset-type single_turn \
  --shared-system-prompt-length 5000 \
  --extra-inputs "max_tokens:500" \
  --extra-inputs "min_tokens:500" \
  --extra-inputs "ignore_eos:true" \
  --request-count 100 \
  --warmup-request-count 10 \
  --artifact-dir /workspace/logs/aiperf
```