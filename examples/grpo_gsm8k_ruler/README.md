# RL on GSM8K with RULER reward

A toy implementation of ART's RULER on GSM8k task and GRPO.

https://github.com/OpenPipe/ART/blob/main/src/art/rewards/ruler.py

https://art.openpipe.ai/fundamentals/ruler



The config files are located in [`gsm8k_ruler.yaml`](gsm8k_ruler.yaml) and [`train_gsm8k_ruler.yaml`](train_gsm8k_ruler.yaml).

Configs to pay attention to:
* `default_workflow_type`: set to `math_ruler_workflow`
* `auxiliary_models`: LLM-as-a-judge for RULER; need to set `max_prompt_tokens`, `max_response_tokens`, `max_model_len` appropriately
* `std_threshold` for GRPO advantage: set to small value, filter out group of experiences with same rewards (e.g., when RULER fails to return valid scores, they are set to all zero)
* `sync_style`: use `dynamic_by_explorer`, due to filtering of experiences
* `lr`: set to small value (2e-6) for stability, as rewards can be noisy

wandb metrics to pay attention to:
* `reward`: reward calculated by RULER
* `gold_reward`: sum of `accuracy_reward` and `format_reward`, rule-based calculation with ground truth (as in original GSM8k example)
