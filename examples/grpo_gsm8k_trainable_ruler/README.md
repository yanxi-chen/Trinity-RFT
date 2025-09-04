# Train one model for both policy and reward


Ref: ART's RULER; Kimi-k2.


Simulate a scenario where only a fraction of tasks have ground-truth answers for rule-based reward.



## Configurations and Metrics

The config files are located in [`gsm8k_ruler.yaml`](gsm8k_ruler.yaml) and [`train_gsm8k_ruler.yaml`](train_gsm8k_ruler.yaml).

Some key configs in this example are:

(TODO)


Some important metrics to pay attention to are:

(TODO)


## Results

(TODO)

Compare with baseline: previous RULER workflow with Qwen2.5-1.5B-Instruct as LLM judge (`auxiliary_models`)



## Potential improvements

balance number of samples / loss weights for generation vs RULER
