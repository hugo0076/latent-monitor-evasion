# LLMs Learn to Evade Latent Monitors from Prior Feedback Alone

Code for the paper. It fits latent-space monitors (logistic, MLP and attention probes) for 60 concepts on a model's residual stream, builds multi-turn 'episodes' in which a monitor's binary verdicts on earlier responses are shown to the model, trains rank-1 LoRA model organisms, and measures activation edits the models make on a final, prefilled turn when asked to imitate one of the two verdict labels.

```
data/raw/          the 60 concept pools
core/              shared code
01..07_*.py        the pipeline stages
experiments/       ~one script per study in the paper
figures/           one script per paper figure
```

Every script writes to `outputs/<model>/` (the pipeline) or `experiments/outputs/<study>/` (the studies).

## Install

```
uv sync
uv pip install torch --torch-backend=cu128
```

Stages and studies that load a model need GPU and access to (potentially) gated model weights on the Hugging Face Hub.

## Data

`data/raw/{concept}.jsonl` holds 500 synthetic sentences for each of 60 concepts, one JSON object per line with the fields `concept`, `subtopic`, `sentence`, `char_len`, `word_count`, `approx_token_len` and `source`. Appendix A.1 describes how they were generated.

`core/concepts.py` splits the concepts into 50 training concepts and 10 held-out concepts: `deception`, `physics`, `astronomy`, `portuguese`, `archaeology`, `sarcastic`, `sycophancy`, `evasiveness_noncommittal`, `imperatives_commands`, `urgency_panic`.

## Pipeline

To reproduce the results in the paper (for `llama-3.1-8b`):

```
uv run python 01_split_pools.py
uv run python 02_extract_activations.py --model llama-3.1-8b
uv run python 03_fit_probes.py --model llama-3.1-8b --device cuda
uv run python 04_train_adapter.py --run r1_sigmoid_mse
uv run python 05_eval.py
uv run python 05_eval.py --adapter r1_sigmoid_mse
uv run python 06_analyse.py --run base
uv run python 06_analyse.py --run r1_sigmoid_mse
uv run python 07_geometry.py --run r1_sigmoid_mse --compare base
```

A trained adapter is a PEFT directory (`adapter_config.json`, `adapter_model.safetensors` and `config.json`) at `outputs/<model>/adapters/<run>/`. `--adapter <run>` in the later stages takes the run name. The adapters used in the paper are published at [redacted for review].

## Experiments and Figure replication

Each study reads the pipeline outputs and writes one JSON file under `experiments/outputs/<study>/`.

`figures/fig_*.py` draws the paper's figures from these outputs, and from the evals for Figures 1, 2 and 4. Output goes to `figures/outputs/`.

## License

The code is released under the MIT License. The data in `data/raw/` is released under CC BY 4.0.
