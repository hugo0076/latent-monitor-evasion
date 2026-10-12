"""Model loading, prompt rendering and residual-stream extraction.

`render` is the only `apply_chat_template` call, so probe fitting, training and evaluation
all see the same prompt format.

Activations are stored per token and packed: one `[total_tokens, D]` array and an index of
offsets. `pool` is the only mean-pooling function. `unpack` and `pad_blocks` restore
per-sentence shapes. Activations are float32, because gemma-3-27b reaches 65280 at layer 46,
which overflows float16.

To fit the probes, each sentence is a single-turn exchange: the user says "Say something."
and the assistant says the sentence. Only the assistant tokens are kept.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODELS: dict[str, dict] = {
    "llama-3.1-8b": {"hf_name": "meta-llama/Llama-3.1-8B-Instruct", "layer": 24},
    "llama-3.3-70b": {"hf_name": "meta-llama/Llama-3.3-70B-Instruct", "layer": 60},
    "gemma-3-27b": {"hf_name": "google/gemma-3-27b-it", "layer": 46},
    "qwen3-14b": {"hf_name": "Qwen/Qwen3-14B", "layer": 30},
}

PROMPT = [{"role": "user", "content": "Say something."}]


def model_dir(outputs_root: Path, model: str, layer: int | None = None) -> Path:
    """The output directory of one model: `outputs/<model>/`, or `outputs/<model>_L<layer>/`
    when `layer` is not the default."""
    default = MODELS[model]["layer"]
    suffix = "" if layer is None or layer == default else f"_L{layer}"
    return Path(outputs_root) / f"{model}{suffix}"


def activations_dir(
    outputs_root: Path, model: str, *, source: str = "base", layer: int | None = None
) -> Path:
    """Activations produced by `source`: "base", or the run name of an adapter."""
    return model_dir(outputs_root, model, layer) / "activations" / source


def probes_dir(
    outputs_root: Path, model: str, *, source: str = "base", layer: int | None = None
) -> Path:
    """Probes and the PC basis, fitted on the activations of `source`.

    "base" is the defender who calibrated before the attack. An adapter name gives monitors
    refitted on that model organism.
    """
    return model_dir(outputs_root, model, layer) / "probes" / source


def find_decoder_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """Locate the decoder layer list across architectures."""
    for path in (
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        obj: object = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError(f"could not find decoder layers in {type(model).__name__}")


def require_cuda() -> None:
    """Raise if torch cannot see a GPU.

    Without a GPU, the stages that load a model run on CPU and take days. The usual cause is
    a torch build for a newer CUDA than the driver, which makes `torch.cuda.is_available()`
    return False instead of raising.
    """
    if torch.cuda.is_available():
        return
    built = torch.version.cuda or "cpu-only build"
    raise RuntimeError(
        f"torch {torch.__version__} cannot see a GPU (built for CUDA {built}).\n"
        "This stage needs one; on CPU it would run for days rather than hours.\n"
        "If the machine does have a GPU, the torch build and the driver disagree — install "
        "a torch matching your driver's CUDA version, e.g.\n"
        "    uv pip install torch --torch-backend=cu128\n"
        "Pass check_cuda=False to load_model to run on CPU anyway."
    )


def load_model(
    hf_name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn: str = "sdpa",
    check_cuda: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model in eval mode, with right-padding.

    Right-padding is necessary because the response span starts at the token length of the
    prompt. Attention interventions use `attn="eager"`, the only mode that builds an explicit
    mask for a hook to edit.
    """
    if check_cuda:
        require_cuda()
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, device_map="auto", dtype=dtype, attn_implementation=attn
    )
    model.eval()
    return model, tokenizer


def render(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[dict[str, str]],
    response: str,
) -> tuple[str, str]:
    """Return `(prompt_text, prompt_text + response)`. The difference in token length is the
    response span."""
    prompt_text = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True
    )
    return prompt_text, prompt_text + response


def render_single(tokenizer: PreTrainedTokenizerBase, sentence: str) -> tuple[str, str]:
    """The single-turn context that the monitors are fitted on."""
    return render(tokenizer, PROMPT, sentence)


def response_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer: int,
    messages: Sequence[dict[str, str]],
    response: str,
    *,
    max_length: int = 8192,
) -> np.ndarray:
    """Per-token residual stream over `response` only, given any message history. [T, D] fp32."""
    device = next(model.parameters()).device
    prompt_text, full_text = render(tokenizer, messages, response)
    n_prompt = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    ).to(device)
    # n_prompt counts the untruncated prompt, so an overflow leaves an empty slice
    if enc.input_ids.shape[1] <= n_prompt:
        raise ValueError(
            f"context overflow: prompt is {n_prompt} tokens but the sequence was truncated to "
            f"{enc.input_ids.shape[1]} at max_length={max_length}, so the response span is "
            f"empty. Raise max_length or shorten the episode."
        )

    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, output) -> None:
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = find_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                use_cache=False,
            )
    finally:
        handle.remove()
    return captured["h"][0, n_prompt:].float().cpu().numpy()


def pack(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-sentence [T_i, D] arrays into `(tokens, offsets)`.

    `offsets` has N+1 entries. Sentence i is `tokens[offsets[i]:offsets[i+1]]`.
    """
    lengths = [len(r) for r in rows]
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = (np.concatenate(rows) if rows else np.zeros((0, 0), np.float32)).astype(
        np.float32
    )
    return tokens, offsets


def unpack(tokens: np.ndarray, offsets: np.ndarray, i: int) -> np.ndarray:
    """The [T_i, D] token block for sentence `i`."""
    return tokens[offsets[i] : offsets[i + 1]]


def pool(tokens: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean over each sentence's response tokens. The only pooling function.

    Returns float32 [N, D]. A sentence with no response tokens pools to zeros, not NaN.
    """
    n, dim = len(offsets) - 1, tokens.shape[-1]
    out = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        block = tokens[offsets[i] : offsets[i + 1]]
        if len(block):
            out[i] = block.astype(np.float32).mean(axis=0)
    return out


def pad_blocks(
    blocks: Sequence[np.ndarray], width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Pad per-sentence `[T_i, D]` blocks into `(H [n, T, D], mask [n, T])`.

    `width` defaults to the longest block. Give it explicitly when several batches must align.
    """
    dim = blocks[0].shape[-1] if len(blocks) else 0
    width = max((len(b) for b in blocks), default=0) if width is None else width
    H = np.zeros((len(blocks), width, dim), dtype=np.float32)
    mask = np.zeros((len(blocks), width), dtype=np.uint8)
    for j, block in enumerate(blocks):
        keep = min(len(block), width)
        H[j, :keep] = block[:keep]
        mask[j, :keep] = 1
    return H, mask


def extract(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer: int,
    sentences: Sequence[str],
    *,
    max_response_tokens: int,
    max_length: int,
    batch_size: int,
) -> tuple[list[np.ndarray], int]:
    """Per-token residual-stream activations for a list of sentences.

    Returns `(rows, n_truncated)`. `rows[i]` is the float32 `[T_i, D]` block of sentence i.
    `n_truncated` counts responses longer than `max_response_tokens`.
    """
    device = next(model.parameters()).device
    layers = find_decoder_layers(model)
    rows: list[np.ndarray] = []
    n_truncated = 0
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output) -> None:
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    for start in range(0, len(sentences), batch_size):
        batch = list(sentences[start : start + batch_size])
        rendered = [render_single(tokenizer, s) for s in batch]
        enc = tokenizer(
            [full for _, full in rendered],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,  # the chat template already emitted BOS
        ).to(device)

        handle = layers[layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()
        h = captured["h"].float().cpu().numpy()

        for j, (prompt_text, _) in enumerate(rendered):
            n_prompt = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
            n_total = int(enc.attention_mask[j].sum().item())
            resp = h[j, n_prompt:n_total]
            if len(resp) > max_response_tokens:
                n_truncated += 1
            rows.append(resp[:max_response_tokens].astype(np.float32))

    return rows, n_truncated
