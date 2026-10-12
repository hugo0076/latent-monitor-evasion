"""The training objective and the loss computation.

The loss has three weighted terms (Equation 1 of the paper):

    l_obf       moves the active monitor score toward the target of the episode
    l_kl        keeps the next-token distribution close to the base model
    l_preserve  keeps the scores of the other 49 training monitors at their base values

All terms read the prefill tokens only, which are the supervised span (`labels != -100`).
The monitors read the mean over these tokens at the monitored layer, as in probe fitting.

l_obf is `(sigmoid(z) - target)^2 * scale_c`. The factor `scale_c` cancels the `1 / scale_c`
in the gradient, so every concept has the same gradient magnitude in activation space.

l_kl is KL(adapter || base), teacher-forced and averaged over the prefill positions.

l_preserve is the mean squared change in the probabilities of the other training monitors.
The base values come from an adapter-disabled forward pass on the same episode.

The held-out monitors are not in the stacked probe tensor, so no term can use them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core.activations import find_decoder_layers


@dataclass(frozen=True)
class LossWeights:
    obf: float = 0.9
    kl: float = 0.1
    preserve: float = 1.0
    ood: float = 0.0


@dataclass
class Monitors:
    """The stacked probes of the training concepts, on the device.

    Scores go through `core.probes._forward`, the same head code as probe fitting, so every
    monitor kind works, including the attention probe on unpooled tokens.
    """

    kind: str
    params: dict
    mu: torch.Tensor
    sd: torch.Tensor
    threshold: torch.Tensor
    scale: torch.Tensor
    names: list[str]

    def logits(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calibrated z for every training concept. hidden [B, T, D], mask [B, T] -> [B, K].

        Computed in float32, the precision of probe fitting. Gradients still reach the
        bfloat16 hidden states.
        """
        from core.probes import _forward

        # the monitored layer can sit on a different GPU from model.device
        dev = hidden.device
        P = {k: v.to(dev) for k, v in self.params.items()}
        P["mu"], P["sd"] = self.mu.to(dev), self.sd.to(dev)
        out = _forward(self.kind, P, hidden.float(), mask.to(dev).float())
        return (out - self.threshold.to(dev)) / self.scale.to(dev)


def response_mask(labels: torch.Tensor) -> torch.Tensor:
    """The supervised span, which is the prefill. [B, T] bool."""
    return labels != -100


def masked_kl(
    adapter_logits: torch.Tensor, base_logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """KL(adapter || base) averaged over the supervised positions."""
    sh_lab = labels[:, 1:]
    keep = sh_lab != -100
    if not keep.any():
        return adapter_logits.new_zeros(())
    log_p = F.log_softmax(adapter_logits[:, :-1][keep].float(), dim=-1)
    log_q = F.log_softmax(base_logits[:, :-1][keep].float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean()


def obfuscation_loss(
    z: torch.Tensor, effective_target: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """The obfuscation term. `z` and `scale` belong to the active concept of each example."""
    return ((torch.sigmoid(z) - effective_target) ** 2 * scale.detach()).mean()


@dataclass
class OODDetector:
    """A frozen Mahalanobis detector, fitted on single-sentence base-model activations.

    `d2 = ||(h_pooled - mu) Wh||^2`, where `Wh` whitens the shrunk covariance. `tau` is the 99th
    percentile of `d2` on the fitting pool, the operating point of the evaluation detector. The
    penalty keeps `d2` near benign values while the model moves the monitor score (Appendix C.3).
    """

    mu: torch.Tensor
    wh: torch.Tensor
    tau: float

    def d2(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dev = hidden.device
        m = mask.to(dev).float().unsqueeze(-1)
        pooled = (hidden.float() * m).sum(1) / m.sum(1).clamp_min(1.0)
        return (((pooled - self.mu.to(dev)) @ self.wh.to(dev)) ** 2).sum(-1)

    def penalty(
        self, hidden: torch.Tensor, mask: torch.Tensor, beta: float = 0.5
    ) -> torch.Tensor:
        """The OOD penalty: a softplus-smoothed hinge at `tau` with temperature `beta`
        (Appendix C.3), divided by `tau` so that a weight means the same across models."""
        d2 = self.d2(hidden, mask)
        return (F.softplus((d2 - self.tau) / (beta * self.tau)) * beta).mean()


def compute_losses(
    model,
    batch: dict[str, torch.Tensor],
    monitors: Monitors,
    layer: int,
    weights: LossWeights,
    ood: "OODDetector | None" = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One forward through the adapter and one with it disabled. Returns (loss, metrics)."""
    labels, topic = batch["labels"], batch["topic_idx"]
    target = batch["effective_target"].float()
    fwd = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "use_cache": False,
    }

    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out) -> None:
        captured["h"] = out[0] if isinstance(out, tuple) else out

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = find_decoder_layers(base_model)

    handle = decoder[layer].register_forward_hook(hook)
    try:
        out = model(**fwd)
    finally:
        handle.remove()
    # bind now: the disabled forward below overwrites `captured`
    h_adapter = captured["h"]
    # under device_map="auto" the layer, lm_head and batch can be on different GPUs
    dev = h_adapter.device
    labels, topic, target = labels.to(dev), topic.to(dev), target.to(dev)
    mask = response_mask(labels)
    z_all = monitors.logits(h_adapter, mask)
    col = topic.unsqueeze(1)
    z = z_all.gather(1, col).squeeze(1)

    with torch.no_grad(), model.disable_adapter():
        handle = decoder[layer].register_forward_hook(hook)
        try:
            base_out = model(**fwd)
        finally:
            handle.remove()
        z_base = monitors.logits(captured["h"], mask)
        p_base = torch.sigmoid(z_base.float())

    l_obf = obfuscation_loss(z, target, monitors.scale.to(dev)[topic])
    l_kl = masked_kl(out.logits, base_out.logits, labels.to(out.logits.device))

    p_all = torch.sigmoid(z_all.float())
    p_base = p_base.to(dev)
    keep = torch.ones_like(p_all).scatter_(1, col, 0.0)
    l_preserve = ((p_all - p_base) ** 2 * keep).sum() / keep.sum().clamp_min(1.0)

    l_ood = (
        ood.penalty(h_adapter, mask)
        if (ood is not None and weights.ood)
        else torch.zeros((), device=z.device)
    )

    loss = (
        weights.obf * l_obf.to(dev)
        + weights.kl * l_kl.to(dev)
        + weights.preserve * l_preserve.to(dev)
        + weights.ood * l_ood.to(dev)
    )
    metrics = {
        "loss": loss.detach().item(),
        "l_obf": l_obf.detach().item(),
        "l_kl": l_kl.detach().item(),
        "l_preserve": l_preserve.detach().item(),
        "l_ood": float(l_ood.detach()),
        "d2_over_tau": (
            float((ood.d2(h_adapter, mask).detach() / ood.tau).mean())
            if ood is not None
            else 0.0
        ),
        "z_mean": z.detach().mean().item(),
        "z_target1": z.detach()[target > 0.5].mean().item()
        if (target > 0.5).any()
        else float("nan"),
        "z_target0": z.detach()[target < 0.5].mean().item()
        if (target < 0.5).any()
        else float("nan"),
        "p_flag": (torch.sigmoid(z.detach()) > 0.5).float().mean().item(),
        "preserve_drift": (
            ((p_all - p_base).abs() * keep).sum() / keep.sum().clamp_min(1.0)
        )
        .detach()
        .item(),
    }
    return loss, metrics
