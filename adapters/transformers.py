"""HF Trainer adapter — drop-in compute_loss override.

Wires CTDLoss into a HuggingFace `Trainer` subclass. Use as:

    from ctd.losses import CTDLoss
    from adapters.transformers import CTDTrainerMixin
    import torch
    from transformers import Trainer

    class MyTrainer(CTDTrainerMixin, Trainer):
        pass

    cache = torch.load("ctd_cache.pt")
    trainer = MyTrainer(
        model=student_model,
        args=training_args,
        train_dataset=ds,
        ctd_cache=cache,
        ctd_loss=CTDLoss(kind="kl"),
        ctd_weight=0.5,           # weight of distill term vs hard CE
    )
    trainer.train()

The adapter expects each training batch to include a tensor of token
positions ('ctd_positions') so we can look up the correct cache rows.
"""

from __future__ import annotations

from typing import Optional


class CTDTrainerMixin:
    """Mixin that adds CTD distillation to HF Trainer's compute_loss.

    Subclass this BEFORE Trainer (`class MyTrainer(CTDTrainerMixin, Trainer)`)
    so the MRO picks up our compute_loss override.
    """

    def __init__(
        self,
        *args,
        ctd_cache: Optional[dict] = None,
        ctd_loss: Optional["CTDLoss"] = None,  # noqa: F821 (fwd ref)
        ctd_weight: float = 0.5,
        ctd_position_field: str = "ctd_positions",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ctd_cache = ctd_cache
        self.ctd_loss = ctd_loss
        self.ctd_weight = ctd_weight
        self.ctd_position_field = ctd_position_field

        if (ctd_cache is not None) != (ctd_loss is not None):
            raise ValueError("Provide both ctd_cache and ctd_loss, or neither.")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute hard-CE + ctd_weight * CTD-distill loss."""
        # Pull ctd_positions out before forward (HF model won't accept it).
        ctd_positions = inputs.pop(self.ctd_position_field, None)

        # Standard CE first.
        outputs = model(**inputs)
        ce_loss = outputs.loss

        if self.ctd_cache is None or ctd_positions is None:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # Look up teacher cache at ctd_positions.
        # ctd_positions: [B, L] — global token indices into the cache.
        cache_values = self.ctd_cache["values"]   # [N_total, K]
        cache_indices = self.ctd_cache["indices"]  # [N_total, K]
        cache_mask = self.ctd_cache["mask"]        # [N_total]

        device = outputs.logits.device
        flat_pos = ctd_positions.view(-1).cpu()
        teacher_topk_log = cache_values[flat_pos].to(device)
        teacher_topk_idx = cache_indices[flat_pos].to(device)
        align_mask = cache_mask[flat_pos].to(device).view(*ctd_positions.shape)

        teacher_topk_log = teacher_topk_log.view(*ctd_positions.shape, -1)
        teacher_topk_idx = teacher_topk_idx.view(*ctd_positions.shape, -1)

        # Shift student logits to predict the NEXT token (HF convention:
        # logits[..., i, :] predicts input_ids[..., i+1]).
        # Caller is responsible for ensuring ctd_positions matches the
        # shifted layout, OR we can shift here. Default: assume aligned.
        student_logits = outputs.logits

        distill_loss = self.ctd_loss(
            student_logits=student_logits,
            teacher_topk_log_values=teacher_topk_log,
            teacher_topk_indices=teacher_topk_idx,
            alignment_mask=align_mask,
        )

        total = (1.0 - self.ctd_weight) * ce_loss + self.ctd_weight * distill_loss
        # Stash for logging.
        outputs["ctd_distill_loss"] = distill_loss.detach()
        outputs["ctd_ce_loss"] = ce_loss.detach()

        return (total, outputs) if return_outputs else total
