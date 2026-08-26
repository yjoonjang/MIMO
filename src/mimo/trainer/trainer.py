"""MIMOTrainer: SentenceTransformerTrainer extended with projection management."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformerTrainer

logger = logging.getLogger(__name__)


class MIMOTrainer(SentenceTransformerTrainer):
    """Extended trainer for MIMO.

    Key extensions:
    - save_model: also saves projection.pt alongside the model.
    - _wrap_model: enables DDP static_graph (see below).
    - compute_loss: calls the DDP-wrapped model directly to avoid nested
      Module forward issues with gradient caching.
    """

    def add_model_card_callback(self, default_args_dict: dict | None = None) -> None:
        """Skip the model card callback — the training datasets here lack the
        metadata it expects."""
        pass

    def __init__(
        self,
        projection: torch.nn.Linear | None = None,
        teacher_model=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.projection = projection
        # Store teacher_model without nn.Module registration so the trainer
        # does not traverse into the (frozen) teacher.
        object.__setattr__(self, "_teacher_model", teacher_model)

    @property
    def teacher_model(self):
        return self._teacher_model

    def _wrap_model(self, model, training=True, dataloader=None):
        """Inject static_graph=True into the DDP config.

        Encoder backbones like XLM-R have pooler parameters that never receive
        gradients under mean pooling, which breaks both settings of
        find_unused_parameters. static_graph=True records parameter usage in
        the first iteration and handles both cases.
        """
        result = super()._wrap_model(model, training=training, dataloader=dataloader)
        if (
            training
            and hasattr(self.accelerator, "ddp_handler")
            and self.accelerator.ddp_handler is not None
        ):
            self.accelerator.ddp_handler.static_graph = True
        return result

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """Save the model and, if present, the projection layer."""
        super().save_model(output_dir, _internal_call=_internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.projection is not None and self.args.should_save:
            projection_path = os.path.join(output_dir, "projection.pt")
            torch.save(self.projection.state_dict(), projection_path)
            logger.info("Projection layer saved to %s", projection_path)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute the loss by calling the DDP-wrapped model directly.

        Calling DDP.forward() from within a nested nn.Module's forward()
        causes "marked as ready twice" errors because DDP's autograd hooks
        conflict with the nested call stack, so the loss is orchestrated here.
        """
        sentence_features = inputs

        if self.args.stage == 1:
            loss = self._compute_stage1_loss(model, sentence_features)
        else:
            loss = self._compute_stage2_loss(model, sentence_features)

        if return_outputs:
            return loss, {}
        return loss

    def _compute_stage1_loss(self, model, sentence_features):
        """Stage 1: embedding distillation via cosine distance."""
        student_sf = sentence_features[0]
        teacher_sf = sentence_features[1]

        student_output = model(student_sf)
        student_emb = student_output["sentence_embedding"]

        self.projection = self.projection.to(device=student_emb.device, dtype=student_emb.dtype)
        projected = self.projection(student_emb)

        with torch.no_grad():
            teacher_output = self.teacher_model(teacher_sf)
            teacher_emb = teacher_output["sentence_embedding"]
        teacher_emb = teacher_emb.to(device=projected.device, dtype=projected.dtype)

        distance_metric = getattr(self.args, "distance_metric", "cosine")
        if distance_metric == "cosine":
            loss = (1 - F.cosine_similarity(projected, teacher_emb, dim=-1)).mean()
        elif distance_metric == "mse":
            loss = F.mse_loss(projected, teacher_emb)
        elif distance_metric == "l2":
            loss = torch.norm(projected - teacher_emb, dim=-1).mean()
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")

        return loss

    def _compute_stage2_loss(self, model, sentence_features):
        """Stage 2: delegate to the gradient-cached loss, injecting the
        DDP-wrapped model so mini-batch forwards run through DDP."""
        loss_fn = self.loss
        if (
            model == self.model_wrapped
            and hasattr(loss_fn, "model")
            and loss_fn.model != model
        ):
            loss_fn.model = model

        loss = loss_fn(sentence_features, None)

        # Log individual loss components if available
        if self.state.global_step % self.args.logging_steps == 0:
            logs = {}
            if hasattr(loss_fn, "_last_infonce_loss"):
                logs["infonce_loss"] = loss_fn._last_infonce_loss
            if hasattr(loss_fn, "_last_distill_loss"):
                logs["distill_loss"] = loss_fn._last_distill_loss
            if hasattr(loss_fn, "_last_kl_loss"):
                logs["kl_loss"] = loss_fn._last_kl_loss
            if logs and self.args.report_to and "wandb" in self.args.report_to:
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log(logs, step=self.state.global_step)
                except ImportError:
                    pass

        return loss
