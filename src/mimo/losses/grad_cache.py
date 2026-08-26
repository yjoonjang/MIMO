"""Shared gradient-caching machinery for large-batch contrastive training.

Gradient caching (Gao et al., 2021) decouples the effective batch size from GPU
memory: embeddings are first computed without gradients in mini-batches, the
loss and its gradients w.r.t. the embeddings are computed on the full batch,
and a backward hook then re-runs each mini-batch forward pass with gradients
enabled, backpropagating the cached embedding gradients.

Reference implementation: https://github.com/luyug/GradCache
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import SentenceTransformer


class RandContext:
    """Random-state context manager for reproducible gradient caching."""

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: list[dict[str, Tensor]],
    loss_obj: "CachedLossBase",
) -> None:
    """Backpropagate cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(
            sentence_features[: loss_obj.num_grad_inputs], loss_obj.cache, loss_obj.random_states
        ):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                if reps_mb.requires_grad:
                    surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                    surrogate.backward()


class CachedLossBase(nn.Module):
    """Base class for gradient-cached losses.

    Subclasses set ``num_grad_inputs`` (how many leading sentence features
    receive gradients) and implement the loss computation on the concatenated
    embeddings.
    """

    num_grad_inputs: int = 2

    def __init__(
        self,
        model: SentenceTransformer,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar

        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed one mini-batch of sentences with the student model."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Iterate over mini-batches for embedding."""
        input_ids: Tensor = sentence_feature["input_ids"]
        batch_size = input_ids.shape[0]
        for i, begin in enumerate(
            tqdm.trange(
                0,
                batch_size,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            end = begin + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state

    def embed_inputs_without_grad(
        self, sentence_features: list[dict[str, Tensor]]
    ) -> list[list[Tensor]]:
        """Step 1 of gradient caching: no-grad embeddings for the grad inputs.

        Populates ``self.random_states`` and returns per-input lists of
        detached mini-batch embeddings that require grad.
        """
        student_reps: list[list[Tensor]] = []
        self.random_states = []
        for sentence_feature in sentence_features[: self.num_grad_inputs]:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            student_reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)
        return student_reps

    def cache_gradients_and_register_hook(
        self, loss: Tensor, student_reps: list[list[Tensor]], sentence_features: list[dict[str, Tensor]]
    ) -> Tensor:
        """Steps 3-4 of gradient caching: backward on the embedding-level loss,
        cache the embedding gradients, and register the re-forward hook."""
        loss.backward()
        self.cache = [[r.grad for r in rs] for rs in student_reps]
        loss = loss.detach().requires_grad_()
        loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        return loss
