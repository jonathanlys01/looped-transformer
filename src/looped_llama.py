"""Modelling code for Looped Llama."""

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Unpack

import torch
from torch import nn
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.cache_utils import Cache, DynamicLayer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.utils import TransformersKwargs, can_return_tuple

from format_utils import format_exec_plan


torch.set_float32_matmul_precision("high")


# Constants
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
CACHE_DIR = os.path.join(os.environ.get("SCRATCH", "/SCRATCH"), "llama-exps")


AUTO_PLAN = (0, 0, 0)
DEFAULT_TEMPERATURE = 1.0  # Default temperature for auto-alignment interpolation


@dataclass
class LoopMetadata:
    loop_iters: tuple[Optional[int], ...]
    loop_layer_idx: tuple[Optional[int], ...]
    exec_plan: tuple[int, ...]

    def iter(self):
        return zip(
            range(len(self.loop_iters)),  # virtual loop index
            self.loop_iters,  # actual loop iteration
            self.loop_layer_idx,  # loop layer index
            self.exec_plan,  # layer index
        )


# Model
class HackedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(self, *args, **kwargs):
        _layer_idx = kwargs.pop("layer_idx", None)

        if _layer_idx is not None:
            assert isinstance(_layer_idx, dict) and "layer_idx" in _layer_idx and "virtual_layer_idx" in _layer_idx, (
                "Expected special_layer_ids to be a dict with 'layer_idx' and 'virtual_layer_idx' keys."
            )

            virtual_layer_idx = _layer_idx["virtual_layer_idx"]

            # Cache fixes
            self.self_attn.layer_idx = virtual_layer_idx

        return super().forward(*args, **kwargs)


class HackedLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.layers = nn.ModuleList([HackedLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

        self.config.exec_plan = None
        self.known_exec_plan = False
        self.loop_plan = None

        self.interpolation_mode = "no"
        self.cache = None
        self.cache_weights = None
        self.loop_metadata = None

    def build_exec_plan(self, plan: tuple[int, int, int] = None) -> None:  # noqa: PLR0915
        """
        Build the execution plan for the model based on the looping plans.
        """

        og_num_hidden_layers = self.config.num_hidden_layers

        if plan is None:
            loop_metadata = LoopMetadata(
                loop_iters=tuple([None] * og_num_hidden_layers),
                loop_layer_idx=tuple([None] * og_num_hidden_layers),
                exec_plan=tuple(range(og_num_hidden_layers)),
            )
            self.loop_metadata = loop_metadata
            return

        assert len(plan) == 3, f"Expected plan to be a tuple of (start, end, n_loops), got {plan}."
        loop_start = plan[0]
        loop_end = plan[1]
        n_loops = plan[2]
        len_ = loop_end - loop_start

        assert 0 <= loop_start < loop_end <= og_num_hidden_layers, (
            f"Invalid execution plan: {plan}. "
            f"Expected start >= 0, end <= {og_num_hidden_layers}, got start={loop_start}, end={loop_end}."
        )

        # Loop metadata
        loop_iters = [None for _ in range(loop_start)]
        loop_layer_idx = [None for _ in range(loop_start)]
        exec_plan = list(range(loop_start))

        for loop_iter in range(n_loops):
            loop_iters.extend([loop_iter] * len_)
            loop_layer_idx.extend(range(len_))
            exec_plan.extend(range(loop_start, loop_end))

        remaining_layers = og_num_hidden_layers - loop_end
        exec_plan.extend(range(loop_end, og_num_hidden_layers))
        loop_iters.extend([None] * remaining_layers)
        loop_layer_idx.extend([None] * remaining_layers)

        loop_metadata = LoopMetadata(
            loop_iters=tuple(loop_iters),
            loop_layer_idx=tuple(loop_layer_idx),
            exec_plan=tuple(exec_plan),
        )

        self.loop_metadata = loop_metadata
        self.loop_plan = (len_, n_loops)

        # In place modif for:
        # - auto-generation of the generation config in lighteval
        # - cache initialization

        self.config.num_hidden_layers = len(exec_plan)
        self.config.exec_plan = exec_plan

        if self.interpolation_mode == "auto-alignment":  # no weights needed
            return

        # Generate weights for interpolation

        if hasattr(self, "cache_weights") and self.cache_weights is None:
            print("Reinitializing cache weights.")
            del self.cache_weights

        n_layers_looped = loop_end - loop_start
        c_weights_ = self._generate_weights(
            interpolation_mode=self.interpolation_mode,
            num_layers_looped=n_layers_looped,
            num_loops=n_loops,
            device=self.device,
        )

        self.cache_weights = nn.Parameter(c_weights_)

    @can_return_tuple
    def forward(  # noqa: C901, PLR0912, PLR0915
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # past_key_values = DynamicCache(config=self.config)
            past_key_values = Cache(layers=[DynamicLayer() for _ in range(self.exec_plan)])

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        assert (
            len(self.layers) == self.config.num_hidden_layers or len(self.config.exec_plan or []) == self.config.num_hidden_layers
        ), f"Expected either n_layer or exec plan of the same length, but got {len(self.layers)} layers."

        exec_plan = range(self.config.num_hidden_layers) if self.config.exec_plan is None else self.config.exec_plan

        if not self.known_exec_plan:
            print(f"Exec plan: {format_exec_plan(exec_plan)}")
            self.known_exec_plan = True  # only print once

        if not hasattr(self, "loop_metadata") or self.loop_metadata is None:
            raise ValueError("Loop metadata is not initialized. Please call `build_exec_plan` first.")
        iterable_metadata = self.loop_metadata

        for virtual_idx, loop_iter, loop_layer_idx, layer_idx in iterable_metadata.iter():
            decoder_layer = self.layers[layer_idx]

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                special_layer_ids={
                    "layer_idx": layer_idx,  # actual layer index (for sliding attention)
                    "virtual_layer_idx": virtual_idx,  # virtual layer index (for cache)
                },
                **kwargs,
            )

            hidden_states = layer_outputs

            hidden_states = self.unified_cached_interpolation(
                x=layer_outputs,
                loop_iter=loop_iter,
                loop_layer_idx=loop_layer_idx,
            )

        self.cache = None  # reset cache after use

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def _generate_weights(
        self,
        interpolation_mode: Union[str, Path],
        num_layers_looped: int,
        num_loops: int,
        device: torch.device,
    ):
        """
        Generate weights for interpolation based on the interpolation mode.
        Out of size: (num_layers_looped, num_loops, num_loops)
        Can trigger broadcast if returns num_layers_looped == 1.
        """

        if interpolation_mode == "no":
            w = torch.eye(num_loops, device=device)  # (num_loops, num_loops)
        elif interpolation_mode == "baseline":
            w = torch.zeros((num_loops, num_loops), device=device)
            for i in range(num_loops):
                w[i, 0] = 1.0  # first loop iteration gets all weight (ignore others)
        elif interpolation_mode == "uniform":  # previously "accu"
            w = torch.ones((num_loops, num_loops), device=device) / (torch.arange(num_loops, device=device) + 1).unsqueeze(1)
        elif interpolation_mode == "ema":
            ma_coeff = 0.5
            w = torch.eye(num_loops, device=device) * ma_coeff
            for i in range(num_loops):
                w[i, 0] = w[i, 0] + (1 - ma_coeff)  # first loop iteration gets the rest of the weight
        elif os.path.exists(interpolation_mode):  # load weights directly from a file
            print(f"Loading weights from {interpolation_mode}")
            w = torch.load(interpolation_mode, map_location=device)
            return w
        else:
            raise NotImplementedError(
                f"Interpolation mode '{interpolation_mode}' is not implemented. Supported: ['no', 'baseline', 'uniform', 'ema']",
            )

        assert w.shape == (num_loops, num_loops), f"Expected weights shape to be (num_loops, num_loops), got {w.shape}"
        w = torch.tril(w)  # lower triangular matrix
        assert torch.isclose(
            w.sum(dim=-1),
            torch.ones(num_loops, device=device),
        ).all(), "Weights must sum to 1 along the last dimension."

        w = w.unsqueeze(0).expand(num_layers_looped, -1, -1)  # (num_layers_looped, num_loops, num_loops)
        return w

    def unified_cached_interpolation(
        self,
        x: torch.Tensor,  # B, L, D
        loop_iter: Optional[int] = None,
        loop_layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        # not in a loop: return x
        if any(v == -1 or v is None for v in (loop_iter, loop_layer_idx)):
            return x

        if self.cache is not None and self.cache.shape[-3] != x.shape[0]:
            self.cache = None

        # manage cache
        if self.cache is None and self.loop_plan is not None:
            n_layers_looped, n_loops = self.loop_plan
            B, L, D = x.shape
            self.cache = torch.zeros((n_layers_looped, n_loops, B, L, D), device=x.device, dtype=x.dtype)

        self.cache[loop_layer_idx, loop_iter, :, :, :] = x  # (B, L, D)

        # Interpolate
        if self.interpolation_mode == "auto-alignment":
            t = self.temperature if hasattr(self, "temperature") else DEFAULT_TEMPERATURE
            # (B, L, D)
            return auto_alignment_interpolation(self.cache[loop_layer_idx, : loop_iter + 1, :, :, :], temperature=t)

        assert self.cache_weights is not None, "Cache weights must be initialized before interpolation."

        self.cache_weights.to(x.device, x.dtype)  # ensure cache weights are on the same device and dtype as x

        # get weights for the current layer and loop iteration
        W = self.cache_weights[loop_layer_idx, loop_iter, : loop_iter + 1]  # (loop_iter + 1,)
        W = torch.softmax(W, dim=0)  # (loop_iter + 1,)

        # slice the cache

        c = self.cache[loop_layer_idx, : loop_iter + 1, :, :, :]  # (loop_iter + 1, B, L, D)

        # compute the weighted sum
        out = torch.einsum("s,sbld->bld", W, c)

        # FIXME: debug
        og_state = self.cache[loop_layer_idx, 0, :, :, :].clone()  # original state

        delta = out - og_state
        noise = torch.randn_like(out)
        noise = noise / noise.norm(dim=-1, keepdim=True)  # normalize
        noised_delta = noise * delta.norm(dim=-1, keepdim=True)
        out = og_state + noised_delta

        return out


def auto_alignment_interpolation(cache: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    cache of size (N, B, L, D)
    """

    # we only care about the last hidden state of the sequence
    subcache = cache[:, :, -1, :]  # (N, B, D)
    subcache = subcache / subcache.norm(dim=-1, keepdim=True)

    ref = subcache[0].expand(subcache.shape[0], -1, -1)  # (N, B, D)
    scores = torch.einsum("nbd,nbd->nb", ref, subcache)  # (N, B)
    scores = torch.softmax(scores / temperature, dim=0)  # (N, B)
    out = torch.einsum("nb,nbld->bld", scores, cache)  # (B, L, D)

    return out


# Get model and tokenizer
def load_hacked_model(
    model_id: str = MODEL_ID,
    interpolation_mode: str = "no",
    plan: Optional[list[tuple[int, int, int]]] = None,
    move_to_cuda: bool = True,
) -> tuple[AutoTokenizer, LlamaForCausalLM]:
    if plan is not None and plan != AUTO_PLAN:
        print("Loading Hacked Llama Model...")
        print("Using execution plan:", plan)
    elif plan is None:
        print("Loading Vanilla Llama Model...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = LlamaForCausalLM.from_pretrained(model_id, cache_dir=CACHE_DIR)
    if plan is not None:
        if plan == AUTO_PLAN:
            plan = (10, 15, 4)

        model.model._original_config = deepcopy(model.model.config)
        model._original_config = deepcopy(model.config)

        model.model = HackedLlamaModel.from_pretrained(model_id, cache_dir=CACHE_DIR)

        model.model.interpolation_mode = interpolation_mode
        model.model.build_exec_plan(plan)
        model.config.num_hidden_layers = model.model.config.num_hidden_layers

        print(f"Using {interpolation_mode = } interpolation mode.")

        model._no_split_modules = ["LlamaDecoderLayer"]

    if move_to_cuda:
        model.to("cuda")

    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_hacked_model(plan=(10, 16, 4), interpolation_mode="uniform")

    print(model._tied_weights_keys)

    print(f"Model n_params: {model.num_parameters()}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # check tied

    diff = model.lm_head.weight - model.model.embed_tokens.weight
    print(f"Weight difference: {diff.abs().max()}")

    print(f"Model embed_tokens ptr: {model.model.embed_tokens.weight.data_ptr()}")
    print(f"Model lm_head ptr: {model.lm_head.weight.data_ptr()}")

    print(model.model)

    print(model.model.config._attn_implementation)
