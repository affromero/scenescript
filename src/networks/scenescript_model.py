# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from enum import IntEnum
from typing import Any

import torch
import torchsparse
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.nn import modules as nn
from torchsparse.utils.collate import sparse_collate

from ..data.geometries import ALL_ENTITY_CLASSES, get_entity_class_from_token
from ..data.language_sequence import LanguageSequence, is_id_param
from ..data.point_cloud import PointCloud
from .decoder import HELPER_TOKEN, SceneScriptDecoder
from .encoder import PointCloudEncoder


def create_TYPE_TOKEN() -> IntEnum:
    values = ["PAD", "START", "STOP", "PART", "NOT_USED", "NOT_USED_1", "COMMAND"]
    for ENTITY_CLASS in ALL_ENTITY_CLASSES:
        for param_key in ENTITY_CLASS.PARAMS_DEFINITION:
            value: str = f"{ENTITY_CLASS.COMMAND_STRING}_{param_key}"
            values.append(value.upper())
    values.append("NUM")
    return IntEnum("TYPE_TOKEN", values, start=0)


def list_rindex(_list: list[Any], value: Any) -> int:
    _list.reverse()
    i = _list.index(value)
    _list.reverse()
    return len(_list) - i - 1


class SceneScriptWrapper(nn.Module):
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_num_tokens = cfg.model.decoder.max_num_tokens
        self.type_token = create_TYPE_TOKEN()

        self.encoder = PointCloudEncoder(
            input_channels=cfg.model.encoder.input_channels,
            d_model=cfg.model.encoder.d_model,
            conv_layers=cfg.model.encoder.conv_layers,
            num_bins=cfg.model.encoder.num_bins,
        )

        self.decoder = SceneScriptDecoder(
            d_model=cfg.model.decoder.d_model,
            num_attn_heads=cfg.model.decoder.num_attn_heads,
            dim_feedforward=cfg.model.decoder.dim_feedforward,
            num_bins=cfg.model.decoder.num_bins,
            max_num_tokens=cfg.model.decoder.max_num_tokens,
            max_num_type_tokens=self.type_token.NUM,  # type: ignore[attr-defined]
            num_decoder_layers=cfg.model.decoder.num_decoder_layers,
        )

    @staticmethod
    def load_from_checkpoint(ckpt_path: str, is_train: bool=False) -> "SceneScriptWrapper":
        ckpt_dict = torch.load(ckpt_path)
        cfg = OmegaConf.create(ckpt_dict["cfg"])

        model_wrapper = SceneScriptWrapper(cfg)
        weights = ckpt_dict["model_state_dict"]
        encoder_weights = {k.replace("encoder.","",1): v for k, v in weights.items() if k.startswith("encoder.")}
        decoder_weights = {k.replace("decoder.","",1): v for k, v in weights.items() if k.startswith("decoder.")}
        model_wrapper.encoder.load_state_dict(encoder_weights)
        model_wrapper.decoder.load_state_dict(decoder_weights)
        if not is_train:
            model_wrapper.eval()

        return model_wrapper

    @property
    def device(self) -> torch.device:
        return next(iter(self.encoder.parameters())).device

    def cpu(self) -> "SceneScriptWrapper":
        self.encoder.cpu()
        self.decoder.cpu()
        return self

    def to(self, device: str | torch.device) -> "SceneScriptWrapper":
        self.encoder.to(device)
        self.decoder.to(device)
        return self

    def train(self) -> "SceneScriptWrapper":
        self.encoder.train()
        self.decoder.train()
        return self

    def eval(self) -> "SceneScriptWrapper":
        self.encoder.eval()
        self.decoder.eval()
        return self

    def cuda(self) -> "SceneScriptWrapper":
        self.encoder.cuda()
        self.decoder.cuda()
        return self

    def top_p(self, logits: torch.Tensor, thres: float) -> torch.Tensor:
        """Filter out logits for nucleus sampling.

        Args:
            logits: [B, num_bins + HELPER_TOKEN.NUM] torch.Tensor.
            thresh: float. 0 means argmax, 1 means random sampling.

        Returns:
            filtered_logits: [B, num_bins + HELPER_TOKEN.NUM] torch.Tensor.

        """
        # Sort the logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cum_probs >= thres

        # Include the bin that pushed cumulative probability above 1 - thresh
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        # Set filtered logits to -inf, effectively ignoring them
        sorted_logits[sorted_indices_to_remove] = float("-inf")

        # Scatter will put logits back in their places
        return sorted_logits.scatter(1, sorted_indices, sorted_logits)

    def type_decoding(self, seq_value: torch.Tensor, seq_type: torch.Tensor) -> torch.Tensor:
        """Decode the next type token.

        Args:
            seq_value: [B, t] torch.LongTensor.
            seq_type: [B, t-1] torch.LongTensor.

        Returns:
            [B, t] torch.LongTensor.

        """
        new_type = torch.zeros_like(seq_type[:, 0])  # [B]
        for b in range(seq_value.shape[0]):
            seq_value_b = seq_value[b]  # [t]
            seq_type_b = seq_type[b]  # [t - 1]

            try:
                # There's already a stop token
                if torch.any(seq_type_b == self.type_token.STOP):  # type: ignore[attr-defined]
                    new_type[b] = self.type_token.PAD  # type: ignore[attr-defined]

                # A PART token was predicted in sequence
                elif seq_value_b[-1] == HELPER_TOKEN.PART:
                    new_type[b] = self.type_token.PART  # type: ignore[attr-defined]

                # A STOP token was predicted in sequence
                elif seq_value_b[-1] == HELPER_TOKEN.STOP:
                    new_type[b] = self.type_token.STOP  # type: ignore[attr-defined]

                # Previously, a COMMAND token was predicted
                elif seq_type_b[-1] == self.type_token.PART:  # type: ignore[attr-defined]
                    new_type[b] = self.type_token.COMMAND  # type: ignore[attr-defined]

                # We are somewhere in the middle of an argument sequence
                else:
                    latest_command_token_idx = list_rindex(
                        seq_type_b.tolist(), self.type_token.COMMAND,  # type: ignore[attr-defined]
                    )
                    command_value = int(
                        seq_value_b[latest_command_token_idx] - HELPER_TOKEN.NUM,
                    )
                    ENTITY_CLASS = get_entity_class_from_token(command_value)

                    type_token_ordering = (
                        [self.type_token.COMMAND]  # type: ignore[attr-defined]
                        + [
                            self.type_token[  # type: ignore[index]
                                f"{ENTITY_CLASS.COMMAND_STRING}_{param_key}".upper()
                            ]
                            for param_key in ENTITY_CLASS.PARAMS_DEFINITION
                            if not is_id_param(param_key)
                        ]
                    )  # e.g. [COMMAND, MAKE_WALL_A_X, MAKE_WALL_A_Y, ..., MAKE_WALL_HEIGHT]

                    token_order_idx = type_token_ordering.index(seq_type_b[-1])
                    new_type[b] = type_token_ordering[token_order_idx + 1]

            except:  # for any errors, just pad
                new_type[b] = self.type_token.PAD  # type: ignore[attr-defined]

        return torch.cat([seq_type, new_type.unsqueeze(-1)], dim=-1)

    def preprocess_point_cloud(self, _point_cloud: torch.Tensor) -> tuple[torchsparse.SparseTensor, torch.Tensor]:
        """Preprocess the point cloud to be fed into the encoder.

        Args:
            point_cloud: [N, 3] torch.FloatTensor.

        Returns:
            sparse_tensor: torchsparse.SparseTensor.

        """
        point_cloud = PointCloud(_point_cloud)

        # Push to positive quadrant
        extent = point_cloud.extent()
        pc_min = [extent["min_x"], extent["min_y"], extent["min_z"]]
        pc_min_torch = torch.as_tensor(pc_min)
        point_cloud.translate(-pc_min_torch)

        # Normalize / Discretize it
        point_cloud.normalize_and_discretize(
            self.cfg.data.num_bins, self.cfg.data.normalization_values,
        )

        # Convert to torchsparse.SparseTensor
        pc_sparse_tensor = torchsparse.SparseTensor(
            coords=point_cloud.coords.int(),
            feats=point_cloud.points.float(),
        )

        pc_sparse_tensor = sparse_collate([pc_sparse_tensor])  # batch_size = 1
        pc_sparse_tensor = pc_sparse_tensor.to(self.device)

        return pc_sparse_tensor, pc_min_torch

    def preprocess_language_grountruth(self, language_sequence: LanguageSequence, pc_min: torch.Tensor) -> torch.Tensor:
        """Preprocess the language sequence to be fed into the decoder.

        Args:
            language_sequence: LanguageSequence instance.
            pc_min: [3] torch.FloatTensor.

        Returns:
            gt_tokens: [max_tokens] torch.LongTensor.

        """
        language_sequence.translate(-pc_min)
        language_sequence.normalize_and_discretize(
            self.cfg.data.num_bins, self.cfg.data.normalization_values,
        )
        # TODO: add tokenization
        return language_sequence.to_seq_value(self.max_num_tokens)


    def postprocess_language(self, seq_value: torch.Tensor, pc_min: torch.Tensor, ids_dict: dict[str, int] | None = None) -> LanguageSequence:
        """Postprocess the language sequence back into the original frame of reference.

        Args:
            seq_value: [T] torch.LongTensor.
            pc_min: [3] torch.FloatTensor.

        """
        language_sequence = LanguageSequence.from_seq_value(seq_value, ids_dict=ids_dict)
        language_sequence.undiscretize_and_unnormalize(
            self.cfg.data.num_bins, self.cfg.data.normalization_values,
        )
        language_sequence.translate(pc_min)

        return language_sequence

    def forward(self, *args: Any, **kwargs: Any) -> LanguageSequence:
        return self.run_inference(*args, **kwargs)

    @torch.no_grad()
    def run_inference(
        self,
        raw_point_cloud: torch.Tensor,
        nucleus_sampling_thresh: float = 0.05,
        verbose: bool=False,
        ids_dict: dict[str, int] | None = None,
    ) -> LanguageSequence:
        """Run the full inference loop.

        Args:
            raw_point_cloud: [N, 3] torch.FloatTensor.
            nucleus_sampling_thresh: float. In [0, 1]. 0 means argmax, 1 means random sampling.
            verbose: bool.

        Returns:
            a LanguageSequence instance.

        """
        start_time = time.time()

        # Encode the visual inputs
        pc_sparse_tensor, pc_min = self.preprocess_point_cloud(raw_point_cloud)
        encoded_visual_input = self.encoder(pc_sparse_tensor)
        context = encoded_visual_input["context"]
        context_mask = encoded_visual_input["context_mask"]

        if verbose:
            print(f"Time taken for input encoding: {time.time() - start_time:.3f}s")
            start_time = time.time()  # reset timer

        B = context.shape[0]
        device = self.device

        seq_value = (
            torch.ones((B, 1), dtype=torch.long, device=device) * HELPER_TOKEN.START
        )
        seq_type = (
            torch.ones((B, 1), dtype=torch.long, device=device) * self.type_token.START  # type: ignore[attr-defined]
        )

        for _ in range(seq_value.shape[1], self.max_num_tokens):
            # Run decoder to get logits
            logits = self.decoder(
                context=context,
                context_mask=context_mask,
                seq_value=seq_value,
                seq_type=seq_type,
            )  # [B, T, num_bins + HELPER_TOKEN.NUM]
            logits_t = logits[:, -1]  # [B, num_bins + HELPER_TOKEN.NUM]
            logits_filtered = self.top_p(logits_t, nucleus_sampling_thresh)

            # Sample a token (across batch)
            probs = F.softmax(logits_filtered, dim=-1)  # [B, ...]
            tokens = torch.multinomial(probs, 1)  # [B, 1]

            # Append token
            seq_value = torch.cat([seq_value, tokens], dim=1)  # [B, t+1]

            # Decode type token
            seq_type = self.type_decoding(seq_value, seq_type)

            # Stop if the sequence has a STOP token
            if torch.sum(seq_value[0] == HELPER_TOKEN.STOP) >= 1:
                break

        if verbose:
            print(
                f"Time taken for autoregressive sampling: {time.time() - start_time:.3f}s"
            )

        seq_value = seq_value[0]  # un-batch-ify
        return self.postprocess_language(seq_value, pc_min, ids_dict=ids_dict)


    def run_train_step(
        self,
        raw_point_cloud: torch.Tensor,
        ground_truth_tokens: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        verbose: bool=False,
    ) -> list[LanguageSequence]:
        """Run the full inference loop.

        Args:
            raw_point_cloud: [N, 3] torch.FloatTensor.
            nucleus_sampling_thresh: float. In [0, 1]. 0 means argmax, 1 means random sampling.
            verbose: bool.

        Returns:
            a list of LanguageSequence instances, length of batch size.

        """
        self.train()
        optimizer.zero_grad()  # Clear gradients
        start_time = time.time()

        # Encode the visual inputs
        pc_sparse_tensor, pc_min = self.preprocess_point_cloud(raw_point_cloud)
        encoded_visual_input = self.encoder(pc_sparse_tensor)
        context = encoded_visual_input["context"]
        context_mask = encoded_visual_input["context_mask"]

        if verbose:
            print(f"Time taken for input encoding: {time.time() - start_time:.3f}s")
            start_time = time.time()  # reset timer

        B = context.shape[0]
        device = self.device

        seq_value = (
            torch.ones((B, ground_truth_tokens.size(1)), dtype=torch.long, device=device) * HELPER_TOKEN.START
        )
        seq_type = (
            torch.ones((B, ground_truth_tokens.size(1)), dtype=torch.long, device=device) * self.type_token.START  # type: ignore[attr-defined]
        )
        loss = torch.tensor(0.0, device=device)

        logits = self.decoder(
            context=context,
            context_mask=context_mask,
            seq_value=seq_value,
            seq_type=seq_type,
        )# [B, T, num_bins + HELPER_TOKEN.NUM]
        logits = logits[:, 1:]
        labels = ground_truth_tokens[:, 1:] # [B, T]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backpropagate the loss
        loss.backward()

        # Update the model weights
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            print(
                f"Time taken for autoregressive sampling: {time.time() - start_time:.3f}s"
            )

        language_sequence = []
        for _seq_value in seq_value:
            language_sequence.append(self.postprocess_language(_seq_value, pc_min))

        return language_sequence
