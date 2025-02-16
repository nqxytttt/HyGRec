# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
import numpy as np

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    lora_type: str = field(default='std', metadata={"help": "Type of LoRA adaptation"})

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
                             and not is_hf_device_map_available,
            "lora_type": self.peft_config.lora_type,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            lora_type: str = 'std',
            **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_type = lora_type.split('-')[0]
        if len(lora_type.split('-')) > 1:
            k = float(lora_type.split('-')[1])  # get the curvature
        else:
            k = -1.0  # default curvature

        # Actual trainable parameters
        if r > 0:
            if self.lora_type == 'std':
                # origianl lora
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.delta = []

            elif self.lora_type == 'hyperbolic':
                self.lora_A = nn.Linear(in_features + 1, r, bias=False)
                self.lora_B = nn.Linear(r + 1, out_features, bias=False)
                # self.k_linear = nn.Linear(in_features, 1, bias=True)
                self.k = nn.Parameter(torch.tensor(k), requires_grad=True)
                # self.k = nn.Parameter(torch.tensor(k), requires_grad=False)
                self.norm_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)

            elif self.lora_type == 'hyperbolic_rot':
                # C2: curvature is inferred from the input x and N1: the x norm is learnable
                self.lora_A = nn.Linear(in_features + 1, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                # self.k_linear = nn.Linear(in_features, 1, bias=True)
                self.k = nn.Parameter(torch.tensor(k), requires_grad=True)
                self.norm_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if hasattr(self, 'lora_A'):
            self.lora_A.train(mode)
            self.lora_B.train(mode)

        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                        transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                        transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if hasattr(self, "lora_A"):
            self.lora_A.eval()
            self.lora_B.eval()
        self.merged = False

    @staticmethod
    def lorentz_expmap0(u, k, dim=-1, min=1e-8):
        x = u.narrow(-1, 1, u.size(-1) - 1)
        sqrtK = torch.sqrt(torch.abs(k))
        x_norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min, max=math.asinh(2 ** 15))
        theta = x_norm / sqrtK

        l_v = sqrtK * torch.cosh(theta)
        r_v = sqrtK * torch.sinh(theta) * x / x_norm
        v = torch.cat((l_v, r_v), dim)
        return v
    @staticmethod
    def lorentz_logmap0(x, k, dim=-1, min=1e-7):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        sqrtK = torch.sqrt(torch.abs(k))
        y_norm = torch.norm(y, p=2, dim=dim, keepdim=True).clamp(min)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[..., 0:1] / sqrtK, min=1.0 + 1e-7)
        res[..., 1:] = sqrtK * torch.arccosh(theta) * y / y_norm
        return res

    def padding_zero(self, x, normalize=True):
        num_dims = len(x.size())
        padding = [0] * (2 * num_dims)
        padding[-2 * num_dims] = 1
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True) * self.norm_scale.exp().clamp(max=10)  # normalization
        else:
            x = x * self.norm_scale.exp().clamp(max=10)
        x = F.pad(x, padding, "constant", value=0)

        return x

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            if self.r > 0 and self.merged:
                matmul_output = self.lora_B.weight @ self.lora_A.weight
                self.weight.data -= transpose(matmul_output.to(previous_dtype), self.fan_in_fan_out) * self.scaling
                self.merged = False

            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                if self.lora_type == 'std':
                    result += self.lora_B(self.lora_A(self.lora_dropout(x.to(self.lora_A.weight.dtype)))) * self.scaling

                elif self.lora_type in ['hyperbolic']:
                    x = x.to(self.lora_A.weight.dtype)
                    x = self.lora_dropout(x)

                    # x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
                    # dynamic_k = 1.0 / (1.0 + torch.exp(x_norm))

                    x = self.padding_zero(x)
                    # (1) exponential map
                    x = self.lorentz_expmap0(x, self.k)
                    # x = self.lorentz_expmap0(x, dynamic_k)

                    # (2) Transformation on the manifold
                    x_space = self.lora_A(x)
                    x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + self.k.abs()).sqrt()  #
                    # x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + dynamic_k).sqrt()
                    x = torch.cat([x_time, x_space], dim=-1)  # cat the time value

                    x_space = self.lora_B(x)
                    x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + self.k.abs()).sqrt()
                    # x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + dynamic_k).sqrt()
                    x = torch.cat([x_time, x_space], dim=-1)  # cat the time value

                    # (3) Logarithmic map
                    x = self.lorentz_logmap0(x, self.k)[..., 1:]
                    # x = self.lorentz_logmap0(x, dynamic_k)[..., 1:]
                    x = x * self.scaling
                    result += x

                elif self.lora_type in ['hyperbolic_rot']:
                    x = x.to(self.lora_A.weight.dtype)
                    x = self.lora_dropout(x)

                    x = self.padding_zero(x)

                    # (1) exponential map
                    x = self.lorentz_expmap0(x, self.k)

                    # (2) Transformation on the manifold
                    x = self.lora_A(x)
                    x_space = self.lora_B(x)

                    x_time = ((x_space ** 2).sum(dim=-1,keepdim=True) + self.k.abs()).sqrt()
                    x = torch.cat([x_time, x_space], dim=-1)  # cat the time value

                    # (3) Logarithmic map
                    x = self.lorentz_logmap0(x, self.k)[...,1:]
                    x = x * self.scaling
                    result += x

                else:
                    raise NotImplementedError
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        if out_features % len(enable_lora) != 0:
            raise ValueError("The length of enable_lora must divide out_features")
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
            self.lora_B = nn.Conv1d(
                r * sum(enable_lora),
                out_features // len(enable_lora) * sum(enable_lora),
                kernel_size=1,
                groups=2,
                bias=False,
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data += transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data -= transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.r > 0 and self.merged and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                delta_w = delta_w.to(self.weight.dtype)
                self.weight.data -= transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
                self.merged = False
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                after_A = self.lora_A(self.lora_dropout(x.to(self.lora_A.weight.dtype)))
                after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
                self,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            # Actual trainable parameters
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                    result += output
                else:
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                    result += output
            return result


    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
                self,
                in_features: int,
                out_features: int,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                enable_lora: List[bool] = [False],
                **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            if out_features % len(enable_lora) != 0:
                raise ValueError("The length of enable_lora must divide out_features")
            self.enable_lora = enable_lora
            # Actual trainable parameters
            if r > 0 and any(enable_lora):
                self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
                self.lora_B = nn.Conv1d(
                    r * sum(enable_lora),
                    out_features // len(enable_lora) * sum(enable_lora),
                    kernel_size=1,
                    groups=2,
                    bias=False,
                )
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
                # Compute the indices
                self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
                self.lora_ind[enable_lora, :] = True
                self.lora_ind = self.lora_ind.view(-1)
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def zero_pad(self, x):
            result = x.new_zeros((*x.shape[:-1], self.out_features))
            result = result.view(-1, self.out_features)
            result[:, self.lora_ind] = x.reshape(
                -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
            )
            return result.view((*x.shape[:-1], self.out_features))

        def forward(self, x: torch.Tensor):
            result = super().forward(x)
            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype
                    if x.dtype != torch.float32:
                        x = x.float()
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
                    result += output
                else:
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B) * self.scaling
                    result += output
            return result
