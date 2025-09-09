# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/smolvlm.py
"""Inference-only SmolVLM model compatible with HuggingFace weights."""

from typing import Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.idefics3 import Idefics3ForConditionalGeneration


class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):
    """SmolVLM model that reuses Idefics3 implementation.

    SmolVLM is built on the same architecture as Idefics3, so we can
    inherit from Idefics3ForConditionalGeneration and just customize
    any SmolVLM-specific behavior if needed.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )


EntryClass = SmolVLMForConditionalGeneration
