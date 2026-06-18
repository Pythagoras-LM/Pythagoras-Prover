from .configuration_llada21_moe import LLaDA2MoeConfig
from .modeling_llada21_moe import LLaDA2MoeModelLM

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

AutoConfig.register("llada21_moe", LLaDA2MoeConfig)
AutoModel.register(LLaDA2MoeConfig, LLaDA2MoeModelLM)
AutoModelForMaskedLM.register(LLaDA2MoeConfig, LLaDA2MoeModelLM)

__all__ = ["LLaDA2MoeConfig", "LLaDA2MoeModelLM"]
