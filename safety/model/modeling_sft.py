import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model


def _freeze_all_params(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def _infer_torch_dtype(dtype_str: Optional[str]):
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp16", "float16", "half"]:
        return torch.float16
    if s in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype_str}")


def _build_lora_config(cfg: Dict[str, Any]) -> LoraConfig:
    """
    cfg 
      r, lora_alpha, lora_dropout, target_modules, bias, modules_to_save, use_rslora, init_lora_weights
    """
    task_type = cfg.get("task_type", TaskType.CAUSAL_LM)
    if isinstance(task_type, str):
        task_type = getattr(TaskType, task_type, TaskType.CAUSAL_LM)

    return LoraConfig(
        task_type=task_type,
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=cfg.get("bias", "none"),
        modules_to_save=cfg.get("modules_to_save", None),
        inference_mode=cfg.get("inference_mode", False),
        use_rslora=cfg.get("use_rslora", False),
        init_lora_weights=cfg.get("init_lora_weights", True),
    )


class LoraSFTConfig(PretrainedConfig):
    model_type = "lora_sft_wrapper"

    def __init__(
        self,
        base_model_name: str = "",
        lora_config: Optional[Dict[str, Any]] = None,
        adapter_name: str = "sft",
        torch_dtype: str = "bfloat16",
        attn_implementation: Optional[str] = "flash_attention_2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.lora_config = lora_config
        self.adapter_name = adapter_name
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation


class LoraSFTModel(PreTrainedModel):
    """
     LoRA SFT wrapper：
    - self.model: base CausalLM + PEFT LoRA adapters
    - forward: adapter_name sft
    """
    config_class = LoraSFTConfig

    def __init__(
        self,
        config: LoraSFTConfig,
        base_model: PreTrainedModel,
        base_tokenizer,
        use_all_parameter: bool,
    ):
        super().__init__(config)
        self.tokenizer = base_tokenizer
        self.use_all_parameter = use_all_parameter

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            logging.info("Tokenizer has no pad token. Using EOS as pad token.")

        if use_all_parameter:
            self.model = base_model
            logging.info("Using all model parameters for training.")
        else:
            _freeze_all_params(base_model)
            sft_lora_cfg = _build_lora_config(config.lora_config)
            self.model = get_peft_model(base_model, sft_lora_cfg, adapter_name=config.adapter_name)
            self.model.set_adapter(config.adapter_name)
            logging.info("Using LoRA adapters for SFT training.")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        训练 SFT：默认 adapter = config.adapter_name
        """
        if not self.use_all_parameter:
            adapter = self.config.adapter_name
            self.model.set_adapter(adapter)

        # 直接让底层 CausalLM 计算 loss（labels 非空时）
        out: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return out

    def freeze(self) -> None:
        _freeze_all_params(self.model)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        - disable_lora: 纯 base 推理（完全不走 LoRA）
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if generation_config is None:
            generation_config = GenerationConfig(
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if self.use_all_parameter:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **kwargs,)
        else:
            with self.model.disable_adapter():
                return self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    **kwargs,
                )

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "LoraSFTModel":

        model_name = config_dict.get("model_name")
        lora_cfg = config_dict.get("lora_config", {})
        adapter_name = config_dict.get("adapter_name", "sft")
        torch_dtype = _infer_torch_dtype(config_dict.get("torch_dtype", "bfloat16"))
        attn_impl = config_dict.get("attn_implementation", "flash_attention_2")
        use_all_parameter = config_dict.get("use_all_parameter", False)
        wrapper_cfg = LoraSFTConfig(
            base_model_name=model_name,
            lora_config=lora_cfg,
            adapter_name=adapter_name,
            torch_dtype=config_dict.get("torch_dtype", "bfloat16"),
            attn_implementation=attn_impl,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_model_path = config_dict.get("load_model_path", None)
        if not load_model_path:
            model = cls(
                config=wrapper_cfg, 
                base_model=base_model, 
                base_tokenizer=tokenizer,
                use_all_parameter=use_all_parameter,
            )
        else:
            model = cls.from_pretrained(
                load_model_path, 
                config=wrapper_cfg,
                base_model=base_model,
                base_tokenizer=tokenizer,
                use_all_parameter=use_all_parameter,
            )
        return model
