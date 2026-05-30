from dataclasses import dataclass
from typing import Optional, Literal

from peft import get_peft_model, LoraConfig, PeftModel
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from typing import Union
from safety.model.trigger import MemGenTrigger
from safety.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
    open_model_parameters
)

@dataclass
class MemGenOutputWithPast(CausalLMOutputWithPast):
    supervised_labels: Optional[torch.LongTensor] = None

class MemGenLoraSwitchMixin:

    # def _insert_lora_adapters(
    #     self, 
    #     weaver_model: PreTrainedModel, 
    #     weaver_lora_config: dict, 
    #     trigger_model: PreTrainedModel, 
    #     trigger_lora_config: dict
    # ) -> tuple[PeftModel, PeftModel]:
        
    #     same_model = weaver_model is trigger_model
    #     weaver_lora_config = LoraConfig(**weaver_lora_config)
    #     trigger_lora_config = LoraConfig(**trigger_lora_config)
        
    #     if same_model:
    #         peft_model = PeftModel(
    #             weaver_model, weaver_lora_config, adapter_name=MemGenWeaver.adapter_name
    #         )
    #         peft_model.add_adapter(peft_config=trigger_lora_config, adapter_name=MemGenTrigger.adapter_name)
    #         return peft_model, peft_model
        
    #     else:
    #         weaver_model_with_lora = PeftModel(
    #             weaver_model, weaver_lora_config, adapter_name=MemGenWeaver.adapter_name
    #         )
    #         trigger_model_with_lora = PeftModel(
    #             trigger_model, trigger_lora_config, adapter_name=MemGenTrigger.adapter_name
    #         )
    #         return weaver_model_with_lora, trigger_model_with_lora

    def _insert_lora_adapters(
        self, 
        model: PreTrainedModel, 
        lora_config: dict,
        adapter_name: str 
    ):
        _lora_config = LoraConfig(**lora_config)
        peft_model = get_peft_model(
            model, _lora_config, adapter_name=adapter_name
        )
        return peft_model
    
    def fix_component(self, name: Literal["weaver", "reasoner"]):

        component = getattr(self, name)
        fix_model_parameters(component)
        if name == "weaver":
            fix_model_parameters(self.weaver_to_reasoner)
            fix_model_parameters(self.reasoner_to_weaver)
    
    def open_component(self, name: Literal["weaver", "reasoner"]):

        component = getattr(self, name)  
        open_model_parameters(component)
        if name == "weaver":
            open_model_parameters(self.weaver_to_reasoner)
            open_model_parameters(self.reasoner_to_weaver)        
        
        fix_model_parameters(component.model.base_model)
        
        for p in component.model.named_parameters():
            if f"lora_A.{name}" in p[0] or f"lora_B.{name}" in p[0]:
                p[1].requires_grad = True

class SFTLoraSwitchMixin:

    def _insert_lora_adapters(
        self, 
        base_model: PreTrainedModel = None,
        base_lora_config: Optional[dict] = None,
    ) -> PeftModel:
        base_lora_config = LoraConfig(**base_lora_config)
        return get_peft_model(
            base_model, base_lora_config, adapter_name='reasoner'
        )

    def fix_component(self, name: Literal["reasoner"]):

        component = getattr(self, name)
        fix_model_parameters(component)

    def open_component(self, name: Literal["reasoner"]):

        component = getattr(self, name)  
        open_model_parameters(component)     
        fix_model_parameters(component.model.base_model)
        
        for p in component.model.named_parameters():
            if f"lora_A.{name}" in p[0] or f"lora_B.{name}" in p[0]:
                p[1].requires_grad = True

class MemGenGenerationMixin(GenerationMixin):

    def _get_next_token(
        self, 
        next_token_logits: torch.Tensor, 
        do_sample: bool, 
        temperature: Optional[float] = 0.0
    ) -> torch.Tensor:
        if len(next_token_logits.shape) != 2:
            raise ValueError("Input logits must be a 2D tensor [batch_size, vocab_size]")
        
        if do_sample and temperature != 0:  # Apply temperature scaling and sample from the resulting probability distribution    
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:  # Greedy decoding: pick the token with the highest probability
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    def _generate_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    def _is_conversation(self, input_ids: torch.Tensor, tokenizer) -> bool:
        if len(input_ids.shape) != 2:
            raise ValueError("input_ids must be a 2D tensor of shape (batch_size, seq_len)")
        
        seq = input_ids[0].tolist()

        # Encode the special tokens to obtain their ID sequences
        im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end_ids   = tokenizer.encode("<|im_end|>",   add_special_tokens=False)

        # Check if the sequence contains at least one <|im_start|> and one <|im_end|>
        has_start = any(seq[i:i+len(im_start_ids)] == im_start_ids for i in range(len(seq) - len(im_start_ids) + 1))
        has_end   = any(seq[i:i+len(im_end_ids)]   == im_end_ids   for i in range(len(seq) - len(im_end_ids) + 1))

        return has_start and has_end

    def _postprocess_assistant_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer
    ) -> torch.Tensor:
        if tokenizer.chat_template != CONVERSATION_TEMPLATE:
            raise ValueError(
                "Invalid tokenizer.chat_template detected.\n"
                f"Expected:\n{CONVERSATION_TEMPLATE}\n\n"
                f"Got:\n{tokenizer.chat_template}\n\n"
                "Please ensure that you are using the correct conversation template."
            )
        
        # Encode the token sequence for "<|im_start|>assistant\n"
        pattern_ids: list[int] = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        batch_size, seq_len = input_ids.shape
        new_labels = labels.clone()

        for b in range(batch_size):
            seq = input_ids[b].tolist()
            for i in range(len(seq) - len(pattern_ids) + 1):
                # Mask positions matching the pattern
                if seq[i : i + len(pattern_ids)] == pattern_ids:
                    new_labels[b, i : i + len(pattern_ids)] = -100

        return new_labels

    # def _check_ends_with_delimiter(
    #     self, input_ids: torch.Tensor, tokenizer, delimiters: list[str]
    # ) -> torch.Tensor:
    #     batch_size = input_ids.size(0)

    #     # Initialize result tensor: False by default
    #     augmentation_decisions = torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device)

    #     # Decode token IDs to text sequences
    #     decoded_inputs = tokenizer.batch_decode(input_ids)

    #     for i in range(batch_size):
    #         ends_with_augment_str = False
    #         # Check if the sequence ends with any of the given delimiters
    #         for aug_str in delimiters:
    #             if decoded_inputs[i].endswith(aug_str):
    #                 ends_with_augment_str = True
    #                 break
            
    #         augmentation_decisions[i] = ends_with_augment_str
        
    #     return augmentation_decisions
    
    # def _select_augment_points_after_delimiter(
    #     self,
    #     input_ids: torch.Tensor,
    #     labels: torch.Tensor,
    #     delimiters: list[str],
    #     tokenizer: PreTrainedTokenizerBase,
    #     max_num: int = 10,
    # ) -> list[int]:

    #     assert input_ids.shape == labels.shape
    #     B, seq_len = input_ids.size(0), input_ids.size(1)

    #     prompt_augment_idx = []
    #     inference_augment_idx = []

    #     for i in range(1, seq_len):  # Skip the first token and last token for augmentation
    #         # Detect the boundary between prompt and label for prompt augmentation
    #         if (labels[:, i] != -100).all() and (labels[:, i - 1] == -100).all():
    #             prompt_augment_idx.append(i)

    #         # Detect valid label regions for inference augmentation
    #         elif (labels[:, i] != -100).all() and (labels[:, i - 1] != -100).all():
    #             batch_tokens_before_i = input_ids[:, :i]
    #             # Assume check_ends_with_delimiter is defined
    #             if any(self._check_ends_with_delimiter(batch_tokens_before_i, tokenizer, delimiters)):
    #                 inference_augment_idx.append(i)
        
    #     # Ensure exactly one prompt augmentation point exists for single-turn processing
    #     if len(prompt_augment_idx) != 1:
    #         raise ValueError("Single-turn forward must have exactly one prompt augment index")

    #     final_points = prompt_augment_idx[:1]

    #     # Limit the number of inference augmentation points to max_num
    #     if len(inference_augment_idx) > max_num: 
    #         inference_augment_idx = inference_augment_idx[:max_num]

    #     final_points.extend(inference_augment_idx)
        
    #     if len(final_points) == 0:
    #         raise RuntimeError("No valid augmentation points found")
        
    #     final_points.sort()
    #     return final_points
    def _check_ends_with_delimiter(
        self, input_ids: torch.Tensor, tokenizer, delimiters: list[str]
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Initialize result tensor: False by default
        augmentation_decisions = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=input_ids.device
        )

        if seq_len == 0:
            return augmentation_decisions

        # --- Cache: delimiter token-id sequences (per tokenizer + delimiters) ---
        cache_key = (id(tokenizer), tuple(delimiters))
        if not hasattr(self, "_delimiter_id_cache"):
            self._delimiter_id_cache = {}
        if cache_key not in self._delimiter_id_cache:
            self._delimiter_id_cache[cache_key] = [
                tokenizer.encode(d, add_special_tokens=False) for d in delimiters
            ]
        delimiter_id_seqs = self._delimiter_id_cache[cache_key]

        # --- Cache: single-token decode strings (per tokenizer) ---
        if not hasattr(self, "_single_token_decode_cache"):
            self._single_token_decode_cache = {}
        if id(tokenizer) not in self._single_token_decode_cache:
            self._single_token_decode_cache[id(tokenizer)] = {}
        decode_cache = self._single_token_decode_cache[id(tokenizer)]

        # Work on a flat view [B]
        out = augmentation_decisions[:, 0]

        # 1) Fast path: token-id suffix match against delimiter encodings
        for ids in delimiter_id_seqs:
            L = len(ids)
            if L == 0 or seq_len < L:
                continue
            ids_t = torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)
            out |= (input_ids[:, seq_len - L : seq_len] == ids_t).all(dim=1)

        # 2) Fallback: decode ONLY the last token (covers merged tokens like "word,")
        if not out.all():
            last_tok_ids = input_ids[:, -1]
            # Keep delimiters tuple for faster any()
            delims = tuple(delimiters)

            # Only check examples that haven't matched
            for b in range(batch_size):
                if out[b]:
                    continue
                tok_id = int(last_tok_ids[b].item())
                s = decode_cache.get(tok_id)
                if s is None:
                    s = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
                    decode_cache[tok_id] = s
                if any(s.endswith(d) for d in delims):
                    out[b] = True

        return augmentation_decisions


    def _select_augment_points_after_delimiter(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        delimiters: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_num: int = 10,
    ) -> list[int]:

        assert input_ids.shape == labels.shape
        B, seq_len = input_ids.size(0), input_ids.size(1)

        prompt_augment_idx = []
        inference_augment_idx = []

        for i in range(1, seq_len):  # Skip the first token and last token for augmentation
            # Detect the boundary between prompt and label for prompt augmentation
            if (labels[:, i] != -100).all() and (labels[:, i - 1] == -100).all():
                prompt_augment_idx.append(i)

            # Detect valid label regions for inference augmentation
            elif (labels[:, i] != -100).all() and (labels[:, i - 1] != -100).all():
                batch_tokens_before_i = input_ids[:, :i]
                # Use the optimized delimiter check (token-id match + last-token decode fallback)
                if self._check_ends_with_delimiter(batch_tokens_before_i, tokenizer, delimiters).any().item():
                    inference_augment_idx.append(i)

        # Ensure exactly one prompt augmentation point exists for single-turn processing
        if len(prompt_augment_idx) != 1:
            raise ValueError("Single-turn forward must have exactly one prompt augment index")

        final_points = prompt_augment_idx[:1]

        # Limit the number of inference augmentation points to max_num
        if len(inference_augment_idx) > max_num:
            inference_augment_idx = inference_augment_idx[:max_num]

        final_points.extend(inference_augment_idx)

        if len(final_points) == 0:
            raise RuntimeError("No valid augmentation points found")

        final_points.sort()
        return final_points

    @torch.no_grad()
    def _should_augment(
        self,
        input_ids: torch.LongTensor,
        sentence_augment_count: torch.LongTensor,
        do_sample: bool,
        temperature: float,
        is_prompt: bool = False,
        gate_triggered: Optional[Union[bool, torch.BoolTensor]] = None,
    ) -> torch.LongTensor:
        tokenizer = self.tokenizer
        delimiters = self.delimiters
        # trigger = self.trigger
        max_augment_num = self.config.max_inference_aug_num

        B = input_ids.size(0)
        device = input_ids.device

        # --- gate branch: bypass trigger model ---
        if gate_triggered is not None:
            if isinstance(gate_triggered, bool):
                gate_mask = torch.full((B,), gate_triggered, dtype=torch.bool, device=device)
            else:
                gate_mask = gate_triggered.to(device=device, dtype=torch.bool)
                assert gate_mask.shape == (B,), f"gate_triggered must be shape (B,), got {gate_mask.shape}"

            if is_prompt:
                # prompt step: all samples are candidates; gate decides 0/1
                return gate_mask.long()

            # inference step: only delimiter-ending samples are candidates
            aug_vector = torch.full((B,), -100, dtype=torch.long, device=device)
            ends_with = self._check_ends_with_delimiter(input_ids, tokenizer, delimiters).squeeze(1)  # [B] bool

            # only when ends_with delimiter: write 0/1 according to gate
            aug_vector[ends_with] = gate_mask[ends_with].long()

            # over limit -> -100
            over_limit = (sentence_augment_count >= max_augment_num)
            aug_vector[over_limit] = -100
            return aug_vector
        else:
            if is_prompt:
                # prompt step: default augment for all
                return torch.ones((B,), dtype=torch.long, device=device)

            aug_vector = torch.full((B,), -100, dtype=torch.long, device=device)
            ends_with = self._check_ends_with_delimiter(input_ids, tokenizer, delimiters).squeeze(1)

            # delimiter-ending samples: default augment = 1
            aug_vector[ends_with] = 1

            # over limit -> -100
            over_limit = (sentence_augment_count >= max_augment_num)
            aug_vector[over_limit] = -100
            return aug_vector
        raise NotImplementedError("Gate branch is disabled.")
        # --- normal branch: use trigger model ---
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        position_ids = self._generate_position_ids(attention_mask)

        if is_prompt:
            aug_vector = torch.zeros((B,), dtype=torch.long, device=device)
            trigger_indices = torch.arange(B, device=device)
        else:
            aug_vector = torch.full((B,), -100, dtype=torch.long, device=device)
            ends_with = self._check_ends_with_delimiter(input_ids, tokenizer, delimiters).squeeze(1)
            aug_vector[ends_with] = 0
            over_limit = (sentence_augment_count >= max_augment_num)
            aug_vector[over_limit] = -100
            trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]

        # if trigger_indices.numel() > 0:
        #     trigger_logits = trigger(
        #         input_ids=input_ids[trigger_indices],
        #         attention_mask=attention_mask[trigger_indices],
        #         position_ids=position_ids[trigger_indices],
        #     )
        #     last_token_logits = trigger_logits[:, -1]  # [b', 2]
        #     next_tokens = self._get_next_token(last_token_logits, do_sample=do_sample, temperature=temperature).view(-1)
        #     aug_vector[trigger_indices] = next_tokens

        return aug_vector
    
    # @torch.no_grad()
    # def _should_augment(
    #     self, 
    #     input_ids: torch.LongTensor, 
    #     sentence_augment_count: torch.LongTensor, 
    #     do_sample: bool,
    #     temperature: float,
    #     is_prompt: bool = False,
    #     gate_triggered: Optional[bool] = None,
    # ) -> torch.LongTensor:
            
    #     tokenizer = self.tokenizer
    #     delimiters = self.delimiters
    #     trigger = self.trigger
    #     max_augment_num = self.config.max_inference_aug_num

    #     batch_size = input_ids.size(0)
        
    #     if gate_triggered is not None:
    #         if is_prompt:
    #             return torch.full(
    #                 (batch_size,),
    #                 1 if gate_triggered else 0,
    #                 dtype=torch.long,
    #                 device=input_ids.device,
    #             )
    #         attention_mask = (input_ids != tokenizer.pad_token_id).long()
    #         aug_vector = torch.full((batch_size,), -100, dtype=torch.long, device=input_ids.device)
    #         ends_with_delimiters = self._check_ends_with_delimiter(input_ids, tokenizer, delimiters).squeeze(1)
    #         aug_vector[ends_with_delimiters] = 1 if gate_triggered else 0
    #         over_limit = (sentence_augment_count >= max_augment_num)
    #         aug_vector[over_limit] = -100
    #         return aug_vector

    #     if is_prompt:  
    #         attention_mask = (input_ids != tokenizer.pad_token_id).long()
    #         position_ids = self._generate_position_ids(attention_mask)
    #         aug_vector = torch.zeros((batch_size,), dtype=torch.long, device=input_ids.device)
    #         trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]

    #     else:  
    #         attention_mask = (input_ids != tokenizer.pad_token_id).long()
    #         position_ids = self._generate_position_ids(attention_mask)
    #         aug_vector = torch.full((batch_size,), -100, dtype=torch.long, device=input_ids.device)
    #         ends_with_delimiters = self._check_ends_with_delimiter(input_ids, tokenizer, delimiters).squeeze(1)
    #         aug_vector[ends_with_delimiters] = 0
    #         over_limit = (sentence_augment_count >= max_augment_num)
    #         aug_vector[over_limit] = -100
    #         trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]

    #     if trigger_indices.numel() > 0:
    #         trigger_logits = trigger(
    #             input_ids=input_ids[trigger_indices],
    #             attention_mask=attention_mask[trigger_indices],
    #             position_ids=position_ids[trigger_indices]
    #         )
    #         last_token_logits = trigger_logits[:, -1]  # [batch, 2]

    #         next_tokens = self._get_next_token(
    #             last_token_logits,
    #             do_sample=do_sample,
    #             temperature=temperature
    #         ).view(-1)

    #         aug_vector[trigger_indices] = next_tokens

    #     return aug_vector


    @torch.no_grad()
    def _append_one_step(
        self,
        reasoner_outputs, 
        current_inputs_embeds: torch.Tensor,
        current_attention_mask: torch.Tensor,
        current_position_ids: torch.Tensor,
        current_input_ids: torch.Tensor,
        do_sample: bool,
        temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = current_inputs_embeds.size(0)
        
        # Append next token
        next_token_logits = reasoner_outputs.logits[:, -1]
        next_token_ids = self._get_next_token(next_token_logits, do_sample=do_sample, temperature=temperature)
        current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)
        
        # Append next token embeds
        next_token_embeds = self.reasoner.get_input_embeddings()(next_token_ids)
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embeds], dim=1)
        
        # Append attention mask
        attn_mask = torch.ones((B, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
        current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
        
        # Append position ids
        next_position_id = current_position_ids[:, -1:] + 1
        current_position_ids = torch.cat([current_position_ids, next_position_id], dim=1)

        return current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids
    

    @torch.no_grad()
    def _left_pad(
        self,
        input_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_num: int
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        
        if input_embeds is not None:
            B, L, D = input_embeds.shape
            pad_embeds = torch.zeros((B, pad_num, D), dtype=input_embeds.dtype, device=input_embeds.device)
            input_embeds = torch.cat([pad_embeds, input_embeds], dim=1)  # [B, pad_num + L, D]
        
        if attention_mask is not None:
            B = attention_mask.size(0)
            pad_mask = torch.zeros((B, pad_num), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)  # [B, pad_num + L]
        
        if position_ids is not None:
            B = position_ids.size(0)
            pad_pos = torch.zeros((B, pad_num), dtype=position_ids.dtype, device=position_ids.device)
            position_ids = torch.cat([pad_pos, position_ids], dim=1)  # [B, pad_num + L]

        return input_embeds, attention_mask, position_ids
    
    @torch.no_grad()
    def _left_clip_pad_tokens(
        self, inputs_embeds: torch.FloatTensor, attention_mask: torch.LongTensor, position_ids: torch.LongTensor
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:

        B, L, D = inputs_embeds.shape

        # Find the index of the first non-padding token in each sequence
        first_nonpad_idx = []
        for b in range(B):
            nonzero = (attention_mask[b] != 0).nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                # Entire row is padding; can potentially trim the whole sequence
                first_nonpad_idx.append(L)
            else:
                first_nonpad_idx.append(nonzero[0].item())
        
        # Determine the minimum number of left-padding tokens across the batch
        min_pad = min(first_nonpad_idx)

        # If no padding on the left, return original tensors
        if min_pad == 0:
            return inputs_embeds, attention_mask, position_ids

        # Trim the left-padding from all sequences in the batch
        inputs_embeds = inputs_embeds[:, min_pad:, :]
        attention_mask = attention_mask[:, min_pad:]
        position_ids = position_ids[:, min_pad:]

        return inputs_embeds, attention_mask, position_ids
    
    @torch.no_grad()
    def _check_generate(self, input_ids: torch.LongTensor, augmentation_pos: torch.LongTensor):
        """检查 augmentation_pos[b][i] == 1 的位置, input_ids[b][:i] (不包括第 i 位) 对应的字符串是否以 delimiters 结尾
        """
        delimiters = self.delimiters
        tokenizer = self.tokenizer

        B, L = input_ids.shape
        assert augmentation_pos.shape == input_ids.shape

        for b in range(B):
            for i in range(1, L):
                is_augment_point = augmentation_pos[b, i].item()
                
                if is_augment_point == -100:
                    continue

                if is_augment_point == 1 or is_augment_point == 0:
                    prefix_input_ids = input_ids[b, :i].unsqueeze(0)
                    
                    ends_with_delimiter = self._check_ends_with_delimiter(
                        prefix_input_ids, tokenizer, delimiters
                    ).item() 

                    if not ends_with_delimiter:
                        decoded_prefix = tokenizer.decode(prefix_input_ids.squeeze(0), skip_special_tokens=False)
                        
                        raise ValueError(
                            f"Augmentation position error at batch {b}, index {i}. "
                            f"augmentation_pos is 1, but the prefix does NOT end with a delimiter.\n"
                            f"Prefix: '...{decoded_prefix[-50:]}'\n"
                            f"Delimiters: {delimiters}"
                        )
                else:
                    raise ValueError(
                        f"Invalid value in augmentation_pos at batch {b}, index {i}: {is_augment_point}. "
                        "Expected 1, 0, or -100."
                    )
