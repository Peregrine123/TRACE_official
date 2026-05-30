import json
import os
from datasets import DatasetDict, load_dataset
from accelerate import Accelerator
from datasets import Dataset , concatenate_datasets , load_from_disk
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from transformers import GenerationConfig
import gc
from data.agent_safety.env import compute_score
from data.agent_safety.env import SafetyJudgeEnv
from data.base_builder import BaseBuilder
import logging
from memgen.utils import (
    log_trainable_params, 
    remove_trainer_checkpoints, 
    open_model_parameters,
    fix_model_parameters,
    create_tensorboard,
    StaticEvalRecorder,
    )
from safety.utils import build_semantic_gate 

class SafetyRunnerMERGE:
    def __init__(
        self,
        model,
        dataset, 
        config: dict,
        working_dir: str,
    ):
        self.config = config
        self.working_dir = working_dir
        self._parse_configs(config.get("run", {}))

        self.processing_class = model.tokenizer
        self.model = model
        self.train_dataset = self._filter_dataset(load_dataset("json", data_files='data/agent_safety/merge_train.jsonl', split="train"))
        self.valid_dataset = self._filter_dataset(load_dataset("json", data_files='data/agent_safety/merge_valid.jsonl', split="train"))
        if dataset == 'rjg':
            self.test_dataset = load_dataset("json", data_files='data/agent_safety/rjg_test.jsonl', split="train")
        elif dataset == 'ASSE':
            self.test_dataset = load_dataset("json", data_files='data/agent_safety/ASSE_test.jsonl', split="train")
        else:
            self.test_dataset = load_dataset("json", data_files='data/agent_safety/ASSE_GPT_test.jsonl', split="train")
        # self.all_dataset = concatenate_datasets([self.train_dataset, self.valid_dataset, self.test_dataset])
        # self.env_cls = data_builder.get_env_cls()
        self.env = SafetyJudgeEnv(config.get("dataset"))
        self.semantic_gate = build_semantic_gate(self.semantic_gate_cfg)

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        tokenizer = self.processing_class
        max_len = self.weaver_sft_training_args.max_length

        def filter_func(sample):
            if "prompt" in sample and sample["prompt"] is not None:
                encoded = tokenizer(sample["prompt"], add_special_tokens=True)
                return len(encoded["input_ids"]) < max_len
            return True

        return dataset.filter(filter_func)

    def _create_weaver_trainer(self):
        return SFTTrainer(
            model=self.model,
            args=self.weaver_sft_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            processing_class=self.processing_class,
        )

    def train(self,
              accelerator: Accelerator = None):
        # self.model.fix_component("trigger")
        if self.model.sft:
            for param in self.model.reasoner.parameters():
                param.requires_grad = True
        # self.model.open_component("weaver")
        if accelerator.is_main_process:
            log_trainable_params(self.model)

        weaver_trainer = self._create_weaver_trainer()
        # print_trainable_modules(weaver_trainer.model)
        # trainable = [(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        # print("trainable params:", len(trainable))
        # print("examples:", [n for n,_ in trainable[:50]])
        # exit(-1)
        weaver_trainer.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if weaver_trainer.is_world_process_zero():
            weaver_trainer.save_model()

        gc.collect()

        output_dir = weaver_trainer.args.output_dir
        remove_trainer_checkpoints(output_dir)

    def _pad_right_to_len(self, x: torch.LongTensor, target_len: int, pad_id: int) -> torch.LongTensor:
        """Right-pad [bs, L] -> [bs, target_len]."""
        if x.size(1) == target_len:
            return x
        if x.size(1) > target_len:
            return x[:, :target_len]
        pad = x.new_full((x.size(0), target_len - x.size(1)), pad_id)
        return torch.cat([x, pad], dim=1)


    def evaluate(self, 
                 **kwargs):
        accelerator = Accelerator()
        if not accelerator.is_main_process:
            return        
        # ---- model setup ----
        self.model = self.model.to(torch.bfloat16)
        self.model.fix_component("weaver")
        self.model.fix_component("reasoner")
        output_dir = os.path.join(self.working_dir, "evaluate")
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "answer.jsonl")
        wrong_path = os.path.join(output_dir, "wrong_answers.jsonl")
        generation_config = GenerationConfig(
            do_sample=self.eval_do_sample,
            temperature=self.eval_temperature,
            max_new_tokens=self.eval_max_new_tokens,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            use_cache=False,
        )

        test_dataloader = accelerator.prepare(
            DataLoader(
                dataset=self.test_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=lambda batch: batch,
            )
        )

        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()
        unwrapped_model = accelerator.unwrap_model(model_wrapped)
        pad_id = self.processing_class.pad_token_id

        total = 0
        correct = 0
        triggered = 0
        with open(log_path, "w", encoding="utf-8") as f, open(wrong_path, "w", encoding="utf-8") as f_wrong:
            progress = tqdm(test_dataloader, disable=not accelerator.is_main_process)
            for test_batch in progress:
                # -----------------------------
                # 1) batch tokenize prompts
                # -----------------------------
                prompts = [s["prompt"] for s in test_batch]
                prompt_inputs = self.processing_class(
                    text=prompts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=True,
                )
                input_ids = prompt_inputs["input_ids"].to(accelerator.device)          # [B, Lp]
                attention_mask = prompt_inputs["attention_mask"].to(accelerator.device)

                B = input_ids.size(0)
                prompt_len = input_ids.size(1)

                # -----------------------------
                # 2) first gate decision (per sample)
                #    gate false -> pure reasoner (no gate_fn, no mem injection)
                #    gate true  -> define gate_fn, call generate_with_gate(use_weaver=True)
                # -----------------------------
                user_questions = [s.get("User question", "") for s in test_batch]
                turns_list = [s.get("Turns", []) for s in test_batch]
                gate_signals = [self.semantic_gate(uq, t) for uq, t in zip(user_questions, turns_list)]
                use_weaver_mask = torch.tensor(
                    [bool(gs.triggered) for gs in gate_signals],
                    device=accelerator.device,
                    dtype=torch.bool,
                )

                idx_weaver = use_weaver_mask.nonzero(as_tuple=True)[0]
                idx_plain = (~use_weaver_mask).nonzero(as_tuple=True)[0]

                out_plain = None
                out_weaver = None
                # -----------------------------
                # 3) plain path: pure reasoner baseline
                # -----------------------------
                if idx_plain.numel() > 0:
                    with torch.no_grad():
                        out_plain = unwrapped_model.reasoner.generate(
                            input_ids=input_ids[idx_plain],
                            attention_mask=attention_mask[idx_plain],
                            generation_config=generation_config,
                        )
                    # uq_sub = [user_questions[i] for i in idx_plain.tolist()]
                    # turns_sub = [turns_list[i] for i in idx_plain.tolist()]

                    # # gate_fn expects a list[str] and returns list[bool]
                    # def gate_fn(inference_texts):
                    #     out = []
                    #     for uq, turns, inf in zip(uq_sub, turns_sub, list(inference_texts)):
                    #         if inf:
                    #             augmented_turns = list(turns) + [{"inference": inf}]
                    #             out.append(bool(self.semantic_gate(uq, augmented_turns).triggered))
                    #         else:
                    #             out.append(bool(self.semantic_gate(uq, turns).triggered))
                    #     return out

                    # with torch.no_grad():
                    #     out_weaver = unwrapped_model.generate(
                    #         input_ids=input_ids[idx_plain],
                    #         attention_mask=attention_mask[idx_plain],
                    #         generation_config=generation_config,
                    #         gate_fn=gate_fn,
                    #         plain = True,
                    #         **kwargs,
                    #     )
                # -----------------------------
                # 4) weaver path: only when first gate says triggered
                #    define gate_fn (batch) and call generate_with_gate(use_weaver=True)
                # -----------------------------
                if idx_weaver.numel() > 0:
                    # Build sub-batch lists aligned with idx_weaver order
                    uq_sub = [user_questions[i] for i in idx_weaver.tolist()]
                    turns_sub = [turns_list[i] for i in idx_weaver.tolist()]

                    # gate_fn expects a list[str] and returns list[bool]
                    def gate_fn(inference_texts):
                        out = []
                        for uq, turns, inf in zip(uq_sub, turns_sub, list(inference_texts)):
                            if inf:
                                augmented_turns = list(turns) + [{"inference": inf}]
                                out.append(bool(self.semantic_gate(uq, augmented_turns).triggered))
                            else:
                                out.append(bool(self.semantic_gate(uq, turns).triggered))
                        return out

                    with torch.no_grad():
                        out_weaver = unwrapped_model.generate(
                            input_ids=input_ids[idx_weaver],
                            attention_mask=attention_mask[idx_weaver],
                            generation_config=generation_config,
                            gate_fn=gate_fn,
                            **kwargs,
                        )
                # -----------------------------
                # 5) merge outputs back to [B, max_len] with padding
                # -----------------------------
                lens = []
                if out_plain is not None:
                    lens.append(out_plain.size(1))
                if out_weaver is not None:
                    lens.append(out_weaver.size(1))

                max_len = max(lens) if lens else prompt_len
                merged = input_ids.new_full((B, max_len), fill_value=pad_id)

                if out_plain is not None:
                    out_plain = self._pad_right_to_len(out_plain, max_len, pad_id)
                    merged[idx_plain] = out_plain

                if out_weaver is not None:
                    out_weaver = self._pad_right_to_len(out_weaver, max_len, pad_id)
                    merged[idx_weaver] = out_weaver

                # -----------------------------
                # 6) decode completion + score + write per sample
                # -----------------------------
                completion_ids = merged[:, prompt_len:]
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

                for i, sample in enumerate(test_batch):
                    completion = completions[i]
                    gt = sample.get("completion", "")

                    score = compute_score(completion, gt)
                    total += 1
                    correct += int(score > 0)

                    gs = gate_signals[i]
                    record = {
                        "prompt": sample["prompt"],
                        "completion": completion,
                        "ground_truth": gt,
                        "gate": {
                            "triggered": bool(gs.triggered),
                            "use_weaver_mask": bool(use_weaver_mask[i]),
                            "angle_deg": float(gs.angle.angle_deg) if hasattr(gs, "angle") else None,
                            "cos_sim": float(gs.angle.cos_sim) if hasattr(gs, "angle") else None,
                            # "entropy": float(gs.entropy) if hasattr(gs, "entropy") else None,
                            "adjacent": bool(gs.adjacent_triggered) if hasattr(gs, "adjacent_triggered") else None,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if bool(gs.triggered):
                        triggered += 1
                    if score == 0:
                        f_wrong.write(json.dumps(record, ensure_ascii=False) + "\n")
            accuracy = correct / max(total, 1)
            f.write(json.dumps({"final_accuracy": accuracy, "triggered": triggered, "total": total}, ensure_ascii=False) + "\n")

        print(f"Evaluation accuracy: {accuracy:.4f}")

    def _parse_configs(self, configs: dict):
        self.train_weaver = configs.get("train_weaver", True)
        self.train_weaver_method = configs.get("train_weaver_method", "sft")
        if self.train_weaver_method != "sft":
            raise ValueError("Safety runner supports SFT only.")

        weaver_config = configs.get("weaver", {})
        weaver_sft_config = weaver_config.get("sft", {})
        self.weaver_sft_training_args = SFTConfig(**weaver_sft_config)
        self.weaver_sft_training_args.gradient_checkpointing = False
        self.weaver_sft_training_args.prediction_loss_only = True
        self.weaver_sft_training_args.bf16_full_eval = True 
        self.weaver_sft_training_args.output_dir = os.path.join(self.working_dir, "weaver")

        interaction_cfg = configs.get("interaction", {})
        self.eval_do_sample = interaction_cfg.get("do_sample", False)
        self.eval_temperature = interaction_cfg.get("temperature", 1.0)
        self.eval_max_new_tokens = interaction_cfg.get("max_response_length", 512)
        self.eval_batch_size = interaction_cfg.get("batch_size", 4)

        self.semantic_gate_cfg = configs.get("semantic_gate", {})
