import json
import os
from datasets import DatasetDict, load_dataset
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_from_disk
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
from accelerate.utils import gather_object
import time

class SafetyRunner:
    def __init__(
        self,
        model,
        data_builder: BaseBuilder,
        config: dict,
        working_dir: str,
    ):
        self.config = config
        self.working_dir = working_dir
        self._parse_configs(config.get("run", {}))

        self.processing_class = model.tokenizer
        self.model = model
        self.dataset_dict = data_builder.get_dataset_dict()
        self.train_dataset = self._filter_dataset(self.dataset_dict["train"])
        self.valid_dataset = self._filter_dataset(self.dataset_dict["valid"])
        self.test_dataset = self.dataset_dict["test"]
        # self.train_dataset.to_json("data/agent_safety/merged_for_tsne_train.jsonl", orient="records", lines=True, force_ascii=False)
        # self.test_dataset.to_json("data/agent_safety/merged_for_tsne_test.jsonl", orient="records", lines=True, force_ascii=False)
        self.env = SafetyJudgeEnv(config.get("dataset"))

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

    def train(self, accelerator: Accelerator = None):
        """训练模型。"""
        if self.model.sft:
            for param in self.model.reasoner.parameters():
                param.requires_grad = True

        if accelerator.is_main_process:
            log_trainable_params(self.model)

        weaver_trainer = self._create_weaver_trainer()
        weaver_trainer.train()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if weaver_trainer.is_world_process_zero():
            weaver_trainer.save_model()

        output_dir = weaver_trainer.args.output_dir
        remove_trainer_checkpoints(output_dir)

        gc.collect()

    def train_and_evaluate(self, accelerator: Accelerator = None, save_model: bool = False):
        """训练完成后直接使用内存中的模型进行评估，避免重新加载。
        
        Args:
            accelerator: Accelerator 实例
            save_model: 是否保存训练后的模型，默认为 False
        """
        # ============ 训练阶段 ============
        if self.model.sft:
            for param in self.model.reasoner.parameters():
                param.requires_grad = True

        if accelerator.is_main_process:
            log_trainable_params(self.model)

        weaver_trainer = self._create_weaver_trainer()
        weaver_trainer.train()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 只在需要时保存模型
        if save_model and weaver_trainer.is_world_process_zero():
            weaver_trainer.save_model()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ============ 评估阶段 ============
        if accelerator.is_main_process:
            logging.info("=" * 50)
            logging.info("Training completed. Starting evaluation...")
            logging.info("=" * 50)

        accelerator.wait_for_everyone()
        self.evaluate(accelerator=accelerator)
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

    def evaluate(self, accelerator: Accelerator = None, **kwargs):
        """多 GPU 并行评估模型（所有样本都走 weaver 路径）。"""

        accelerator = Accelerator()

        is_main = accelerator.is_main_process

        # ---- model setup ----
        self.model = self.model.to(torch.bfloat16)
        self.model.fix_component("weaver")
        self.model.fix_component("reasoner")

        output_dir = os.path.join(self.working_dir, "evaluate")
        if is_main:
            os.makedirs(output_dir, exist_ok=True)

        accelerator.wait_for_everyone()

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

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

        model_wrapped, test_dataloader = accelerator.prepare(self.model, test_dataloader)
        model_wrapped.eval()
        unwrapped_model = accelerator.unwrap_model(model_wrapped)
        pad_id = self.processing_class.pad_token_id

        # 每个进程的本地统计
        local_total = 0
        local_correct = 0
        local_records = []
        local_wrong_records = []

        progress = tqdm(
            test_dataloader,
            disable=not is_main,
            desc="Evaluating"
        )

        torch.cuda.synchronize()
        start = time.time()

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
            input_ids = prompt_inputs["input_ids"].to(accelerator.device)
            attention_mask = prompt_inputs["attention_mask"].to(accelerator.device)

            prompt_len = input_ids.size(1)

            # -----------------------------
            # 2) 所有样本都走 weaver 路径
            # -----------------------------
            with torch.no_grad():
                output_ids = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    **kwargs,
                )

            # -----------------------------
            # 3) decode completion + score
            # -----------------------------
            completion_ids = output_ids[:, prompt_len:]
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            for i, sample in enumerate(test_batch):
                completion = completions[i]
                gt = sample.get("completion", "")

                score = compute_score(completion, gt)
                local_total += 1
                local_correct += int(score > 0)

                record = {
                    "prompt": sample["prompt"],
                    "completion": completion,
                    "ground_truth": gt,
                }
                local_records.append(record)

                if score == 0:
                    local_wrong_records.append(record)

            if is_main:
                progress.set_postfix({
                    "acc": f"{local_correct / max(local_total, 1):.4f}"
                })

        # -----------------------------
        # 4) 收集所有进程的结果
        # -----------------------------
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print("elapsed:",elapsed)
        accelerator.wait_for_everyone()

        total_tensor = torch.tensor([local_total], device=accelerator.device)
        correct_tensor = torch.tensor([local_correct], device=accelerator.device)

        all_totals = accelerator.gather(total_tensor)
        all_corrects = accelerator.gather(correct_tensor)

        all_records = gather_object(local_records)
        all_wrong_records = gather_object(local_wrong_records)

        # -----------------------------
        # 5) 主进程写入结果
        # -----------------------------
        if is_main:
            total = all_totals.sum().item()
            correct = all_corrects.sum().item()
            accuracy = correct / max(total, 1)

            with open(log_path, "w", encoding="utf-8") as f:
                for record in all_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                summary = {
                    "final_accuracy": accuracy,
                    "correct": correct,
                    "total": total
                }
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

            with open(wrong_path, "w", encoding="utf-8") as f_wrong:
                for record in all_wrong_records:
                    f_wrong.write(json.dumps(record, ensure_ascii=False) + "\n")

            logging.info(f"Evaluation completed. Accuracy: {accuracy:.4f} ({correct}/{total})")
            print(f"Evaluation accuracy: {accuracy:.4f}")

        accelerator.wait_for_everyone()

        print()
        if is_main:
            return accuracy
        return None


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