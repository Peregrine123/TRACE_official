import argparse
from datetime import datetime
import os
import random
from accelerate import Accelerator
import numpy as np
import torch
from transformers.utils import logging as hf_logging
from common.config import Config
from common.logger import setup_logger
from data.agent_safety.builder import SafetyJudgeBuilder
from safety.model import MemGenModel
from safety.runner import SafetyRunner
from safety.runner_merge import SafetyRunnerMERGE
import logging


def set_seed(random_seed: int, use_gpu: bool):
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Safety MemGen")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    return parser.parse_args()


def build_working_dir(config: Config) -> str:
    mode = config.run_cfg.mode
    seed = config.run_cfg.seed
    model_name = config.model_cfg.model_name.split("/")[1]
    file, _ = os.path.splitext(config.dataset_cfg.file)
    parent_dir = os.path.join("results", mode, 'supply', model_name, file, f"seed{seed}")
    base_model_status = config.model_cfg.get("base_model_status", False)
    sft = config.model_cfg.get("sft", False)
    if sft:
        base_model_status = "sft"
    max_prompt_aug_num = config.model_cfg.max_prompt_aug_num
    prompt_latents_len = config.model_cfg.weaver.prompt_latents_len
    max_inference_aug_num = config.model_cfg.max_inference_aug_num
    inference_latents_len = config.model_cfg.weaver.inference_latents_len

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        working_dir = (
            f"{file}_bms_{base_model_status}_pn={max_prompt_aug_num}_pl={prompt_latents_len}_"
            f"in={max_inference_aug_num}_il={inference_latents_len}_{time}"
        )
    elif mode == "evaluate":
        angle = config.run_cfg.semantic_gate.angle_threshold_deg
        working_dir = (
            f"{file}_bms_{base_model_status}_pn={max_prompt_aug_num}_pl={prompt_latents_len}_"
            f"in={max_inference_aug_num}_il={inference_latents_len}_atg={angle}_{time}"
        )
        if config.run_cfg.interaction.get("angle_search", False):
            working_dir = os.path.join(working_dir, "angle_search")
    elif mode == "train_and_evaluate":
        working_dir = (
            f"{file}_bms_{base_model_status}_pn={max_prompt_aug_num}_pl={prompt_latents_len}_"
            f"in={max_inference_aug_num}_il={inference_latents_len}_{time}"
        )
    return os.path.join(parent_dir, working_dir)


def main():
    accelerator = Accelerator()

    # 只在主进程设置日志级别，避免重复输出
    if accelerator.is_main_process:
        hf_logging.set_verbosity_info()
        logging.getLogger().setLevel(logging.INFO)
    else:
        hf_logging.set_verbosity_error()
        logging.getLogger().setLevel(logging.ERROR)

    args = parse_args()
    config = Config(args)

    set_seed(config.run_cfg.seed, use_gpu=True)

    working_dir = build_working_dir(config)

    config.run_cfg.log_dir = os.path.join(working_dir, "logs")

    # 只在主进程设置 logger 和打印配置
    if accelerator.is_main_process:
        setup_logger(output_dir=config.run_cfg.log_dir)
        config.pretty_print()
        print(f"Set seed: {config.run_cfg.seed}")

    # 等待主进程完成日志设置
    accelerator.wait_for_everyone()

    config_dict = config.to_dict()
    data_builder = SafetyJudgeBuilder(config_dict.get("dataset"))
    model = MemGenModel.from_config(config_dict.get("model"))

    runner = SafetyRunner(
        model=model,
        data_builder=data_builder,
        config=config_dict,
        working_dir=working_dir,
    )

    mode = config.run_cfg.mode

    if mode == "train":
        runner.train(accelerator=accelerator)
    elif mode == "evaluate":
        torch.cuda.reset_peak_memory_stats()
        runner.evaluate(accelerator=accelerator)
    elif mode == "train_and_evaluate":
        runner.train_and_evaluate(accelerator=accelerator,save_model=False)


if __name__ == "__main__":
    main()