from datasets import DatasetDict, load_dataset
from typing import Dict
import os
from data.base_builder import BaseBuilder
from data.agent_safety.env import SafetyJudgeEnv

class SafetyJudgeBuilder(BaseBuilder):   # Env
    
    def get_env_cls(self):
        return SafetyJudgeEnv

    def _build_datasets(self) -> DatasetDict:
        data_files = os.path.join("data", self.dataset_name, self.file_name)
        raw_dataset = load_dataset("json", data_files=data_files, split="train")
        # if self.description:
        #     def add_description(example: Dict) -> Dict:
        #         example['completion'] = example['risk_description'] + "\n" + 'Answer:' + example['completion']
        #         return example
        #     raw_dataset = raw_dataset.map(add_description)
         
        val_size = int(len(raw_dataset) * self.config.get("val_ratio"))
        train_test_split = raw_dataset.train_test_split(test_size=val_size, shuffle=True)

        test_valid_split = train_test_split["test"].train_test_split(test_size=0.5, shuffle=True)
        dataset_dict = DatasetDict({
            'train': train_test_split["train"],
            'valid': test_valid_split['train'],
            'test': test_valid_split['test']
        })

        return dataset_dict
    
    def _build_sft_datasets(self) -> DatasetDict:
        return self._build_datasets()


    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()
