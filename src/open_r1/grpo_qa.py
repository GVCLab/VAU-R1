# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video_QA as Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from src.open_r1.my_qwen_utils import process_vision_info
from tqdm import tqdm
import torch
import json
import random
import ast
import csv


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "answer"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    train_video_folder: str = field(
        default="/share/wy/Video/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    eval_video_folder: str = field(
        default="/home/zhuliyun/datasets/msad",
        metadata={"help": "Path to the folder containing evaluation video files."},
    )


def is_valid_two_d_list_format(s):
    pattern = r'^\[(\(\d+(\.\d+)?,\s*\d+(\.\d+)?\)(,\s*\(\d+(\.\d+)?,\s*\d+(\.\d+)?\))*(,)?|)\]$'
    if not re.match(pattern, s):
        return False
    try:
        # 尝试将字符串转换为 Python 对象
        lst = ast.literal_eval(s)
        # 检查对象是否为列表
        if not isinstance(lst, list):
            return False
        # 检查列表中的每个元素是否为元组
        for item in lst:
            if not isinstance(item, tuple):
                return False
            # 检查元组是否包含两个元素
            if len(item) != 2:
                return False
            # 检查元组中的元素是否为数字
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]: # 保证符合时序区间
                return False
        return True
    except:
        return False
        

def answer_reward(completions, solution, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""

    def extract_characters_regex(s):
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is",
            "The correct option is",
            "Best answer:" "Best option:",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
            return ""

        matches = re.search(r"[ABCDEFG]", s)
        if matches is None:
            return ""
        return matches[0]
    
    rewards = []

    for content, sol in zip(completions, solution): 
        reward = 0.0
        
        pattern_answer = r'<answer>(.*?)</answer>'

        # 使用 search 方法查找首个匹配项
        match_answer = re.search(pattern_answer, content, re.DOTALL)

        if match_answer:
            # 获取捕获组中的内容
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(sol['answer']):
                reward = 1.0

        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has <think> and <answer> correctly."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]

    reward_list = []
    for i, match in enumerate(matches):
        if match:
            r = 1.0
        else:
            r = 0.0
        reward_list.append(r)
    return reward_list


reward_funcs_registry = {
    "answer": answer_reward,
    "format": format_reward,
}

# SYSTEM_PROMPT = (
#     "You are an advanced anomaly detector assigned to analyze a video. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )

# SYSTEM_PROMPT = (
#     "You are an advanced anomaly detection assistant assigned to analyze videos through a structured conversation. "
#     "When the user asks a question, you should first carefully reason through the problem internally, and then present the final answer. "
#     "Your reasoning process must be enclosed within <think> </think> tags, and your final answer must be enclosed within <answer> </answer> tags. "
#     "The expected format is: <think> your detailed reasoning process here </think><answer> your final answer here </answer>. "
#     "Ensure that both the reasoning and the answer are clear, coherent, and well-structured."
# )


def load_csv_dataset(train_data_path, eval_data_path, train_video_folder, eval_video_folder):#, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
    def create_dataset_from_csv(file_path, split_name):
        if split_name == "train":
            video_folder = train_video_folder
        elif split_name == "eval":
            video_folder = eval_video_folder
        else:
            raise ValueError(f"Unknown split name: {split_name}")

        examples = []
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                # print(row)
                options = []
                options.extend(["A. " + row['Option 1'], "B. " + row['Option 2'], "C. " + row['Option 3'], "D. " + row['Option 4']])
                
                msad_video_folder = "/root/autodl-tmp/dataset/msad"
                ucf_video_folder = "/root/autodl-tmp/dataset/ucf-crime/all_videos"
                ecva_video_folder = "/root/autodl-tmp/dataset/ecva"

                original_name = row['Video Name']

                if 'msad' in original_name.lower():
                    video_folder = msad_video_folder
                elif 'ucf' in original_name.lower():
                    video_folder = ucf_video_folder
                elif 'ecva' in original_name.lower():
                    video_folder = ecva_video_folder
                else:
                    raise ValueError(f"Unknown dataset prefix in video name: {original_name}")

                for prefix in ['msad_', 'ucf_', 'ecva_']:
                    if original_name.lower().startswith(prefix):
                        original_name = original_name[len(prefix):]

                video_path = os.path.join(video_folder, original_name)
                example = {
                    "problem": {
                        "question": row['Question'],
                        "options": options
                    },
                    "solution": {
                        "answer": row['Correct Option'],
                    },
                    "video_path": video_path,  
                }

                examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:1])
        dataset = Dataset.from_list(examples)


        dataset.client = None
        def __getitem__(self, idx): # Define getitem within the scope where dataset is available
            retry_count = 0
            example = dataset[idx]
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset

            try:
                messages = [{"role": "user", "content": [{"type": "video", "video": example["video_path"][0], "total_pixels": 3584 * 28 * 28, "min_pixels": 16 * 28 * 28,},]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, client=self.client)
                fps_inputs = video_kwargs['fps']
                # # data_to_return["image_inputs"] = [torch.load(os.path.join(example["video_path"][0], "image_inputs.pt"))]
                data_to_return["video_inputs"] = [video_inputs]
                # with open(os.path.join(example["video_path"][0], "video_kwargs.json"), 'r') as f:
                data_to_return["video_kwargs"] = [video_kwargs]
            except Exception as e:
                print(f"Warning: Error loading preprocessed data from {example['video_path'][0]}, falling back to video_path. Error: {e}")
                retry_count += 1
                MAX_RETRY = 20
                if retry_count > MAX_RETRY:
                    raise RuntimeError(f"Tried {MAX_RETRY} times but still failed.")
                print(idx)
                idx = idx + 1
                return self.__getitem__(idx)

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) # Bind getitem to the dataset

        return dataset
    train_dataset = create_dataset_from_csv(train_data_path, "train")
    eval_dataset = create_dataset_from_csv(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset, now handles both raw and preprocessed data
    dataset = load_csv_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.train_video_folder,
        script_args.eval_video_folder
        # script_args.preprocessed_data_path # Pass preprocessed_data_path
    )

    # import pdb; pdb.set_trace()


    if not training_args.use_vllm:
        trainer_cls = Qwen2VLGRPOTrainer
    else:
        raise NotImplementedError
    
    print("using: ", trainer_cls)

    # from peft import LoraConfig, get_peft_model

    # lora_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     inference_mode=False,
    #     r=16,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    # )

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        # peft_config=lora_config,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub 
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)