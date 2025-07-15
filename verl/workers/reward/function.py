# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib.util
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer
import re
import pydevd_pycharm

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


ScoreFunction = Callable[[str, str], RewardScore]


@dataclass
class FunctionRewardManager:
    config: RewardConfig
    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        """Load score function."""
        if self.config.score_function is None:
            raise ValueError("Score function is not provided.")

        if not os.path.exists(self.config.score_function):
            raise FileNotFoundError(f"Score function file {self.config.score_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_score_fn", self.config.score_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_score_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load score function: {e}")

        if not hasattr(module, self.config.score_function_name):
            raise AttributeError(f"Module {module} does not have function {self.config.score_function_name}.")

        score_fn: ScoreFunction = getattr(module, self.config.score_function_name)
        print(f"Using score function `{self.config.score_function_name}` from `{self.config.score_function}`.")
        self.score_fn = partial(score_fn, **self.config.score_function_kwargs)

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        if not self.config.multi_turn_rewards:
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
                )
                ground_truth = data_item.non_tensor_batch["ground_truth"]

                score = self.score_fn(response_str, ground_truth)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
        else:
            # pydevd_pycharm.settrace('47.83.127.143', port=47508, stdoutToServer=True, stderrToServer=True)
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                turn_sequence_mask = data_item.batch["turn_sequence_mask"]
                end_of_response_position_mask = data_item.batch["end_of_response_position_mask"]
                max_turn = torch.max(turn_sequence_mask).item() + 1
                end_of_response_position = end_of_response_position_mask.nonzero(as_tuple=True)[0]
                pre_pids = set()
                gt_pids = set(data_item.non_tensor_batch["answer_page_idx"])
                turn_level_reward_metrics = defaultdict(list)
                for mti in range(max_turn):
                    turn_response_ids = torch.masked_select(response_ids, (turn_sequence_mask == mti))
                    response_str = self.tokenizer.decode(
                        turn_response_ids, skip_special_tokens=self.config.skip_special_tokens
                    )

                    if mti != max_turn - 1:
                        co_pid = set(data_item.non_tensor_batch['page_ids']['N.O'][mti])
                        score = self.score_fn(response_str, None)
                        acs = len(co_pid.intersection(gt_pids))
                        rps = len(co_pid.intersection(pre_pids))
                        if score["overall"] == 0.0:
                            if acs == 1.0 and rps == 0.0:
                                oa = score["overall"] + 1.0
                            elif acs == 1.0 and rps == 1.0:
                                oa = score["overall"] - 0.5
                            elif acs == 0.0 and rps == 0.0 and co_pid:
                                oa = score["overall"] + 0.0
                            elif acs == 0.0 and rps == 0.0 and not co_pid:
                                oa = score["overall"] - 0.5
                            elif acs == 0.0 and rps == 1.0:
                                oa = score["overall"] - 0.5
                        # if acs != 0 or rps != 0:
                        #     oa = (acs - rps * self.config.repetition_penalty_factor) * 0.6 + score["overall"]
                        # else:
                        #     oa = 0.05
                            score["overall"] = oa
                        # score["accuracy"] = (acs - rps * self.config.repetition_penalty_factor) * 0.6
                        # if score["overall"] == 0.4:
                        #     acs = len(co_pid.intersection(gt_pids)) / (len(gt_pids) + 1e-8)
                        #     rps = len(co_pid.intersection(pre_pids)) / (len(co_pid) + 1e-8)
                        #     oa = (acs - rps * self.config.repetition_penalty_factor) * 0.6 + score["overall"]
                        #     score["overall"] = oa
                        #     score["accuracy"] = (acs - rps * self.config.repetition_penalty_factor) * 0.6
                        pre_pids.update(co_pid)
                    else:
                        ground_truth = data_item.non_tensor_batch["ground_truth"]
                        score = self.score_fn(response_str, ground_truth)
                    for key, value in score.items():
                        turn_level_reward_metrics[key].append(value)

                    reward_tensor[i, end_of_response_position[mti]] = score["overall"]
                for key, value in turn_level_reward_metrics.items():
                    if key in {'overall', 'format', 'accuracy', 'turn_response_length'}:
                        reward_metrics[key].append((sum(value) / len(value)))
                    else:
                        reward_metrics[key].append(sum(value))
                reward_metrics['is_accuracy'].append(len(pre_pids.intersection(gt_pids)) / len(gt_pids))
                reward_metrics['num_turn'].append(max_turn)

        return reward_tensor, reward_metrics
