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
import torch.distributed as dist
from ...utils.py_functional import compute_query_weights_and_distinct_reward, ndcg_at_k, fetch_reward


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
            # rank = int(os.environ.get('RANK', '0'))
            # # 只有在 Rank 0 且在正确的节点上时才连接
            # # 这里的逻辑假设你的 SSH 隧道是建立在运行 Rank 0 的那个节点上的
            # if rank == 0:
            #     # try:
            #     # print(f"Rank {rank}: Attempting to connect to Debug Server...")
            #     pydevd_pycharm.settrace('127.0.0.1', port=47508, stdoutToServer=True, stderrToServer=True)
            #     #     print(f"Rank {rank}: Connected!")
            #     # except Exception as e:
            #     #     print(f"Rank {rank}: Failed to connect to Debug Server. Error: {e}")
            #     # 选择性：如果连接失败，是否继续运行？
            #     # pass
            # else:
            #     print(f"Rank {rank}: Skipping Debug Server connection.")
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                query_mask = data_item.batch["query_mask"]
                turn_sequence_mask = data_item.batch["turn_sequence_mask"]
                end_of_response_position_mask = data_item.batch["end_of_response_position_mask"]
                max_turn = torch.max(turn_sequence_mask).item() + 1
                end_of_response_position = end_of_response_position_mask.nonzero(as_tuple=True)[0]
                pre_pids_image = set()
                pre_pids_text = set()
                pre_ndcg = None
                all_queries = []
                gt_pids = set(data_item.non_tensor_batch["answer_page_idx"])
                turn_level_reward_metrics = defaultdict(list)
                for mti in range(max_turn):
                    turn_response_ids = torch.masked_select(response_ids, (turn_sequence_mask == mti))
                    response_str = self.tokenizer.decode(
                        turn_response_ids, skip_special_tokens=self.config.skip_special_tokens
                    )

                    # if max_turn > 1:
                    if mti != max_turn - 1:
                        score = self.score_fn(response_str, None, self.config.mm_fetch)
                        if score["overall"] == 0.0 and data_item.non_tensor_batch["ground_truth"] != 'The problem is not answerable':
                            if score["search"] > 0:
                                current_search_image = set(data_item.non_tensor_batch['page_ids']['N.O'][mti][:self.config.usage_top_n])

                                if self.config.recall_ocr:
                                    current_search_text = set(data_item.non_tensor_batch['ocr_page_ids']['N.O'][mti][:self.config.usage_top_n])
                                    acs = len(((current_search_image | current_search_text) & gt_pids) - pre_pids_image - pre_pids_text)
                                    rps = 0.0
                                    nss = 0.0
                                    pre_pids_text.update(data_item.non_tensor_batch['ocr_page_ids']['N.O'][mti])
                                    pre_pids_image.update(data_item.non_tensor_batch['page_ids']['N.O'][mti])
                                else:
                                    acs = len((current_search_image & gt_pids) - pre_pids_image)
                                    nss = 0.0
                                    rps = 0.0
                                    pre_pids_image.update(data_item.non_tensor_batch['page_ids']['N.O'][mti][:self.config.usage_top_n])
                            elif score["fetch_image"] > 0:
                                current_search_image = set(data_item.non_tensor_batch['page_ids']['N.O'][mti])
                                acs = fetch_reward(data_item.non_tensor_batch['page_ids']['N.O'][mti], gt_pids)
                                rps = len(current_search_image.intersection(pre_pids_image))
                                nss = 0.0
                                pre_pids_image.update(data_item.non_tensor_batch['page_ids']['N.O'][mti])
                            else:
                                assert score["fetch_text"] > 0
                                acs = fetch_reward(data_item.non_tensor_batch['ocr_page_ids']['N.O'][mti], gt_pids)
                                current_search_text = set(
                                    data_item.non_tensor_batch['ocr_page_ids']['N.O'][mti])
                                rps = len(current_search_text.intersection(pre_pids_text))
                                pre_pids_text.update(data_item.non_tensor_batch['ocr_page_ids']['N.O'][mti])
                                nss = 0.0
                            score['overall'] = score['overall'] + acs - rps - nss

                            if self.config.search_query_repetition_penalty:
                                query_positions = (query_mask == mti).nonzero(as_tuple=True)[0] + 1
                                query_ids = torch.masked_select(response_ids, (query_mask == mti))
                                query_tokens = self.tokenizer.convert_ids_to_tokens(query_ids)
                                query_reward = torch.zeros_like(query_ids, dtype=reward_tensor.dtype, device=reward_tensor.device)
                                if len(all_queries) > 0:
                                    res = compute_query_weights_and_distinct_reward(
                                        current_query_tokens=query_tokens,
                                        prior_queries_tokens=all_queries,
                                        n=3,
                                        alpha=0.3,  # start small; warm up later
                                        mode="max",  # or "max" (stricter) / "avg"
                                        fallback_n_if_short=True,
                                    )
                                    query_reward = torch.tensor(res["weights"], dtype=reward_tensor.dtype, device=reward_tensor.device) * res["reward"]
                                all_queries.append(query_tokens)
                                reward_tensor[i, query_positions] = query_reward
                    else:
                        ground_truth = data_item.non_tensor_batch["ground_truth"]
                        score = self.score_fn(response_str, ground_truth, self.config.mm_fetch)
                    for key, value in score.items():
                        turn_level_reward_metrics[key].append(value)

                    reward_tensor[i, end_of_response_position[mti]] = score["overall"]
                for key, value in turn_level_reward_metrics.items():
                    if key in {'overall', 'format', 'accuracy', 'turn_response_length'}:
                        reward_metrics[key].append((sum(value) / len(value)))
                    else:
                        reward_metrics[key].append(sum(value))
                reward_metrics['is_accuracy'].append(len(pre_pids_image.union(pre_pids_text).intersection(gt_pids)) / len(gt_pids))
                reward_metrics['num_turn'].append(max_turn)
                # if dist.is_initialized():
                #     dist.barrier()

        return reward_tensor, reward_metrics
