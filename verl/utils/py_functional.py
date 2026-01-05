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
"""
Contain small python utility functions
"""

import importlib.util
import re
from functools import lru_cache
from typing import Any, Dict, List, Union, Tuple, Set, Optional

import numpy as np
import yaml
from yaml import Dumper


def is_sci_notation(number: float) -> bool:
    pattern = re.compile(r"^[+-]?\d+(\.\d*)?[eE][+-]?\d+$")
    return bool(pattern.match(str(number)))


def float_representer(dumper: Dumper, number: Union[float, np.float32, np.float64]):
    if is_sci_notation(number):
        value = str(number)
        if "." not in value and "e" in value:
            value = value.replace("e", ".0e", 1)
    else:
        value = str(round(number, 3))

    return dumper.represent_scalar("tag:yaml.org,2002:float", value)


yaml.add_representer(float, float_representer)
yaml.add_representer(np.float32, float_representer)
yaml.add_representer(np.float64, float_representer)


@lru_cache
def is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def union_two_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Union two dict. Will throw an error if there is an item not the same object with the same key."""
    for key in dict2.keys():
        if key in dict1:
            assert dict1[key] == dict2[key], f"{key} in dict1 and dict2 are not the same object"

        dict1[key] = dict2[key]

    return dict1


def append_to_dict(data: Dict[str, List[Any]], new_data: Dict[str, Any]) -> None:
    """Append dict to a dict of list."""
    for key, val in new_data.items():
        if key not in data:
            data[key] = []

        data[key].append(val)


def unflatten_dict(data: Dict[str, Any], sep: str = "/") -> Dict[str, Any]:
    unflattened = {}
    for key, value in data.items():
        pieces = key.split(sep)
        pointer = unflattened
        for piece in pieces[:-1]:
            if piece not in pointer:
                pointer[piece] = {}

            pointer = pointer[piece]

        pointer[pieces[-1]] = value

    return unflattened


def flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    flattened = {}
    for key, value in data.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value

    return flattened


def convert_dict_to_str(data: Dict[str, Any]) -> str:
    return yaml.dump(data, indent=2)


def _ngram_list(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return the list of contiguous n-grams (as tuples) from tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def _ngram_set(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return set(_ngram_list(tokens, n))

def _overlap_max(
    cur_grams: Set[Tuple[str, ...]],
    history_sets: List[Set[Tuple[str, ...]]],
) -> Tuple[float, int, Set[Tuple[str, ...]]]:
    """
    Max fraction overlap vs any prior: |G_cur ∩ G_j| / |G_cur|
    Returns (score, j_idx, repeated_ngrams_for_that_j)
    """
    if not cur_grams:
        return 0.0, -1, set()
    denom = max(1, len(cur_grams))
    best, best_idx, best_rep = 0.0, -1, set()
    for j, G_j in enumerate(history_sets):
        rep = cur_grams & G_j
        score = len(rep) / denom
        if score > best:
            best, best_idx, best_rep = score, j, rep
    return best, best_idx, best_rep

def _jaccard_max(
    cur_grams: Set[Tuple[str, ...]],
    history_sets: List[Set[Tuple[str, ...]]],
) -> Tuple[float, int, Set[Tuple[str, ...]]]:
    """
    Max Jaccard vs any prior: |∩| / |∪|
    """
    if not cur_grams:
        return 0.0, -1, set()
    best, best_idx, best_rep = 0.0, -1, set()
    for j, G_j in enumerate(history_sets):
        inter = cur_grams & G_j
        union = cur_grams | G_j
        score = len(inter) / max(1, len(union))
        if score > best:
            best, best_idx, best_rep = score, j, inter
    return best, best_idx, best_rep

def _overlap_avg(
    cur_grams: Set[Tuple[str, ...]],
    history_sets: List[Set[Tuple[str, ...]]],
) -> float:
    """Average fraction overlap across prior queries."""
    if not cur_grams or not history_sets:
        return 0.0
    denom = max(1, len(cur_grams))
    return sum(len(cur_grams & G_j) / denom for G_j in history_sets) / len(history_sets)

def compute_query_weights_and_distinct_reward(
    current_query_tokens: List[str],
    prior_queries_tokens: List[List[str]],
    n: int = 3,
    alpha: float = 0.15,
    mode: str = "max",    # "max", "jaccard_max", or "avg"
    fallback_n_if_short: bool = True,
) -> Dict[str, object]:
    """
    Merge of (1) per-token weights w_u for the current <search> span and
    (2) Distinct-n negative reward.

    Strategy:
      • We try n = min(n, len(query)) and, if fallback enabled, step down n→1
        until we *find repeated n-grams*. The first n that yields any repeats
        is used for weights (so weights focus on the actually duplicated bits).
      • The reward/overlap are also computed at that n (if none found, overlap=0).

    Returns dict:
      - weights: List[float] of length len(current_query_tokens), sum to 1.0
      - reward: float (negative)
      - overlap: float in [0,1]
      - n_used: int
      - mode: str
      - culprit_index: int (index of worst prior for "max"/"jaccard_max"; -1 if none)
      - repeated_ngrams: Set[Tuple[str,...]] (the culprits for diagnostics)
    """
    L = len(current_query_tokens)
    if L == 0:
        return {
            "weights": [],
            "reward": 0.0,
            "overlap": 0.0,
            "n_used": n,
            "mode": mode,
            "culprit_index": -1,
            "repeated_ngrams": set(),
        }

    # Determine the starting n we attempt
    start_n = min(n, L) if fallback_n_if_short else n

    # Precompute per-n history n-gram sets
    def history_sets_for(nn: int) -> List[Set[Tuple[str, ...]]]:
        return [_ngram_set(pq, nn) for pq in prior_queries_tokens]

    # We'll fill these once we settle on n_used
    n_used = start_n
    weights: List[float] = [1.0 / L] * L  # default uniform
    overlap = 0.0
    culprit_index = -1
    repeated_ngrams: Set[Tuple[str, ...]] = set()

    # Try decreasing n to find any repeats (for weights).
    found_repeats = False
    nn = start_n
    while True:
        cur_list = _ngram_list(current_query_tokens, nn)
        cur_set = set(cur_list)
        H_sets = history_sets_for(nn)

        # Identify repeated grams via union-of-history (good for weights)
        H_union = set().union(*H_sets) if H_sets else set()
        repeated_positions = []  # start positions of repeated n-grams in current query

        if cur_list and H_union:
            for i, gram in enumerate(cur_list):
                if gram in H_union:
                    repeated_positions.append(i)

        # If repeats found at this n, compute weights based on counts
        if repeated_positions:
            counts = [0] * L
            for i in repeated_positions:
                for u in range(i, min(i + nn, L)):
                    counts[u] += 1
            total = sum(counts)
            if total > 0:
                weights = [c / total for c in counts]
                n_used = nn
                found_repeats = True
                # Compute overlap & culprit for diagnostics at this n
                if mode == "max":
                    overlap, culprit_index, repeated_ngrams = _overlap_max(cur_set, H_sets)
                elif mode == "jaccard_max":
                    overlap, culprit_index, repeated_ngrams = _jaccard_max(cur_set, H_sets)
                elif mode == "avg":
                    overlap = _overlap_avg(cur_set, H_sets)
                else:
                    raise ValueError("mode must be one of {'max','jaccard_max','avg'}")
                break

        # No repeats at this n
        if not fallback_n_if_short or nn == 1:
            # Compute overlap anyway (will be 0 if no intersection)
            if cur_set:
                if mode == "max":
                    overlap, culprit_index, repeated_ngrams = _overlap_max(cur_set, H_sets)
                elif mode == "jaccard_max":
                    overlap, culprit_index, repeated_ngrams = _jaccard_max(cur_set, H_sets)
                elif mode == "avg":
                    overlap = _overlap_avg(cur_set, H_sets)
                else:
                    raise ValueError("mode must be one of {'max','jaccard_max','avg'}")
            n_used = nn
            break

        nn -= 1  # try smaller n

    reward = -alpha * overlap  # negative reward (penalty for duplication)

    return {
        "weights": weights,
        "reward": float(reward),
        "overlap": float(overlap),
        "n_used": int(n_used),
        "mode": mode,
        "culprit_index": int(culprit_index),
        "repeated_ngrams": repeated_ngrams,
    }


import numpy as np


def ndcg_at_k(retrieved_ids, ground_truth_ids, k):
    """
    计算 NDCG@k (二值相关性：相关=1，不相关=0)

    参数:
    retrieved_ids: list, 模型检索出的有序文档ID列表
    ground_truth_ids: set/list, 真实相关的文档ID集合
    k: int, 截断位置
    """
    # 1. 截断列表，只取前 k 个
    k_eff = min(len(retrieved_ids), k)
    if k_eff == 0:
        return 0.0

    top_k = retrieved_ids[:k_eff]

    # 2. 生成相关性向量 (Relevance Vector)
    # 如果文档在 ground_truth 里，相关性为 1，否则为 0
    # 例如: [1, 0, 1, 0, 0]
    relevance = [1 if doc_id in ground_truth_ids else 0 for doc_id in top_k]

    # --- 计算 DCG (Discounted Cumulative Gain) ---
    # 公式: sum( rel_i / log2(rank_i + 1) )
    # Python索引 i 从 0 开始，所以 rank = i + 1
    # 分母 = log2((i+1) + 1) = log2(i + 2)
    dcg = 0.0
    for i, rel in enumerate(relevance):
        if rel > 0:
            dcg += rel / np.log2(i + 2)

    # --- 计算 IDCG (Ideal DCG) ---
    # 理想情况：所有相关的文档都排在最前面
    # 真实的“相关文档总数”有多少个？
    num_real_relevant = len(ground_truth_ids)

    # 理想的相关性向量：前面全是1，直到填满相关的数量或者达到k
    # 例如：如果 k=5, 但只有 2 个相关文档，理想向量是 [1, 1, 0, 0, 0]
    num_ideal_ones = min(k, num_real_relevant)

    idcg = 0.0
    for i in range(num_ideal_ones):
        idcg += 1.0 / np.log2(i + 2)

    # --- 计算 NDCG ---
    if idcg == 0:
        return 0.0

    return dcg / idcg


def fetch_reward(fetch_idx_list, gt_idxs):
    if len(fetch_idx_list) == 1:
        fetch_idx = fetch_idx_list[0]
        if fetch_idx in gt_idxs:
            return 1.0
        else:
            gt_idxs_int = [int(idx) for idx in gt_idxs]
            fetch_idx_int = int(fetch_idx)
            return 0.3 * float(np.exp(-1/2 * (sum([abs(fetch_idx_int - idx) for idx in gt_idxs_int]) / len(gt_idxs_int))))
    else:
        return -0.5
