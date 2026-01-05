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

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union
import re

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from collections import defaultdict

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.tokenizer import get_processor
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig

from functools import wraps
import requests
import time
from PIL import Image
import base64
from io import BytesIO
from copy import deepcopy
from ...models.transformers.qwen2_vl import get_rope_index
from ...utils.dataset import ImageProcessMixin
import io
import pydevd_pycharm
import torch.distributed as dist


no_proxy_conf = {
    "http": None,
    "https": None
}

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(model_path: str, trust_remote_code: bool) -> Optional[Dict[int, float]]:
    processor = get_processor(model_path, trust_remote_code=trust_remote_code)
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None

def image_to_bytes(image):
    """将Image列的PIL图片转换为字节"""
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")  # 可根据需要选择格式
        return buffer.getvalue()


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            limit_mm_per_prompt={"image": config.limit_images} if config.limit_images > 0 else None,
            disable_mm_preprocessor_cache=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
        )
        #             max_num_batched_tokens=config.max_num_batched_tokens,
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(model_path, trust_remote_code=config.trust_remote_code),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max - 1:
                        print(f"Retry {func.__name__} failed after {max} times")
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator


class vLLMRolloutAgent(vLLMRollout, ImageProcessMixin):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer, processor):
        super().__init__(model_path, config, tokenizer)
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_token_id = tokenizer.encode(processor.image_token)[0]
        self.image_start_id = tokenizer.encode("<|vision_start|>")[0]
        self.image_end_id = tokenizer.encode("<|vision_end|>")[0]
        self.recall_ocr = config.recall_ocr
        self.mm_fetch = config.mm_fetch
        self.max_ocr_length = 1024
        if self.recall_ocr:
            self.result_prefix_ids = self.tokenizer.encode("\n<|im_start|>user\n<result> Retrieved Image Pages: ")
            self.ocr_prefix_ids = self.tokenizer.encode("\nRetrieved OCR Pages: ")
        else:
            self.result_prefix_ids = self.tokenizer.encode("\n<|im_start|>user\n<result>")
        self.result_suffix_ids = self.tokenizer.encode("</result><|im_end|>\n<|im_start|>assistant\n")
        self.top_n = config.top_n
        self.usage_top_n = config.usage_top_n
        self.max_pixels = config.max_pixels
        self.min_pixels = config.min_pixels
        self.vllm_image_limit = config.limit_images
        self.max_turn_num = config.max_turn_num

    @retry(max=5, sleep=1)
    def batch_search(self, query: Union[str, List[str]], doc_ids: Union[str, List[str]], top_n=5, recall_ocr=False):
        if len(query) == 0:
            return 'invalid query'

        url = f'{self.config.search_url}/batch_search'
        if isinstance(query, str):
            query = [query]
        data = {'query': query, 'top_n': top_n, 'id': doc_ids, 'recall_ocr': recall_ocr}
        if not recall_ocr:
            ocr_response = None
            response = requests.post(url, json=data, proxies=no_proxy_conf)
            result_list = []
            page_id_list = []
            result_str_list = []
            for item in response.json():
                # curr_result = ''
                curr_result = []
                curr_page_id = []
                curr_result_str = []

                for line in item:
                    base64_str = line['contents']
                    base64_binary = base64_str.encode("utf-8")
                    curr_result.append(Image.open(BytesIO(base64.urlsafe_b64decode(base64_binary))))
                    curr_page_id.append(line['id'])
                    curr_result_str.append(base64_str)
                    # curr_result += f"{line['contents']}\n\n"
                result_list.append(curr_result)
                page_id_list.append(curr_page_id)
                result_str_list.append(curr_result_str)
            return result_list, page_id_list, result_str_list
        else:
            response, ocr_response = requests.post(url, json=data, proxies=no_proxy_conf).json()

            result_list = []
            page_id_list = []
            result_str_list = []
            for item in response:
                # curr_result = ''
                curr_result = []
                curr_page_id = []
                curr_result_str = []

                for line in item:
                    base64_str = line['contents']
                    base64_binary = base64_str.encode("utf-8")
                    curr_result.append(Image.open(BytesIO(base64.urlsafe_b64decode(base64_binary))))
                    curr_page_id.append(line['id'])
                    curr_result_str.append(base64_str)
                    # curr_result += f"{line['contents']}\n\n"
                result_list.append(curr_result)
                page_id_list.append(curr_page_id)
                result_str_list.append(curr_result_str)

            ocr_result_list = []
            ocr_page_id_list = []
            for item in ocr_response:
                # curr_result = ''
                curr_result = []
                curr_page_id = []

                for line in item:
                    curr_result.append(line['contents'])
                    curr_page_id.append(line['id'])
                    # curr_result += f"{line['contents']}\n\n"
                ocr_result_list.append(curr_result)
                ocr_page_id_list.append(curr_page_id)

            return result_list, page_id_list, result_str_list, ocr_result_list, ocr_page_id_list

    @retry(max=5, sleep=1)
    def batch_fetch(self, query: Union[str, List[str]], doc_ids: Union[str, List[str]], mm_fetch=False):
        if len(query) == 0:
            return 'invalid query'

        url = f'{self.config.search_url}/batch_fetch'
        if isinstance(query, str):
            query = [query]
        data = {'query': query, 'id': doc_ids, 'mm_fetch': mm_fetch}
        response = requests.post(url, json=data, proxies=no_proxy_conf)

        result_list = []
        page_id_list = []
        result_str_list = []
        for item in response.json():
            # curr_result = ''
            curr_result = []
            curr_page_id = []
            curr_result_str = []

            # for line in item:
            if item['id'] != 'error':
                modal, idx = item['id'].split(', ')
                if modal == 'image':
                    base64_str = item['contents']
                    base64_binary = base64_str.encode("utf-8")
                    curr_result.append(Image.open(BytesIO(base64.urlsafe_b64decode(base64_binary))))
                    curr_page_id.append(item['id'])
                    curr_result_str.append(base64_str)
                else:
                    curr_result.append(item['contents'])
                    curr_page_id.append(item['id'])
                    curr_result_str.append(item['contents'])
            else:
                curr_result.append(item['contents'])
                curr_page_id.append(item['id'])
                curr_result_str.append(item['contents'])
                # curr_result += f"{line['contents']}\n\n"
            result_list.append(curr_result)
            page_id_list.append(curr_page_id)
            result_str_list.append(curr_result_str)

        return result_list, page_id_list, result_str_list

    @retry(max=5, sleep=1)
    def search(self, query: str):
        if query == '':
            return 'invalid query'

        url = f'{self.config.search_url}/search'
        data = {'query': query, 'top_n': 5}
        response = requests.post(url, json=data)
        retrieval_text = ''
        for line in response.json():
            retrieval_text += f"{line['contents']}\n\n"
        retrieval_text = retrieval_text.strip()
        return retrieval_text

    def extract_search_content(self, text: str) -> ((int, int), str):
        try:
            start_tag = '<search>'
            end_tag = '</search>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return (start_pos + len(start_tag), end_pos), text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return (10000, 10000), ""

    def extract_fetch_content(self, text: str) -> str:
        try:
            start_tag = '<fetch>'
            end_tag = '</fetch>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                if 'image' in multi_modal_data.keys():
                    vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
                else:
                    vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids)})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": {'image': []}} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        with self.update_sampling_params(**prompts.meta_info):
            curr_inputs = []
            for input in vllm_inputs:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(deepcopy(input))

            doc_ids = []
            for did in non_tensor_batch.pop("doc_id"):
                for _ in range(self.sampling_params.n):
                    doc_ids.append(deepcopy(did))

            # track the status of each input (settings of max token)
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            # collect the result mask of each rollout
            result_mask_list = [[] for _ in range(len(curr_inputs))]

            # collect the results
            result_ids_list = [[] for _ in range(len(curr_inputs))]

            result_attention_mask = [[] for _ in range(len(curr_inputs))]

            turn_sequence_mask = [[] for _ in range(len(curr_inputs))]

            query_mask_list = [[] for _ in range(len(curr_inputs))]

            ocr_mask_list = [[] for _ in range(len(curr_inputs))]
            # collect the gotten page number
            page_numbers_list = [{'N.O': defaultdict(list)} for _ in range(len(curr_inputs))]
            ocr_page_numbers_list = [{'N.O': defaultdict(list)} for _ in range(len(curr_inputs))]

            # collect the gotten page image
            result_image_list = [{'data': []} for _ in range(len(curr_inputs))]

            # generate until all inputs are finished
            turn_idx = 0
            while active_indices and turn_idx < self.max_turn_num:
                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]

                with self.update_sampling_params(n=1, max_tokens=max(active_max_tokens), detokenize=True):
                    completions = self.inference_engine.generate(
                        prompts=active_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                search_queries = []
                search_indices = []
                search_doc_ids = []
                fetch_queries = []
                fetch_indices = []
                fetch_doc_ids = []

                # process each output
                new_active_indices = []
                outputs = [output for completion in completions for output in completion.outputs]
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[i].token_ids

                    finish_reason = outputs[i].finish_reason
                    stop_reason = outputs[i].stop_reason

                    if finish_reason == 'stop' and stop_reason is None:
                        if '<search>' in outputs[i].text and '</search>' in outputs[i].text:
                            (q_start, q_end), search_content = self.extract_search_content(outputs[i].text + self.tokenizer.eos_token)
                            enc = self.processor.tokenizer(outputs[i].text + self.tokenizer.eos_token, return_offsets_mapping=True, add_special_tokens=False)
                            offs = np.array(enc["offset_mapping"], dtype=np.int32)
                            starts, ends = offs[:, 0], offs[:, 1]
                            if not len(starts) == len(output_ids):
                                # pydevd_pycharm.settrace('47.83.127.143', port=47508, stdoutToServer=True,
                                #                         stderrToServer=True)
                                output_ids = enc['input_ids']
                                outputs[i].token_ids = enc['input_ids']
                            query_mask = (((starts >= q_start) & (ends < q_end)) * (turn_idx + 1)).astype(np.int64).tolist()
                            query_mask_list[idx] += query_mask
                            ocr_mask_list[idx] += [0] * len(output_ids)
                            search_queries.append(search_content)
                            search_indices.append(idx)
                            search_doc_ids.append(doc_ids[idx])
                            new_active_indices.append(idx)
                            ## update the current input
                            curr_inputs[idx]['prompt_token_ids'] += output_ids
                            result_mask_list[idx] += [1] * len(output_ids)
                            result_ids_list[idx] += output_ids
                            result_attention_mask[idx] += [1] * len(output_ids)
                            turn_sequence_mask[idx] += [turn_idx] * len(output_ids)
                        elif '<fetch>' in outputs[i].text and '</fetch>' in outputs[i].text:
                            fetch_content = self.extract_fetch_content(outputs[i].text)
                            fetch_queries.append(fetch_content)
                            fetch_indices.append(idx)
                            fetch_doc_ids.append(doc_ids[idx])
                            new_active_indices.append(idx)
                            ## update the current input
                            curr_inputs[idx]['prompt_token_ids'] += output_ids
                            result_mask_list[idx] += [1] * len(output_ids)
                            result_ids_list[idx] += output_ids
                            result_attention_mask[idx] += [1] * len(output_ids)
                            turn_sequence_mask[idx] += [turn_idx] * len(output_ids)
                            query_mask_list[idx] += [0] * len(output_ids)
                            ocr_mask_list[idx] += [0] * len(output_ids)
                        else:
                            curr_inputs[idx]['prompt_token_ids'] += output_ids
                            result_mask_list[idx] += [1] * len(output_ids)
                            result_ids_list[idx] += output_ids
                            result_attention_mask[idx] += [1] * len(output_ids)
                            turn_sequence_mask[idx] += [turn_idx] * len(output_ids)
                            query_mask_list[idx] += [0] * len(output_ids)
                            ocr_mask_list[idx] += [0] * len(output_ids)
                    else:
                        curr_inputs[idx]['prompt_token_ids'] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                        result_ids_list[idx] += output_ids
                        result_attention_mask[idx] += [1] * len(output_ids)
                        turn_sequence_mask[idx] += [turn_idx] * len(output_ids)
                        query_mask_list[idx] += [0] * len(output_ids)
                        ocr_mask_list[idx] += [0] * len(output_ids)

                # batch process the search requests
                almost_full_indices = []
                if search_queries:
                    if not self.recall_ocr:
                        search_results, search_page_ids, search_results_str = self.batch_search(search_queries, search_doc_ids, top_n=self.top_n, recall_ocr=self.recall_ocr)
                        ocr_search_results, ocr_search_page_ids = None, None
                    else:
                        search_results, search_page_ids, search_results_str, ocr_search_results, ocr_search_page_ids = self.batch_search(
                            search_queries,
                            search_doc_ids,
                            top_n=self.top_n,
                            recall_ocr=self.recall_ocr)

                    for idx, result, p_id, result_str in zip(search_indices, search_results, search_page_ids, search_results_str):
                        # update the output, add the search result
                        processed_result = [self.process_image(image) for image in result]
                        image_inputs = self.processor.image_processor(processed_result, return_tensors='pt')
                        top_ki = 0
                        curr_inputs[idx]['prompt_token_ids'].extend(self.result_prefix_ids)
                        result_mask_list[idx] += [0] * len(self.result_prefix_ids)
                        result_ids_list[idx] += self.result_prefix_ids
                        result_attention_mask[idx] += [1] * len(self.result_prefix_ids)
                        turn_sequence_mask[idx] += [-1] * len(self.result_prefix_ids)
                        query_mask_list[idx] += [0] * len(self.result_prefix_ids)
                        ocr_mask_list[idx] += [1] * len(self.result_prefix_ids)
                        page_numbers_list[idx]['N.O'][turn_idx].extend(p_id)
                        while top_ki < self.usage_top_n:
                            if len(result_mask_list[idx]) + (image_inputs['image_grid_thw'][top_ki].prod() // self.processor.image_processor.merge_size**2) + 2 < self.config.response_length and len(result_image_list[idx]['data']) < self.vllm_image_limit:
                                result_image_list[idx]['data'].append(result_str[top_ki])
                                pstr = self.tokenizer.encode("{}-th page: ".format(str(int(p_id[top_ki]) + 1)))
                                if 'multi_modal_data' not in curr_inputs[idx].keys():
                                    curr_inputs[idx]['multi_modal_data'] = {'image': []}
                                curr_inputs[idx]['multi_modal_data']['image'].append(processed_result[top_ki])
                                curr_inputs[idx]['prompt_token_ids'].extend(pstr + [self.image_start_id, self.image_token_id, self.image_end_id])
                                result_mask_list[idx] += [0] * (image_inputs[
                                                                    'image_grid_thw'][top_ki].prod() // self.processor.image_processor.merge_size**2) + [0] * 2 + [0] * len(pstr)
                                query_mask_list[idx] += [0] * (image_inputs[
                                                                    'image_grid_thw'][top_ki].prod() // self.processor.image_processor.merge_size**2) + [0] * 2 + [0] * len(pstr)
                                ocr_mask_list[idx] += [1] * len(pstr) + [0] * (image_inputs[
                                                                    'image_grid_thw'][top_ki].prod() // self.processor.image_processor.merge_size**2) + [0] * 2
                                turn_sequence_mask[idx] += [-1] * (image_inputs[
                                                                    'image_grid_thw'][top_ki].prod() // self.processor.image_processor.merge_size**2) + [-1] * 2 + [-1] * len(pstr)
                                result_attention_mask[idx] += [1] * (image_inputs[
                                                                    'image_grid_thw'][top_ki].prod() // self.processor.image_processor.merge_size**2) + [1] * 2 + [1] * len(pstr)
                                result_ids_list[idx] += pstr + [self.image_start_id] + [self.image_token_id] * (image_inputs[
                                                                    'image_grid_thw'][top_ki].prod() // self.processor.image_processor.merge_size**2) + [self.image_end_id]
                            else:
                                almost_full_indices.append(idx)
                                break
                            top_ki += 1

                    if self.recall_ocr:
                        for idx, result, p_id in zip(search_indices, ocr_search_results, ocr_search_page_ids):
                            # update the output, add the search result
                            top_ki = 0
                            curr_inputs[idx]['prompt_token_ids'].extend(self.ocr_prefix_ids)
                            result_mask_list[idx] += [0] * len(self.ocr_prefix_ids)
                            ocr_mask_list[idx] += [1] * len(self.ocr_prefix_ids)
                            result_ids_list[idx] += self.ocr_prefix_ids
                            result_attention_mask[idx] += [1] * len(self.ocr_prefix_ids)
                            turn_sequence_mask[idx] += [-1] * len(self.ocr_prefix_ids)
                            query_mask_list[idx] += [0] * len(self.ocr_prefix_ids)
                            ocr_page_numbers_list[idx]['N.O'][turn_idx].extend(p_id)
                            while top_ki < self.usage_top_n:
                                ocr = self.tokenizer.encode(result[top_ki] + '\n')
                                if len(ocr) > self.max_ocr_length:
                                    ocr = ocr[:self.max_ocr_length]
                                pstr = self.tokenizer.encode("{}-th page: ".format(str(int(p_id[top_ki]) + 1)))
                                curr_inputs[idx]['prompt_token_ids'].extend(pstr + ocr)
                                result_mask_list[idx] += [0] * (len(pstr) + len(ocr))
                                query_mask_list[idx] += [0] * (len(pstr) + len(ocr))
                                turn_sequence_mask[idx] += [-1] * (len(pstr) + len(ocr))
                                result_attention_mask[idx] += [1] * (len(pstr) + len(ocr))
                                ocr_mask_list[idx] += [1] * (len(pstr) + len(ocr))
                                result_ids_list[idx] += pstr + ocr
                                top_ki += 1

                            curr_inputs[idx]['prompt_token_ids'].extend(self.result_suffix_ids)
                            result_mask_list[idx] += [0] * len(self.result_suffix_ids)
                            query_mask_list[idx] += [0] * len(self.result_suffix_ids)
                            ocr_mask_list[idx] += [1] * len(self.result_suffix_ids)
                            turn_sequence_mask[idx] += [-1] * len(self.result_suffix_ids)
                            result_attention_mask[idx] += [1] * len(self.result_suffix_ids)
                            result_ids_list[idx] += self.result_suffix_ids
                    else:
                        for idx in search_indices:
                            curr_inputs[idx]['prompt_token_ids'].extend(self.result_suffix_ids)
                            result_mask_list[idx] += [0] * len(self.result_suffix_ids)
                            query_mask_list[idx] += [0] * len(self.result_suffix_ids)
                            ocr_mask_list[idx] += [1] * len(self.result_suffix_ids)
                            turn_sequence_mask[idx] += [-1] * len(self.result_suffix_ids)
                            result_attention_mask[idx] += [1] * len(self.result_suffix_ids)
                            result_ids_list[idx] += self.result_suffix_ids

                if fetch_queries:
                    fetch_results, fetch_page_ids, fetch_results_str = self.batch_fetch(fetch_queries, fetch_doc_ids, self.mm_fetch)
                    for idx, result, p_id, result_str in zip(fetch_indices, fetch_results, fetch_page_ids, fetch_results_str):
                        # update the output, add the fetch result
                        if p_id[0] != 'error':
                            pattern = r"^(image|text),\s*([0-9]\d*)$"
                            match = re.match(pattern, p_id[0])
                            modal, p_id_s = match.group(1), match.group(2)
                            if modal == 'image':
                                processed_result = [self.process_image(image) for image in result]
                                image_inputs = self.processor.image_processor(processed_result, return_tensors='pt')
                                if len(result_ids_list[idx]) + (image_inputs['image_grid_thw'][0].prod() // self.processor.image_processor.merge_size ** 2) + len(self.result_prefix_ids) + len(self.result_suffix_ids) + 2 < self.config.response_length and len(result_image_list[idx]['data']) < self.vllm_image_limit:
                                    result_image_list[idx]['data'].append(result_str[0])
                                    page_numbers_list[idx]['N.O'][turn_idx].append(p_id_s)
                                    if 'multi_modal_data' not in curr_inputs[idx].keys():
                                        curr_inputs[idx]['multi_modal_data'] = {'image': []}
                                    curr_inputs[idx]['multi_modal_data']['image'].append(processed_result[0])
                                    curr_inputs[idx]['prompt_token_ids'].extend(self.result_prefix_ids + [self.image_start_id, self.image_token_id, self.image_end_id] + self.result_suffix_ids)
                                    result_mask_list[idx] += [0] * (image_inputs[
                                                                        'image_grid_thw'][0].prod() // self.processor.image_processor.merge_size ** 2) + [0] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + 2)
                                    query_mask_list[idx] += [0] * (image_inputs[
                                                                        'image_grid_thw'][0].prod() // self.processor.image_processor.merge_size ** 2) + [0] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + 2)
                                    ocr_mask_list[idx] += [1] * len(self.result_prefix_ids) + [0] * (image_inputs[
                                                                        'image_grid_thw'][0].prod() // self.processor.image_processor.merge_size ** 2) + [0] * 2 + [1] * len(self.result_suffix_ids)
                                    turn_sequence_mask[idx] += [-1] * (image_inputs[
                                                                        'image_grid_thw'][0].prod() // self.processor.image_processor.merge_size ** 2) + [-1] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + 2)
                                    result_attention_mask[idx] += [1] * (image_inputs[
                                                                        'image_grid_thw'][0].prod() // self.processor.image_processor.merge_size ** 2) + [1] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + 2)
                                    result_ids_list[idx] += self.result_prefix_ids + [self.image_start_id] + [self.image_token_id] * (image_inputs[
                                                                        'image_grid_thw'][0].prod() // self.processor.image_processor.merge_size ** 2) + [self.image_end_id] + self.result_suffix_ids
                                else:
                                    almost_full_indices.append(idx)
                            else:
                                ocr = self.tokenizer.encode(result[0])
                                ocr_page_numbers_list[idx]['N.O'][turn_idx].append(p_id_s)
                                curr_inputs[idx]['prompt_token_ids'].extend(
                                    self.result_prefix_ids + ocr + self.result_suffix_ids)
                                result_mask_list[idx] += [0] * (
                                            len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(
                                        ocr))
                                query_mask_list[idx] += [0] * (
                                            len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(
                                        ocr))
                                ocr_mask_list[idx] += [1] * (
                                        len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(
                                    ocr))
                                turn_sequence_mask[idx] += [-1] * (
                                            len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(
                                        ocr))
                                result_attention_mask[idx] += [1] * (
                                            len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(
                                        ocr))
                                result_ids_list[idx] += self.result_prefix_ids + ocr + self.result_suffix_ids
                        else:
                            fetch_err_msg = self.tokenizer.encode(result[0])
                            curr_inputs[idx]['prompt_token_ids'].extend(self.result_prefix_ids + fetch_err_msg + self.result_suffix_ids)
                            result_mask_list[idx] += [0] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(fetch_err_msg))
                            query_mask_list[idx] += [0] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(fetch_err_msg))
                            ocr_mask_list[idx] += [1] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(fetch_err_msg))
                            turn_sequence_mask[idx] += [-1] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(fetch_err_msg))
                            result_attention_mask[idx] += [1] * (len(self.result_prefix_ids) + len(self.result_suffix_ids) + len(fetch_err_msg))
                            result_ids_list[idx] += self.result_prefix_ids + fetch_err_msg + self.result_suffix_ids
                            # check if need to truncate for active indices
                length_checked_active_indices = []
                for idx in active_indices:
                    assert len(result_ids_list[idx]) == len(result_mask_list[idx]), f"result_ids_list: {len(result_ids_list)}, result_mask_list: {len(result_mask_list[idx])}"
                    if len(result_mask_list[idx]) >= self.config.response_length:
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                        query_mask_list[idx] = query_mask_list[idx][:self.config.response_length]
                        ocr_mask_list[idx] = ocr_mask_list[idx][:self.config.response_length]
                        turn_sequence_mask[idx] = turn_sequence_mask[idx][:self.config.response_length]
                        result_attention_mask[idx] = result_attention_mask[idx][:self.config.response_length]
                        result_ids_list[idx] = result_ids_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(result_ids_list[idx])
                        if idx in new_active_indices and idx not in almost_full_indices:
                            length_checked_active_indices.append(idx)
                active_indices = length_checked_active_indices
                turn_idx += 1

            padding_indices = []
            for ai in range(len(curr_inputs)):
                if len(result_image_list[ai]['data']) == 0:
                    padding_indices.append(ai)
            if len(padding_indices) > 0:
                padding_results = [[Image.new('RGB', (150, 150), 'white')]] * len(padding_indices)
                padding_results_str = [[base64.b64encode(image_to_bytes(padding_results[0][0])).decode("utf-8")]] * len(padding_indices)
                padding_page_ids = [['0']] * len(padding_indices)
                for idx, result, p_id, result_str in zip(padding_indices, padding_results, padding_page_ids, padding_results_str):
                    processed_result = [self.process_image(image) for image in result]
                    image_inputs = self.processor.image_processor(processed_result, return_tensors='pt')
                    padding_img_len = len(result_ids_list[idx]) + (image_inputs['image_grid_thw'][
                                                                       0].prod() // self.processor.image_processor.merge_size ** 2) + len(
                        self.result_prefix_ids) + len(self.result_suffix_ids) + 2
                    result_image_list[idx]['data'].append(result_str[0])
                    page_numbers_list[idx]['N.O'][-1].append(p_id[0])
                    if 'multi_modal_data' not in curr_inputs[idx].keys():
                        curr_inputs[idx]['multi_modal_data'] = {'image': []}
                    curr_inputs[idx]['multi_modal_data']['image'].append(processed_result[0])
                    ext_len = len(result_ids_list[idx])
                    if ext_len + padding_img_len > self.config.response_length:
                        t = self.config.response_length - padding_img_len
                        result_mask_list[idx] = result_mask_list[idx][:t]
                        query_mask_list[idx] = query_mask_list[idx][:t]
                        ocr_mask_list[idx] = ocr_mask_list[idx][:t]
                        result_attention_mask[idx] = result_attention_mask[idx][:t]
                        turn_sequence_mask[idx] = turn_sequence_mask[idx][:t]
                        result_ids_list[idx] = result_ids_list[idx][:t]
                    result_mask_list[idx] += [0] * (image_inputs[
                                                        'image_grid_thw'][
                                                        0].prod() // self.processor.image_processor.merge_size ** 2) + [
                                                 0] * (len(self.result_prefix_ids) + len(
                        self.result_suffix_ids) + 2)
                    query_mask_list[idx] += [0] * (image_inputs[
                                                        'image_grid_thw'][
                                                        0].prod() // self.processor.image_processor.merge_size ** 2) + [
                                                 0] * (len(self.result_prefix_ids) + len(
                        self.result_suffix_ids) + 2)
                    ocr_mask_list[idx] += [1] * (image_inputs[
                                                       'image_grid_thw'][
                                                       0].prod() // self.processor.image_processor.merge_size ** 2) + [
                                                1] * (len(self.result_prefix_ids) + len(
                        self.result_suffix_ids) + 2)
                    turn_sequence_mask[idx] += [-1] * (image_inputs[
                                                        'image_grid_thw'][
                                                        0].prod() // self.processor.image_processor.merge_size ** 2) + [
                                                 -1] * (len(self.result_prefix_ids) + len(
                        self.result_suffix_ids) + 2)
                    result_attention_mask[idx] += [1] * (image_inputs[
                                                             'image_grid_thw'][
                                                             0].prod() // self.processor.image_processor.merge_size ** 2) + [
                                                      1] * (len(self.result_prefix_ids) + len(
                        self.result_suffix_ids) + 2)
                    result_ids_list[idx] += self.result_prefix_ids + [self.image_start_id] + [
                        self.image_token_id] * (
                                                    image_inputs[
                                                        'image_grid_thw'][
                                                        0].prod() // self.processor.image_processor.merge_size ** 2) + [
                                                self.image_end_id] + self.result_suffix_ids

            response_ids = VF.pad_2d_list_to_length(
                result_ids_list, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)
            result_mask = VF.pad_2d_list_to_length(
                result_mask_list, 0, max_length=self.config.response_length
            ).to(input_ids.device)
            query_mask = VF.pad_2d_list_to_length(
                query_mask_list, 0, max_length=self.config.response_length
            ).to(input_ids.device)
            ocr_mask = VF.pad_2d_list_to_length(
                ocr_mask_list, 0, max_length=self.config.response_length
            ).to(input_ids.device)
            turn_sequence_mask = VF.pad_2d_list_to_length(
                turn_sequence_mask, -1, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                    )

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        # response_length = response_ids.size(1)
        response_mask = VF.pad_2d_list_to_length(
                result_attention_mask, 0, max_length=self.config.response_length
            ).to(input_ids.device)
        delta_position_id = []
        end_of_response_position_mask = torch.zeros_like(result_mask)
        for i, input in enumerate(curr_inputs):
            max_turn = torch.max(turn_sequence_mask[i, :]).item() + 1
            for mti in range(max_turn):
                end_of_response_position_mask[i, torch.max((turn_sequence_mask[i, :] == mti).nonzero(as_tuple=True)[0]).item()] = 1
            if 'multi_modal_data' in input.keys():
                d = input['multi_modal_data']['image']
            else:
                d = []
            if len(d) > 0:
                image_inputs = self.processor.image_processor(d, return_tensors='pt')
                delta_pid = get_rope_index(
                    self.processor,
                    input_ids=response_ids[i, :],
                    image_grid_thw=image_inputs['image_grid_thw'],
                )
            else:
                delta_pid = get_rope_index(
                    self.processor,
                    input_ids=response_ids[i, :].unsqueeze(0),
                    image_grid_thw=None,
                )
            delta_position_id.append(delta_pid.unsqueeze(0))
        delta_position_id = torch.cat(delta_position_id, dim=0).to(position_ids.device)
        non_tensor_batch["multi_modal_data"] = np.array(result_image_list, dtype=object)
        non_tensor_batch["token_consumption"] = response_mask.sum(dim=-1).cpu().numpy()
        # delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        # delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        # if position_ids.dim() == 3:  # qwen2vl mrope
        #     delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_mask
        non_tensor_batch['page_ids'] = np.array(page_numbers_list, dtype=object)
        non_tensor_batch['doc_id'] = np.array(doc_ids, dtype=object)
        non_tensor_batch['ocr_page_ids'] = np.array(ocr_page_numbers_list, dtype=object)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "loss_mask": loss_mask,
                "end_of_response_position_mask": end_of_response_position_mask,
                "turn_sequence_mask": turn_sequence_mask,
                "position_ids": position_ids,
                "query_mask": query_mask - 1,
                "ocr_mask": ocr_mask
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
