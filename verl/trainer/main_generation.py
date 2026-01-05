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
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import os

import pandas as pd
import json
from omegaconf import OmegaConf

from examples.score_function.doc_agent import extract_answer, remove_boxed, last_boxed_only_string
from ..protocol import DataProto
from ..workers.fsdp_workers import FSDPWorker
from ..utils.tokenizer import get_tokenizer, get_processor
from ..protocol import pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from .config import GenerationConfig
from ..utils.dataset import RLHFDataset, collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
import datasets
from tqdm import tqdm
import pydevd_pycharm


@ray.remote(num_cpus=1)
def main_task(config: GenerationConfig):
    print(json.dumps(config.to_dict(), indent=2))

    # instantiate tokenizer
    tokenizer = get_tokenizer(
        config.worker.actor.model.model_path,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )
    processor = get_processor(
        config.worker.actor.model.model_path,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )

    if config.worker.rollout.temperature == 0.:
        assert config.worker.rollout.n == 1, 'When temperature=0, n_samples must be 1.'
    assert config.worker.rollout.n >= 1, "n_samples should always >= 1"
    result_dataset = pd.read_parquet(config.data.test_files)

    dataset = RLHFDataset(
        data_path=config.data.test_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.data.prompt_key,
        answer_key=config.data.answer_key,
        image_key=config.data.image_key,
        max_prompt_length=config.data.max_prompt_length,
        truncation="right",
        format_prompt=config.data.format_prompt,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        filter_overlong_prompts=config.data.filter_overlong_prompts,
    )
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=config.data.rollout_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(FSDPWorker), config=config.worker, role='actor_rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    num_batch = len(dataloader)
    output_lst = [[] for _ in range(config.worker.rollout.n)]
    final_answer_lst = [[] for _ in range(config.worker.rollout.n)]
    retrieved_page_lst = [[] for _ in range(config.worker.rollout.n)]
    token_consumption_lst = [[] for _ in range(config.worker.rollout.n)]

    for batch_idx, batch_dict in enumerate(tqdm(dataloader)):
        data = DataProto.from_single_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        for n_sample in range(config.worker.rollout.n):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            final_answers = []
            retrieved_pages = []
            token_consumption = []

            # pydevd_pycharm.settrace('47.76.117.131', port=47508, stdoutToServer=True, stderrToServer=True)
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch['prompts'].shape[-1]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = data_item.batch['responses'][:valid_response_length]
                page_number = data_item.non_tensor_batch['page_ids']
                tc = data_item.non_tensor_batch['token_consumption']
                token_consumption.append(tc)
                retrieved_pages.append(page_number)
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)
                answer_part = extract_answer(response_str)
                if answer_part is not None:
                    try:
                        answer = remove_boxed(last_boxed_only_string(answer_part))
                    except Exception as e:
                        answer = answer_part
                else:
                    answer = 'NULL'
                final_answers.append(answer)


            output_lst[n_sample].extend(output_texts)
            final_answer_lst[n_sample].extend(final_answers)
            retrieved_page_lst[n_sample].extend(retrieved_pages)
            token_consumption_lst[n_sample].extend(token_consumption)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()
    final_answer_lst = np.array(final_answer_lst, dtype=object)
    final_answer_lst = np.transpose(final_answer_lst, axes=(1, 0)).tolist()
    retrieved_page_lst = np.array(retrieved_page_lst, dtype=object)
    retrieved_page_lst = np.transpose(retrieved_page_lst, axes=(1, 0)).tolist()
    token_consumption_lst = np.array(token_consumption_lst)
    token_consumption_lst = np.transpose(token_consumption_lst, axes=(1, 0)).tolist()
    eval_answer_lst = [ans[0] for ans in final_answer_lst]

    # add to the data frame
    result_dataset['responses'] = output_lst
    result_dataset['extracted_answer'] = final_answer_lst
    # result_dataset['retrieved_pages'] = retrieved_page_lst
    result_dataset['ans_ravqa_' + os.path.split(config.data.test_files)[1].split('.parquet')[0]] = eval_answer_lst

    # write to a new parquet
    output_dir = config.trainer.save_checkpoint_path
    os.makedirs(output_dir, exist_ok=True)
    result_dataset_list = datasets.Dataset.from_pandas(result_dataset).to_list()
    json.dump(result_dataset_list, open(os.path.join(output_dir, os.path.split(config.data.test_files)[1].replace('.parquet', '.json')), 'w'))
    json.dump(retrieved_page_lst, open(os.path.join(output_dir, 'retrieved_page_lst.json'), 'w'))
    json.dump(token_consumption_lst, open(os.path.join(output_dir, 'token_consumption.json'), 'w'))


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(GenerationConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    generation_config = OmegaConf.merge(default_config, cli_args)
    generation_config: GenerationConfig = OmegaConf.to_object(generation_config)
    generation_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            }
        }
        ray.init(runtime_env=runtime_env)

    ray.get(main_task.remote(generation_config))


if __name__ == "__main__":
    main()
