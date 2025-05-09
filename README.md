# RAVQA

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

## Document VQA

### Dataset Preprocessing

#### Corpus Building

Change the raw data path and the target path in `rag_serving/build_corpus.py`

```shell
python rag_serving/build_corpus.py
```

#### Image Index Building

```shell
python index_builder.py --retrieval_method vdr-2b-v1 --model_path llamaindex/vdr-2b-v1 --corpus_path /scratch-scc/projects/scc_ulsb_fe/yang/images_corpus/images.parquet --save_dir /scratch-scc/projects/scc_ulsb_fe/yang/images_index --max_length 512 --batch_size 128 --faiss_type Flat --index_modal image --sentence_transformer  --save_embedding
```

### Launch RL

#### Tool Environment Serving

1. Get the IP address of the server

    ```shell
    hostname --ip-address
    ```

2. Start serving

    ```shell
    python rag_serving/serving.py --config rag_serving/serving_config.yaml --num_retriever 4 --port 42354
    ```

#### RL Training



## General VQA
