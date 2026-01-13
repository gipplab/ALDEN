# ALDEN: Agentic Long-document Document Intelligence

**ALDEN** is a multi-modal reinforcement learning framework designed for **Agentic Visually-Rich Document Understanding (A-VRDU)**. Built upon [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), it introduces a novel **fetch** action, cross-level reward and a visual semantic anchoring mechanism to enable efficient navigation and reasoning over long, high-resolution documents.

This repository contains the official implementation of our paper: **[ALDEN: Reinforcement Learning for Active Navigation and Evidence Gathering in Long Documents.](https://arxiv.org/pdf/2510.25668)**.

## 🛠️ Installation

### Installing the Training Environment

```shell
conda create -n alden python=3.10
conda activate alden
git clone https://github.com/gipplab/ALDEN.git
cd ./ALDEN
pip install -e .
```

### Installing the Single-Vector Retriever Environment

```shell
conda create -n alden-sv python=3.10
cd ./ALDEN
pip install -r single-vec_retriever_requirements.txt
cd ./flashrag
pip install -e .
```

### Installing the Multi-vector Retriever Environment

```shell
conda create -n alden-mv python=3.10
cd ./ALDEN
pip install -r multi-vec_retriever_requirements.txt
cd ./flashrag
pip install -e .
```

## 📂 Dataset Preprocessing

### 1. Corpus Building

We provide the processed training corpus on Hugging Face: **[SkyFishQ/ALDEN](https://www.google.com/search?q=https://huggingface.co/SkyFishQ/ALDEN)**.

If you wish to build the corpus from scratch using your own data:

1. Modify the `raw_data_path` and `target_path` in `rag_serving/build_corpus.py`.
2. Run the build script:

```bash
python rag_serving/build_corpus.py
```

### 2. Image Index Building

We use `flashrag` to build the dense retrieval index for document images.

```bash
cd ./flashrag/flashrag/retriever

python index_builder.py \
    --retrieval_method vdr-2b-v1 \
    --model_path llamaindex/vdr-2b-v1 \
    --corpus_path /path/to/your/images_corpus/images.parquet \
    --save_dir /path/to/save/images_index \
    --max_length 512 \
    --batch_size 128 \
    --faiss_type Flat \
    --index_modal image \
    --sentence_transformer \
    --save_embedding
```

```bash
python index_builder.py \
	--retrieval_method gte-Qwen2-1.5B-instruct \
	--model_path Alibaba-NLP/gte-Qwen2-1.5B-instruct \
	--corpus_path /path/to/your/images_corpus/images.parquet \
	--save_dir /path/to/save/images_index \
	--max_length 4096 \
	--batch_size 128 \
	--faiss_type Flat \
	--index_modal text \
	--sentence_transformer \
	--save_embedding
```

```bash
python index_builder.py 
	--retrieval_method jina-colbert-v2 \
	--model_path jinaai/jina-colbert-v2 \
	--corpus_path /path/to/your/images_corpus/images.parquet \ 
	--save_dir /path/to/save/images_index \ 
	--max_length 4096 \
	--batch_size 128 \
	--faiss_type Flat \
	--index_modal text \
	--save_embedding
```

```bash
python index_builder.py 
	--retrieval_method colqwen2-v1.0 \
	--model_path vidore/colqwen2-v1.0 \
	--corpus_path /path/to/your/images_corpus/images.parquet \ 
	--save_dir /path/to/save/images_index \
	--max_length 4096 \
	--batch_size 128 \
	--faiss_type Flat \
	--index_modal image \
	--save_embedding
```

*Note: Please replace `/path/to/your/...` with your actual file paths.*

## 🚀 Launch RL Training

ALDEN uses a decoupled architecture where the environment (RAG tools) and the agent (RL training) run separately.

### Step 1: Tool Environment Serving

First, launch the RAG environment server which handles the `<search>` and `<fetch>` actions.

1. **Get the Server IP:**

    ```bash
    hostname --ip-address
    ```

    *Take note of this IP address, you will need to configure it in the training script.*

2. **Start the Service:**

    ```bash
    python rag_serving/serving.py \
        --config rag_serving/serving_config_single-vec.yaml \
        --num_retriever 8 \
        --port 42354
    ```

    or 

    ```bash
    python rag_serving/serving.py \
        --config rag_serving/serving_config_multi-vec.yaml \
        --num_retriever 8 \
        --port 42354
    ```

### Step 2: RL Training

Once the tool server is running, start the training. Ensure the server URL in the training script points to the IP obtained in Step 1.

```bash
bash examples/baselines/qwen2_5_vl_7b_doc_agent_ppo.sh
```

## ⚡ Inference

To run inference on test sets:

```bash
bash examples/baselines/qwen2_5_vl_7b_doc_agent_generation.sh
```

## 💾 Model Utils

### Merge Checkpoints in the Hugging Face Format

```bash
python3 scripts/model_merger.py \
    --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## 📜 Citation

If you find this project useful, please cite our paper:

```
@article{yang2025alden,
  title={ALDEN: Reinforcement Learning for Active Navigation and Evidence Gathering in Long Documents},
  author={Yang, Tianyu and Ruas, Terry and Tian, Yijun and Wahle, Jan Philip and Kurzawe, Daniel and Gipp, Bela},
  journal={arXiv preprint arXiv:2510.25668},
  year={2025}
}
```

## 🙌 Acknowledgements

This work is built upon the following excellent open-source projects:

- [EasyR1](https://github.com/hiyouga/EasyR1): For the RL infrastructure.
- [VAGEN](https://github.com/mll-lab-nu/VAGEN): For visual agent baselines.
- [verl](https://github.com/volcengine/verl): For efficient RL training.
- [ReCall](https://github.com/Agent-RL/ReCall): For RAG integration concepts.

We greatly appreciate their valuable contributions to the community.
