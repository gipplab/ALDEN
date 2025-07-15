from typing import List, Union
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from flashrag.retriever.utils import load_model, pooling, parse_query, parse_image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from PIL import Image
import torch
import math


class Encoder:
    """
    Encoder class for encoding queries using a specified model.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        pooling_method (str): The method used for pooling.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.
        instruction (str): Additional instructions for parsing queries.

    Methods:
        encode(query_list: List[str], is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
    """

    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, instruction):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.gpu_num = torch.cuda.device_count()
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.inference_mode()
    def single_batch_encode(self, query_list: Union[List[str], str], is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__ or (isinstance(self.model, torch.nn.DataParallel) and "T5" in type(self.model.module).__name__):
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            pooler_output = output.get('pooler_output', None)
            last_hidden_state = output.get('last_hidden_state', None)
            query_emb = pooling(
                pooler_output, last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
        if "dpr" not in self.model_name:
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb

    @torch.inference_mode()
    def encode(self, query_list: List[str], batch_size=64, is_query=True) -> np.ndarray:
        print('-------------encode----------------')
        print(query_list)
        print('--------------------------------')
        query_emb = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: "):
            query_emb.append(self.single_batch_encode(query_list[i : i + batch_size], is_query))
        query_emb = np.concatenate(query_emb, axis=0)
        return query_emb

    @torch.inference_mode()
    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=64, is_query=True) -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        query_emb = self.encode(query_list, batch_size, is_query)
        return query_emb


class ColEncoder:
    """
    Encoder class for encoding queries using a specified model.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        pooling_method (str): The method used for pooling.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.
        instruction (str): Additional instructions for parsing queries.

    Methods:
        encode(query_list: List[str], is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
    """

    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, instruction):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.gpu_num = torch.cuda.device_count()
        self.model = ColQwen2_5.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  # or "mps" if on Apple Silicon
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        self.tokenizer = ColQwen2_5_Processor.from_pretrained(model_path)

    @torch.inference_mode()
    def encode(self, query_list: List[str], batch_size=64, is_query=False, modal=None) -> np.ndarray:
        if is_query:
            query_batch = self.tokenizer.process_queries(query_list).to(self.model.device)
        else:
            query_batch = self.tokenizer.process_images(query_list).to(self.model.device)

        query_emb = self.model(**query_batch)
        query_emb = query_emb.detach().to(torch.float).cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb

    @torch.inference_mode()
    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=64, is_query=True) -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        query_emb = self.encode(query_list, batch_size, is_query)
        return query_emb


class STEncoder:
    """
    STEncoder class for encoding queries using SentenceTransformers.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.
        instruction (str): Additional instructions for parsing queries.

    Methods:
        encode(query_list: List[str], batch_size=64, is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
        multi_gpu_encode(query_list: List[str], is_query=True, batch_size=None) -> np.ndarray:
            Encodes a list of queries into embeddings using multiple GPUs.
    """

    def __init__(self, model_name, model_path, max_length, use_fp16, instruction):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.model = SentenceTransformer(
            model_path, device="cuda", trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2"
            }
        )#"device_map": "cuda:0",

    @torch.inference_mode()
    def encode(self, query_list, batch_size=64, is_query=True, modal='') -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query) if not self.model_name == 'vdr-2b-v1' else query_list
        query_emb = self.model.encode(
            query_list, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
        )
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb

    @torch.inference_mode()
    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=None, is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)
        pool = self.model.start_multi_process_pool()
        query_emb = self.model.encode_multi_process(
            query_list,
            pool,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        self.model.stop_multi_process_pool(pool)
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb

class VDREncoder:
    def __init__(self, model_name, model_path, max_length, use_fp16, instruction):
        self.model_name = model_name
        self.max_pixels = 768 * 28 * 28
        self.min_pixels = 1 * 28 * 28
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )
        self.model.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        self.document_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"
        self.query_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>"

    @torch.inference_mode()
    def encode_query(self, queries: list[str], batch_size=64, is_query=True, modal='') -> np.ndarray:
        dummy_image = Image.new('RGB', (56, 56))
        inputs = self.processor(
            text=[self.query_prompt % x for x in queries],
            images=[dummy_image for _ in queries],
            videos=None,
            padding='longest',
            return_tensors='pt'
        ).to('cuda')

        cache_position = torch.arange(0, len(queries))
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False)

        with torch.no_grad():
            output = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True
            )

        embeddings = output.hidden_states[-1][:, -1]
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1).cpu().numpy()

    @torch.inference_mode()
    def encode(self, documents: list[Image.Image]):
        def round_by_factor(number: float, factor: int) -> int:
            return round(number / factor) * factor

        def ceil_by_factor(number: float, factor: int) -> int:
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: float, factor: int) -> int:
            return math.floor(number / factor) * factor

        def smart_resize(height: int, width: int) -> tuple[int, int]:
            h_bar = max(28, round_by_factor(height, 28))
            w_bar = max(28, round_by_factor(width, 28))
            if h_bar * w_bar > self.max_pixels:
                beta = math.sqrt((height * width) / self.max_pixels)
                h_bar = floor_by_factor(height / beta, 28)
                w_bar = floor_by_factor(width / beta, 28)
            elif h_bar * w_bar < self.min_pixels:
                beta = math.sqrt(self.min_pixels / (height * width))
                h_bar = ceil_by_factor(height * beta, 28)
                w_bar = ceil_by_factor(width * beta, 28)
            return w_bar, h_bar

        def resize(image: Image.Image):
            new_size = smart_resize(image.height, image.width)
            return image.resize(new_size)

        inputs = self.processor(
            text=[self.document_prompt] * len(documents),
            images=[resize(x) for x in documents],
            videos=None,
            padding='longest',
            return_tensors='pt'
        ).to('cuda')

        cache_position = torch.arange(0, len(documents))
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False)

        with torch.no_grad():
            output = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True
            )

        embeddings = output.hidden_states[-1][:, -1]
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1).cpu().numpy()

class ClipEncoder:
    """ClipEncoder class for encoding queries using CLIP."""

    def __init__(self, model_name, model_path):

        self.model_name = model_name
        self.model_path = model_path
        self.load_clip_model()

    def load_clip_model(self):
        from transformers import AutoModel, AutoProcessor

        with open(os.path.join(self.model_path, "config.json")) as f:
            config = json.load(f)
        model_type = config.get("architectures", [None])[0]
        self.model_type = model_type

        if model_type == "CLIPModel" or model_type == "ChineseCLIPModel":
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            # set model max length for chineseclipmodel
        elif model_type.endswith("CLIPModel"):
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.processor = None
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")

        self.model.eval()
        self.model.cuda()

        # set model max length for model that not specified in config.json
        if self.processor is not None and self.processor.tokenizer.model_max_length > 100000:
            try:
                model_max_length = config['text_config']['max_position_embeddings']
            except:
                model_max_length = 512
            self.processor.tokenizer.model_max_length = model_max_length    


    @torch.inference_mode()
    def single_batch_encode(self, query_list: Union[List[str], str], modal="image") -> np.ndarray:
        encode_func_dict = {
            "text": self.encode_text,
            "image": self.encode_image,
        }
        return encode_func_dict[modal](query_list)

    @torch.inference_mode()
    def encode(self, query_list: List[str], batch_size=64, modal="image") -> np.ndarray:
        if not isinstance(query_list, list):
            query_list = [query_list]
        query_emb = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: "):
            query_emb.append(self.single_batch_encode(query_list[i : i + batch_size], modal))
        query_emb = np.concatenate(query_emb, axis=0)
        return query_emb

    @torch.inference_mode()
    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=64, is_query=True) -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        query_emb = self.encode(query_list, batch_size, is_query)
        return query_emb

    @torch.inference_mode()
    def encode_image(self, image_list: List) -> np.ndarray:
        # Each item in image_list: PIL Image, local path, or URL
        if self.model_type == "CLIPModel" or self.model_type == 'ChineseCLIPModel':
            # need handle image
            image_list = [parse_image(image) for image in image_list]
            inputs = self.processor(images=image_list, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            image_emb = self.model.get_image_features(**inputs)
            image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
            image_emb = image_emb.detach().cpu().numpy().astype(np.float32)
        elif self.model_type.endswith("CLIPModel"):
            image_emb = self.model.encode_image(image_list)
        else:
            raise NotImplementedError(f"Unsupported model type: {self.model_type}")
        return image_emb #ndarray bsz * dim

    @torch.inference_mode()
    def encode_text(self, text_list: List[str]) -> np.ndarray:
        # Each item in image_list: PIL Image, local path, or URL
        if self.model_type == "CLIPModel" or self.model_type == 'ChineseCLIPModel':
            inputs = self.processor(
                text=text_list,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            text_emb = self.model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            text_emb = text_emb.detach().cpu().numpy().astype(np.float32)
        elif self.model_type.endswith("CLIPModel"):
            text_emb = self.model.encode_text(
                text_list,
                padding=True,
                truncation=True,
            )
        else:
            raise NotImplementedError(f"Unsupported model type: {self.model_type}")
        return text_emb
