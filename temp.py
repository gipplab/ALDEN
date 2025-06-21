from PIL import Image
from PIL.Image import Image as ImageObject
from typing import Any, Dict, List, Optional, Union
from io import BytesIO
import math


def process_image(image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
    if (image.width * image.height) > 4194304:
        resize_factor = math.sqrt(4194304 / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
    if (image.width * image.height) < 262144:
        resize_factor = math.sqrt(262144 / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def compuete_image_token_len_batch(example):
    processed_result = [process_image(img) for img in example['image']]
    image_inputs = pp.image_processor(processed_result, return_tensors='pt')
    length = (image_inputs['image_grid_thw'].prod(dim=-1) // pp.image_processor.merge_size ** 2).tolist()
    example['img_token_len'] = length
    return example


def compuete_image_token_len(example):
    processed_result = process_image(example['image'])
    image_inputs = pp.image_processor([processed_result], return_tensors='pt')
    length = image_inputs['image_grid_thw'][0].prod() // pp.image_processor.merge_size ** 2
    example['img_token_len'] = length
    return example

def compute_app_image_token_len(example):
    processed_result = example['image']
    h, w = processed_result.size
    example['img_token_len_app'] = h * w //14**2//4
    return example

from qdrant_client import QdrantClient
from qdrant_client.http import models
collection_name = 'test'
qdrant_client = QdrantClient(
    ":memory:"
)
qdrant_client.recreate_collection(
    collection_name=collection_name,  # the name of the collection
    on_disk_payload=True,  # store the payload on disk
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=100
    ),  # it can be useful to swith this off when doing a bulk upload and then manually trigger the indexing once the upload is done
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ),
    ),
)

batch_size = 128  # Adjust based on your GPU memory constraints

# Use tqdm to create a progress bar
with tqdm(total=len(self.corpus), desc="Indexing Progress") as pbar:
    k = 0
    for i in range(0, len(self.corpus), batch_size):
        image_embeddings = eee[k]
        # Prepare points for Qdrant
        points = []
        for j, embedding in enumerate(image_embeddings):
            # Convert the embedding to a list of vectors
            multivector = embedding.tolist()
            points.append(
                models.PointStruct(
                    id=i + j,  # we just use the index as the ID
                    vector=multivector,  # This is now a list of vectors
                    payload={
                        "source": "internet archive"
                    },  # can also add other metadata/data
                )
            )
        # Upload points to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
        # Update the progress bar
        pbar.update(batch_size)
        k+=1
print("Indexing complete!")