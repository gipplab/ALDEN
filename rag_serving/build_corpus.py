from datasets import Dataset, DatasetDict, Sequence
import os
import tqdm
from PIL import Image
from tqdm import tqdm
import io
import datasets
from datasets import concatenate_datasets
import json

docs_path = '/mnt/vast-nhr/projects/nii00224/raw_data/tmp_image/PaperTab'
ocr_path = '/mnt/vast-nhr/projects/nii00224/raw_data/tmp_text/PaperTab_dsocr'
corpus_path = '/mnt/lustre-grete/usr/u15991/MDocAgent/tmp_hybrid_corpus_latest/PaperTab'

def image_to_bytes(image):
    """将Image列的PIL图片转换为字节"""
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")  # 可根据需要选择格式
        return buffer.getvalue()


def generate_data(data_path: str, doc_id: str):
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        text_path = os.path.join(ocr_path, doc_id, img_name.replace(".jpg", ".txt"))
        img = Image.open(img_path, 'r')
        yield {
            "id": str(img_name.split('.jpg')[0].split('p')[-1]),
            "text": open(text_path).read(),
            "image": image_to_bytes(img)
        }
def main():
    id2idx = {}
    corpus = []
    imag_num = 0
    for doc in tqdm(os.listdir(docs_path)):
        # if os.path.exists(os.path.join(docs_path, doc, 'corpus.parquet')):
        #     os.remove(os.path.join(docs_path, doc, 'corpus.parquet'))
        dt = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join(docs_path, doc), "doc_id": doc})
        # dt = dt.cast_column('image', datasets.Image())
        id2idx[doc] = [0 + imag_num, len(dt) + imag_num]
        imag_num += len(dt)
        corpus.append(dt)
    corpus = concatenate_datasets(corpus)
    corpus.to_parquet(os.path.join(corpus_path, 'images.parquet'))
    with open(os.path.join(corpus_path, 'id2idx.json'), 'w') as f:
        json.dump(id2idx, f)


if __name__ == "__main__":
    main()
