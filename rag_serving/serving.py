from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio
from collections import deque
import io
import base64
import os

from flashrag.config import Config
from flashrag.utils import get_retriever

def image_to_bytes(image):
    """将Image列的PIL图片转换为字节"""
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")  # 可根据需要选择格式
        return buffer.getvalue()

app = FastAPI()

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None

def init_retriever(args):
    global retriever_semaphore
    config = Config(args.config)
    gpu_ids = config.gpu_id.split(',')
    retriever_idx = 0
    for id in gpu_ids:
        print(id + '\n')
        device_name = "cuda:" + id
        config["device_name"] = device_name
        for _ in range(2):
            if retriever_idx < args.num_retriever:
                print(f"Initializing retriever {retriever_idx+1}/{args.num_retriever}")
                retriever = get_retriever(config)
                retriever_list.append(retriever)
                available_retrievers.append(retriever_idx)
                retriever_idx += 1
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }

class QueryRequest(BaseModel):
    query: str
    top_n: int = 10
    return_score: bool = False
    id: str

class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: int = 10
    return_score: bool = False
    id: List[str]
    recall_ocr: bool = False
    mm_fetch: bool = False

class Document(BaseModel):
    id: str
    contents: str


@app.post("/fetch", response_model=Union[Tuple[List[Document], List[float]], List[Document]])
async def search(request: QueryRequest):
    query = request.query
    id = request.id

    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )
    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results = retriever_list[retriever_idx].fetch(query, id)
            return [Document(id=result['id'], contents=result['contents']) for result in results]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/search", response_model=Union[Tuple[List[Document], List[float]], List[Document]])
async def search(request: QueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score
    id = request.id

    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].search(query, top_n, return_score, id=id)
                return [Document(id=result['id'], contents=image_to_bytes(result['image'])) for result in results], scores
            else:
                results = retriever_list[retriever_idx].search(query, top_n, return_score, id=id)
                return [Document(id=result['id'], contents=image_to_bytes(result['image'])) for result in results]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/batch_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]], Tuple[List[List[Document]], List[List[Document]]], Tuple[List[List[Document]], List[List[float]], List[List[Document]], List[List[float]]]])
async def batch_search(request: BatchQueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score
    id = request.id
    recall_ocr = request.recall_ocr

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if return_score:
                if not recall_ocr:
                    results, scores = retriever_list[retriever_idx].batch_search(query, id, top_n, return_score,
                                                                                 recall_ocr)
                    return [[Document(id=result['id'],
                                      contents=base64.b64encode(image_to_bytes(result['image'])).decode("utf-8")) for
                             result in results[i]] for i in range(len(results))], scores
                else:
                    results, scores, ocr_results, ocr_scores = retriever_list[retriever_idx].batch_search(query, id,
                                                                                                          top_n,
                                                                                                          return_score,
                                                                                                          recall_ocr)
                    return [[Document(id=result['id'],
                                      contents=base64.b64encode(image_to_bytes(result['image'])).decode("utf-8")) for
                             result in results[i]] for i in range(len(results))], scores, [
                        [Document(id=ocr_result['id'], contents=ocr_result['text']) for ocr_result in ocr_results[i]]
                        for i in range(len(ocr_results))], ocr_scores
            else:
                if not recall_ocr:
                    results = retriever_list[retriever_idx].batch_search(query, id, top_n, return_score, recall_ocr)
                    return [[Document(id=result['id'],
                                      contents=base64.b64encode(image_to_bytes(result['image'])).decode("utf-8")) for
                             result in results[i]] for i in range(len(results))]
                else:
                    results, ocr_results = retriever_list[retriever_idx].batch_search(query, id, top_n, return_score, recall_ocr)
                    return [[Document(id=result['id'],
                                      contents=base64.b64encode(image_to_bytes(result['image'])).decode("utf-8")) for
                             result in results[i]] for i in range(len(results))], [[Document(id=ocr_result['id'],
                                      contents=ocr_result['text']) for
                             ocr_result in ocr_results[i]] for i in range(len(ocr_results))]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/batch_fetch", response_model=List[Document])
async def batch_fetch(request: BatchQueryRequest):
    query = request.query
    id = request.id
    mm_fetch = request.mm_fetch

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results, modals = retriever_list[retriever_idx].batch_fetch(query, id, mm_fetch)
            final_results = []
            for res, md in zip(results, modals):
                if isinstance(res, dict):
                    if md == 'image':
                        final_results.append(Document(id='image' + ', ' + res['id'], contents=base64.b64encode(image_to_bytes(res['image'])).decode("utf-8")))
                    else:
                        final_results.append(Document(id='text' + ', ' + res['id'], contents=res['text']))
                else:
                    final_results.append(Document(id='error', contents=res))
            return final_results
        finally:
            available_retrievers.append(retriever_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="./serving_config.yaml",
        help="path to serving config"
    )
    parser.add_argument(
        "--num_retriever", 
        type=int, 
        default=1,
        help="number of retriever to use, more retriever means more memory usage and faster retrieval speed"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=80,
        help="port to use for the serving"
    )
    args = parser.parse_args()
    
    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

