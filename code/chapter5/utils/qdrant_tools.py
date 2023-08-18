# MIT License
#
# Copyright (c) 2023 suwei007@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from typing import Union, Dict, List

EMBEDDING_DIMENSION = 384


def init_collection(collection_name="embedding_with_qdrant",qdrant_url="http://localhost:6333"):
    """
    init Qdrant collection , notice 
    """
    connection = QdrantClient(url=qdrant_url)
    if collection_name is None:
        raise TypeError("collection_name been required!")
     
    create_result=connection.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION,distance=models.Distance.COSINE)
    )
    result=connection.get_collections()
    print(f"init_collection: {collection_name} completed.")
    print("Qdrant collections:")
    for col in result.collections:
        print(f"    {col.name}")
    return create_result

def remove_collection(collection_name=None,qdrant_url="http://localhost:6333"):
    connection = QdrantClient(url=qdrant_url)
    result=connection.delete_collection(collection_name)
    print(result)
    return result

def list_collection(collection_name=None,qdrant_url="http://localhost:6333"):
    connection = QdrantClient(url=qdrant_url)
    result=connection.get_collections()
    return result.collections


def search_from_qdrant(embeddings,collection_name="embedding_with_qdrant",qdrant_url="http://localhost:6333"):
    """
    search from QDrant database
    """
    connection = QdrantClient(url=qdrant_url)
    search_result=connection.search(
            collection_name=collection_name,
            query_vector=embeddings,
            limit=5
        )
    return search_result

def create_qdrant_point(data: Union[str,Dict], embeddings):
    """
    create Qdrant Point from data
    """
    
    point_id =str(uuid.uuid4())
    if isinstance(data,dict):
        payload=data
    else:
        payload={'key':point_id,"desc":data}
    return PointStruct(id=point_id,vector=embeddings,payload=payload)

def insert_points_to_qdrant(points: List[PointStruct], collection_name="embedding_with_qdrant",qdrant_url="http://localhost:6333"):
    """
    insert points to Qdrant
    """
    connection = QdrantClient(url=qdrant_url)
    operation_info= connection.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
    return operation_info
