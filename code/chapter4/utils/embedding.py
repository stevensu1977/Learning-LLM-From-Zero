import os
import json

from typing import List,Union



from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from sentence_transformers import SentenceTransformer, util


from .qdrant_tools import init_collection,remove_collection,list_collection,create_qdrant_point, insert_points_to_qdrant,search_from_qdrant

#load embedding model, you can choice your embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def load_files(folder_path,file_ext_name: str):
    if file_ext_name not in [".json",".pdf"]:
        raise TypeError(f"Not support {file_ext_name}")
    json_files = [f for f in os.listdir(folder_path) if f.endswith(file_ext_name)]
    init_collection()
    for file_name in json_files:
        print(f"load {file_name}")
        if file_ext_name == ".json":
            load_json_to_qdrant(folder_path+"/"+file_name)
        elif file_ext_name == ".pdf":
            print(f"{file_ext_name=}")
            pdf_text=load_data_from_pdf(folder_path+"/"+file_name)
            pdf_chunks=get_text_chunks(pdf_text)
            points=[]
            for chunk in pdf_chunks:
                chunk_embeddings=build_embedding(chunk.replace("\n",""))
                points.append(create_qdrant_point(chunk.replace("\n",""),chunk_embeddings))
            operation_info=insert_points_to_qdrant(points)
            print(f"insert embedding to Qdrant, {operation_info.status}")

    

def load_json_to_qdrant(file_name=""):
    if file_name=="":
        raise TypeError("file_name been required")
    if os.path.exists(file_name) is False:
        print(f"{file_name} not exists")
        return 
    
    with open(file_name) as input_file:
        data=json.load(input_file)
        points=[]
        for item in data:
            embedding_data=build_embedding(item["desc"])
            points.append(create_qdrant_point(item,embedding_data))
        print(f"row: {len(points)}")

        operation_info=insert_points_to_qdrant(points)

        print(f"insert embedding to Qdrant, {operation_info.status}")



def load_data_from_pdf(pdf_path=""):
    """
    Load data from pdf
    """
    if pdf_path=="":
        raise TypeError("pdf_path benn required")
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader= PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Split PDF text to chunks 
    """
    text_splitter= CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def build_embedding(chunks: List[str]):
    """
    Build embedding data from chunks
    """
    return embedder.encode(chunks, convert_to_tensor=False).tolist()

def search_query(query: str):
    query_embedding = embedder.encode(query,  convert_to_tensor=False).tolist()
    return search_from_qdrant(query_embedding)
    


def search_embedding(pdf_chunks, pdf_embeddings,queries: List[str]=['What is SDXL?'],top_k=5):
    """
    search embedding data with query list and return top X
    """
    if pdf_embeddings is None:
        raise TypeError("need build pdf embeddings first!")

    """
        # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
        hits = hits[0]      #Get the hits for the first query
        for hit in hits:
            print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """
    # Query sentences:
    
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        hits = util.semantic_search(query_embedding, pdf_embeddings, top_k=top_k)
        hits = hits[0]      #Get the hits for the first query
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        result = []
        for hit in hits:
            result.append(pdf_chunks[hit['corpus_id']].replace("\n",""))
        return "".join(result)
        

