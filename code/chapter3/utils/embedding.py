
from typing import List



from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from sentence_transformers import SentenceTransformer, util

#load embedding model, you can choice your embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')


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
    return embedder.encode(chunks, convert_to_tensor=True)


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
        

