# Chapter 3: Embedding your PDF and build context with it and send to LLM

We should reused chatper2 source code ,  we write some new code , it's can read pdf and embedding content ,  when we input prompt , first search pdf content and build prompt send to LLM. In this chapter we not use vector database store multiple pdf, only use some simple code .

![image](../images/chapter3-architecture.png)

### Try

We download SDXL arxiv paper pdf , and run code , may be code it's not perfect but work :p . 

```bash
curl -o ./2307.01952.pdf -L https://arxiv.org/pdf/2307.01952
cd code/chapter3
python chapter3.py --model_name ../Llama2/models/llama-2-13b-chat-hf --interaction --pdf_path ../../2307.01952.pdf

#output example 
model_name='../Llama2/models/llama-2-13b-chat-hf'
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████| 3/3 [02:02<00:00, 40.95s/it]
Enter a value (type 'Q' to quit): what is SDXL?
User entered: what is SDXL?
⠹ Agent Thinking......
  Hello! I'd be happy to help answer your questions about SDXL.

First, I want to clarify that SDXL stands for "Self-Attention based Denoising Diffusion Model," which is a type of generative text-image foundation model that uses self-attention mechanisms and denoising diffusion processes to improve the quality of generated images. It is designed to address some limitations of traditional image generation models, such as opaqueness and limited interpretability.

Regarding your first question, SDXL is not a specific pre-trained model like ResNet or Transformer. Instead, it is a general framework that includes several components, including a text encoder, an image generator, and a self-attention mechanism. The key idea behind SDXL is to use self-attention to allow the model to focus more effectively on certain parts of the input text when generating corresponding images.

As for your second question, yes, SDXL has been specifically designed to handle long-range dependencies in both text and image modalities. In order to achieve this, SDXL employs a multi-resolution approach that combines low-resolution and high-resolution features to capture both local and global contextual information. Additionally, the self-attention mechanism allows the model to selectively focus on different regions of the input data when generating images.

I hope this helps! Let me know if you have any other questions or need further clarification.
Enter a value (type 'Q' to quit): 
```





### Explain how the code works 

Load pdf, embedding it ,  search embedding data with query , build context into prompt , send it to LLM.

#### Step1. Load your PDF (utils/embedding.py)

We use PyPDF2 load our PDF, read pdf and split it to chunks.

```python
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
```

#### Step2. Embedding your PDF content (utils/embedding.py)

We use sentence_transformers library process embedding. 

```python
#load embedding model, you can choice your embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
        
```



####  Step3. Combined all togther ,  search embedding data and build prompt context, send to LLM 

```python
#when we input some query , etc. What is SDXL, we need search your input in embedding data 
#....
def load_model(model_name=None,interaction=False,pdf_path="",top_k=5):
  ...
	data_from_pdf=load_data_from_pdf(pdf_path=pdf_path)
	pdf_chunks= get_text_chunks(data_from_pdf)
	pdf_embeddings=build_embedding(pdf_chunks)
  ...
  while interaction:
    ...
    #notice there have differents with chapter2.py , we search embeddings and build_dialogs with search result
    result=search_embedding(pdf_chunks,pdf_embeddings,[user_input],top_k)
    prompt_dialogs=build_dialogs(user_input,context=result)
    ...
  
  
def build_dialogs(prompt: str,context=""):
    """
    Build dialogs from simple text
    """
    if context!="":
        return  [
        {"role": "user", "content": f"Context:\n{context}\nUser: {prompt}"},
       ]
    return  [
        {"role": "user", "content": prompt},
       ]
```





