# Chapter 4: 使用Qdrant 存储embedding数据 

在第三章，我们已经展示了如何嵌入PDF文本并使用句子转换器搜索它，但如果我们有更多的PDF，需要嵌入更多的上下文并搜索它，我们该怎么做呢？答案是使用 `向量数据库`。你可以在这个链接中找到更多信息 [什么是向量数据库](https://zilliz.com/learn/what-is-vector-database)。在本章中，我们将向你展示如何使用向量数据库（我们选择Qdrant存储我们的嵌入），用Qdrant 替换sentence transformers 的内存搜索。

![chapter4-architecture](/Users/wsuam/Documents/github/Learning-LLM-From-Zero/images/chapter4-architecture.png)



#### Step1. 启动Qdrant 

我们需要先安装docker ,然后通过docker 启动Qdrant。

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```



### Step2. Qdrant python编程

在使用Qdrant Python客户端之前，我们需要创建集合(collection)，`recreate_collection`总是创建新的集合，如果名称存在，则Qdrant会删除它并创建新的集合，因此在生产环境中使用时需要非常小心。

请注意，我们使用`recreate_collection`，所以每次启动`chapter4.py`时，你都会得到相同名称的新集合，如果你不知道它的工作原理，请不要在生产环境中使用`recreate_collection`函数。

```py
#step1 create vector collection
connection.recreate_collection(
     collection_name="embedding_with_qdrant",
     vectors_config=models.VectorParams(size=256,distance=models.Distance.COSINE)
)
```

然后我们使用`sentence_transformers`嵌入我们的文本并创建`PointStuct`，将其放入Qdrant集合中。

```python
#load embedding model, you can choice your embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder.encode(your_input, convert_to_tensor=False).tolist()

#create Qdrant PointStruct and put it into collection
point=PointStruct(id=point_id,vector=embeddings,payload=payload)

operation_info= connection.upsert(
            collection_name=collection_name,
            wait=True,
            points=[point]
        )
```



### Step3  升级chatper3.py 使用Qdrant存储嵌入数据

现在，我们可以将`chatper3.py`重写为`chapter4.py`，使用Qdrant存储我们的嵌入数据，你可以在`qdrant_tools.py`和`embedding.py`中找到。



```bash
#now you can load pdf or json from folder , just setup --file_path, the whole in the path data should be loaded.
python chapter4.py  --model_name ../../../Llama2/models/llama-2-13b-chat-hf --interaction --file_path ../../pdf --file_ext_name .pdf
```

![10f864d4-83da-4eea-9e59-9e58f00cdbac](/Users/wsuam/Documents/github/Learning-LLM-From-Zero/images/chapter4-screen.jpeg)
