# Chapter 1: 使用Meta 官方代码运行LLAMA2 模型 

你需要三个步骤在本地或云端的 GPU 机器上运行 LLAMA2, 根据7B/13B需要不同的显卡/显存。

### 步骤1. 下载 LLAMA2-7B-Chat 模型

有很多 LLAMA2 模型，例如 llama2-7b、llama2-7b-chat、llama2-13b、llama2-70b、llama2-13b-chat、llama2-70b-chat，我们只需要：**llama2-7b-chat**。

这些模型之间有什么不同（llama2-7b、llama2-13b、llama2-70b、llama2-7b-chat、llama2-13b-chat、llama2-70b-chat）？

- 7b、13b、70b 表示 70 亿、130 亿、700 亿参数的模型。
- 如果模型名称带有“chat”，例如 llama2-7b-chat，则表示该模型是在聊天完成上进行微调的，与 gpt-3.5 和 ChatGPT 相同。

下载完成后，你可以找到模型权重文件，类似以下目录结构：

```bash
~/Llama2/models/llama-2-7b-chat$ ls -lh
total 13G
drwxrwxr-x 3 ubuntu ubuntu 4.0K Jul 25 01:57 arrow
-rw-rw-r-- 1 ubuntu ubuntu  100 Jul 24 15:26 checklist.chk
-rw-rw-r-- 1 ubuntu ubuntu  13G Jul 24 17:12 consolidated.00.pth
-rw-rw-r-- 1 ubuntu ubuntu  102 Jul 24 15:26 params.json
```



### 步骤2. 克隆 Meta 官方代码仓库并准备 Python 环境

我们使用 meta 官方的示例代码，你需要设置好 Python 环境。

```bash
git clone https://github.com/facebookresearch/llama.git
cd llama

python -m venv .venv 
. .venv/bin/activate
pip install -r requirements.txt
```



### 步骤3 运行 meta 官方推理代码

只需要更改“--ckpt_dir”参数，使用你自己的模型文件夹即可。

```bash

torchrun   --nproc_per_node 1 example_chat_completion.py \
           --ckpt_dir ../Llama2/models/llama-2-7b-chat \
           --tokenizer_path tokenizer.model  \
           --max_seq_len 512 --max_batch_size 4

```

几秒钟后，你可以得到如下结果：

```bash
Loaded in 45.50 seconds
User: I am going to Paris, what should I see?

Assistant: Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.

User: What is so great about #1?

> Assistant:  I'm glad you're interested in learning more about the Eiffel Tower! It's a truly iconic landmark that offers breathtaking views of Paris. Here are some of the reasons why it's so great:
1. Unique Design: The Eiffel Tower is an engineering marvel that was designed by Gustave Eiffel and his team in the late 19th century. Its unique design, with its lattice-like structure and iron beams, makes it stand out from other buildings and landmarks around the world.
2. Historical Significance: The Eiffel Tower was built for the

==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant:  Eiffel Tower high
Love locks on bridges glow
City of art and dreams

==================================

System: Always answer with emojis

User: How to go from Beijing to NY?

> Assistant:  🛬🕰️🌄

==================================

User: I am going to ChengDu, what should I see ? 

> Assistant:  Thank you for asking! Chengdu, the capital city of Sichuan Province in China, is a fascinating destination with a rich cultural heritage and plenty of interesting attractions to explore. Here are some top recommendations for things to see and do while you're there:
1. Giant Panda Research and Breeding Center: Chengdu is home to the Giant Panda Research and Breeding Center, where you can learn about these adorable animals and even get a chance to see them up close.
2. Wide and Narrow Alleys: Take a stroll through the historic Wide and Narrow Alleys, a network of pedestrian-friendly streets lined with traditional teahouses, shops, and restaurants.
3. Jinli Street: Jinli Street is a bustling pedestrian street filled with street performers, vendors, and traditional Sichuan architecture. Don't miss trying some local snacks and drinks here!
4. Mount Qingcheng: If you're interested in nature and hiking, consider visiting Mount Qingcheng, a beautiful mountain with plenty of trails and scenic views.
5. Leshan Giant Buddha: Located about an hour's drive from Chengdu, the Leshan Giant Buddha is a UNESCO World Heritage Site and one of the largest stone Buddhas in the world.
6. Sichuan Museum: If you're interested in learning more about the history and culture of Sichuan Province, the Sichuan Museum is a great place to visit.
7. Huangchengtan Park: This beautiful park is home

==================================
```



### 解释代码的工作原理

1. example_chat_completion.py 使用 llama.Llama.build 函数加载模型。
2. llama/model.py，Meta 在这个文件中为 Llama 创建了原生 PyTorch 网络，这些类都继承自 torch.nn.Module：Transformer、TransformerBlock、FeedForward、Attention、RMSNorm。# TODO 深入研究 LLAMA 网络层
3. llama/tokenizer.py，Tokenizer 类,主要是用于分词(token,包括各种符号)。
4. llama/generation.py ,  

   ```python
   """
   Inference func
   # tokenizer build prompt_tokens from text prommpts
   # temperature 0~1, LLAMA2 temperature refers to a parameter that can be adjusted to control the creativity or novelty of the generated text from the LLAMA2 language model. The temperature value can be set to a value between 0 and 1, When a temperature of 0 is used, the model always generates the same text, while a temperature of 1 results in the most diverse and unpredictable text. 
   In general, a lower temperature value may be preferred for tasks that require more factual or conservative text, such as language translation or summarization. A higher temperature value may be preferred for tasks that require more creative or imaginative text, such as poetry or fiction writing.
   """
   @torch.inference_mode()
   def generate(
           self,
           prompt_tokens: List[List[int]],
           max_gen_len: int,
           temperature: float = 0.6,
           top_p: float = 0.9,
           logprobs: bool = False,
           echo: bool = False,
   ) -> Tuple[List[List[int]], Optional[List[List[float]]]]
   ```

### 总结

#TODO 
