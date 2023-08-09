# Chapter 1: Run LLAMA2 model  with python code in 5 minutes

You need three steps to run LLAMA2 on you GPU machine, locally or  Cloud.



#### Step1. Download LLAMA2-7B-Chat model

There have many LLAM2 model etc llama2-7b, llama2-7b-chat, we only need: **llama2-7b-chat**

What's  different between those model (llama2-7b,  llama2-13b, llama2-70b, llama2-7b-chat,  llama2-13b-chat, llama2-70b-chat)?

* 7b, 13b, 70b means  7 billion , 13 billions, 70 billion parameter model
* llama2-7b-chat, if with "chat", means model fine-tuned on chat completions , it's same like gpt-3.5 and chatGPT.

After download completed , you could found model weights like this :

```bash
~/Llama2/models/llama-2-7b-chat$ ls -lh
total 13G
drwxrwxr-x 3 ubuntu ubuntu 4.0K Jul 25 01:57 arrow
-rw-rw-r-- 1 ubuntu ubuntu  100 Jul 24 15:26 checklist.chk
-rw-rw-r-- 1 ubuntu ubuntu  13G Jul 24 17:12 consolidated.00.pth
-rw-rw-r-- 1 ubuntu ubuntu  102 Jul 24 15:26 params.json
```



#### Step2. Git clone meta offically code and prepare python environment

We use meta offical example code , you need setup your python enviroments.

```bash
git clone https://github.com/facebookresearch/llama.git
cd llama

python -m venv .venv 
. .venv/bin/activate
pip install -r requirements.txt
```



#### Step3 Run Meta offically example code

Just change "--ckpt_dir" parameter, use your model floder

```bash

torchrun   --nproc_per_node 1 example_chat_completion.py \
           --ckpt_dir ../Llama2/models/llama-2-7b-chat \
           --tokenizer_path tokenizer.model  \
           --max_seq_len 512 --max_batch_size 4

```

After some seconds ,  you can get result like this :

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



#### Explain how the code works 

1. example_chat_completion.py use llama.Llama.build function load model
2. llama/model.py, in this file Meta create native pytorch network for Llama,   those class inherited from torch.nn.Module:    Transformer, TransformerBlock,FeedForward, Attention, RMSNorm.   

 #TODO Dive Deep LLAMA network layer .



#### Summary

#TODO 
