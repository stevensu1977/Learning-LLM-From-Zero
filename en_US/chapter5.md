# Chapter 5: Use vLLM build a inference service like openai chatGPT

In this chapter5 We use [vLLM](https://vllm.ai/) build inference service , why we choice vLLM , there have two reasons: 1. vLLM have good performence, you can found  more performance information https://vllm.ai/ , 2.  vLLM can be deployed as a server that mimics the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API.

![chapter5-architecture](../images/chapter5-architecture.png)



#### Step1. Install vLLM and start vLLM Server

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model ./Llama2/models/llama-2-7b-chat-hf

#list models
 curl http://127.0.0.1:8000/v1/models|jq
 {
  "object": "list",
  "data": [
    {
      "id": "./Llama2/models/llama-2-7b-chat-hf",
      "object": "model",
      "created": 1692330491,
      "owned_by": "vllm",
      "root": "./Llama2/models/llama-2-7b-chat-hf",
      "parent": null,
      "permission": [
        {
          "id": "modelperm-ed8520baef03464d8314f1010b48f7ec",
          "object": "model_permission",
          "created": 1692330491,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```



### Step2. Use OpenAI client invoke vLLM API Server

We  can use open client send our prompt to vLLM API Server.

```py
#use openai client send query to vLLM API Server
...
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
...
 completion = openai.Completion.create(
            model="./Llama2/models/llama-2-7b-chat-hf",
            prompt=prompt,
            temperature=0.6,
            max_tokens=2048,
            )
  ...
```



### Step3 start chat coversation 

Start chat coversation

```bash
python chapter5.py --file_path ../../pdf/
```

![10f864d4-83da-4eea-9e59-9e58f00cdbac](/Users/wsuam/Documents/github/Learning-LLM-From-Zero/images/chapter4-screen.jpeg)
