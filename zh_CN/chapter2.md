# Chapter 2: 使用HugginfFace transformers 接口运行LLAMA2模型

### 步骤1. 将 Meta LLAMA2-7B-Chat 官方模型权重转换为 Huggingface 格式

你需要使用转换工具将官方模型权重转换为 Huggingface 格式。如果我们想要转换 Llama2/models/7B 目录下的模型（该脚本仅支持 '7,13,70' 模型，如果你使用 '7'，它将使用 '7B' 作为文件夹名称，因此你需要将 'Llama2/models/llama2-7b-chat' 文件夹重命名为 'Llama2/models/7B'）。

```python
convert_llama_weights_to_hf.py --input_dir Llama2/models --output_dir Llama2/models/llama-2-13b-chat-hf --model_size 7
```



### 步骤2. 使用 transformers 库加载 Llama2 模型

使用 Huggingface transformers 库加载 Llama2 模型，你可以启动它进行一次性推理或使用 `--interaction`进行交互式推理

* 测试一次性推理, 请修改 `--mode_name`  参数指向你的LLAMA2模型路径 

```bash
cd code/chapter2
python chapter2.py --model_name ../../Llama2/models/llama-2-13b-chat-hf

#output
model_name='../Llama2/models/llama-2-13b-chat-hf'
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [01:39<00:00, 33.17s/it]
 Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.  [INST] Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.


```

* 测试交互式推理, 输入 'Q' or 'q' 可以推出交互循环。

```python
python main_hf.py --model_name ../Llama2/models/llama-2-13b-chat-hf --interaction

#output 
model_name='../Llama2/models/llama-2-13b-chat-hf'
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565
Loading checkpoint shards: 100%|████████████████████████████████████| 3/3 [01:53<00:00, 37.97s/it]
Enter a value (type 'Q' to quit): write helloworld  use golang
User entered: write helloworld  use golang
⠹ Agent Thinking......

Hello! As a helpful and knowledgeable AI language model, I can certainly assist with writing "Hello World" using GoLang! Here it is:

package main
import "fmt"
func main() {
    fmt.Println("Hello, World!") // Output: Hello, World!
}

This code will print out "Hello, World!" when run. The `main` package imports the `fmt` package, which provides functions for formatting output. We define a function called `main`, which prints out our greeting message using `fmt.Println()`. Simple enough? Let me know if there's anything else I can help with!

Enter a value (type 'Q' to quit)
```



### 解释代码的工作原理

1. 我们使用 transformers 库中的两个类：AutoTokenizer 和 AutoModelForCausalLM。
2. AutoTokenizer 类。
3. AutoModelForCausalLM 类。
4. 我们使用 'load_model function' 加载 llama2-7b-chat-hf 模型

```python
def load_model(model_name=None):
    """
    this function we local llama2-7b-chat-hf model with HF AutoModelForCausalLM
    """
    if model_name is None:
        raise Exception("need model_name")
        return 
    print(f"{model_name=}")
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    prompt = llama_v2_prompt(dialogs)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    output = model.generate(input_ids, max_length=2048, temperature=0.6,do_sample=True,repetition_penalty=1.3)
    output_text = tokenizer.decode(output[0],skip_special_tokens=True)
    new_output_text=output_text[output_text.index('[/INST]')+len("[/INST]"):]
    print(new_output_text)
```

5.  我们需要准备正确格式的提示，这里有一个 llama_v2_prompt 函数，它可以将对话转换为单个提示。

```python
def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    
    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")
    
    return "".join(messages_list)

```



### 总结

