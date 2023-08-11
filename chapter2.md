# Chapter 2: Run LLAMA2 model  with huggingface transformers

#### Step1. Covert Meta  LLAMA2-7B-Chat  official  model weights to HF format

install huggingface transformers 

```py
pip install transformers
```



You need use convert official model weights to Huggingface format , if we want convert Llam2/models/7B  (this script model_size only support '7,13,70', if you use '7' it would use '7B' as folder name , so you need  changed 'Llama2/models/llama2-7b-chat' folder name to 'Llama2/models/7B' , )

```python
convert_llama_weights_to_hf.py --input_dir Llama2/models --output_dir Llama2/models/llama-2-13b-chat-hf --model_size 7
```



#### Step2. Use transformers libraray load Llama2 model

Use Huggingface transformers library load Llama2 model , you can start it once or use --interaction

* run once 

```bash
python main_hf.py --model_name ../Llama2/models/llama-2-13b-chat-hf


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

* use interaction model,  remember  input 'Q' or 'q' quit while loop.

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
```



#### Explain how the code works 

1. We use transformers libliry two class: AutoTokenizer, AutoModelForCausalLM
2. AutoTokenizer 
3. AutoModelForCausalLM 
4. We load llama2-7b-chat-hf model use 'load_model function'

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

5.  We need prepare prompt  as correct format ,  there have llama_v2_prompt func, it can convert dialogs to single prompt .

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



