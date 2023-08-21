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
"""Module is main module use PEFT model test , merge """
import argparse

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_prompt(input_content=""):
    """
    build prompt with instruct template
    """
    return f'''
    ###Human: {input_content}
    '''

def load_peft(peft_model_id=""):
    """
    PEFT model loader
    """
    if peft_model_id=="":
        raise TypeError("PEFT model id have been required")
    config = PeftConfig.from_pretrained(peft_model_id)
    print(peft_model_id ,config.base_model_name_or_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)
    return (model, tokenizer)
    
def load_base_model(peft_model_id):
    """
    base model loader
    """
    config = PeftConfig.from_pretrained(peft_model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model,tokenizer

def llama_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.92,
) -> str:
    """
    Initialize the pipeline
    Uses Hugging Face GenerationConfig defaults
        https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig
    Args:
        model (transformers.AutoModelForCausalLM): Falcon model for text generation
        tokenizer (transformers.AutoTokenizer): Tokenizer for model
        prompt (str): Prompt for text generation
        max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
        temperature (float, optional): The value used to modulate the next token probabilities.
            Defaults to 1.0
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(
        device
    )  # tokenize inputs, load on device

    # when running Torch modules in lower precision, it is best practice to use the torch.autocast context manager.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        response = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded_output = tokenizer.decode(
        response["sequences"][0],
        skip_special_tokens=True,
    ) 
    return decoded_output[len(prompt) :]

def merge(peft_model_id,merge_path="./merged_model"):
    """
    merge PEFT model and base model 
    """
    config = PeftConfig.from_pretrained(peft_model_id)

    print("merge start")
    base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, 
                torch_dtype=torch.float16, 
                load_in_8bit=False, 
                device_map="auto", 
                trust_remote_code=True
    )

    peft_model= PeftModel.from_pretrained(base_model, peft_model_id)
    peft_model.merge_and_unload().save_pretrained(merge_path)
    print("merge completed")


def main():
    """
    main func
    """
    parser = argparse.ArgumentParser("PEFT Model loader")
    parser.add_argument("--model_path",required=True, type=str, help='PEFT model path or HuggingFace Hub model id')
    parser.add_argument("--prompt",type=str,default="",help="input prompt")
    parser.add_argument("--temperature",type=float,default=0.3,help="LLM temperature , default 0.3")
    parser.add_argument("--merge_path",type=str,default="./merged_model",help="merge model path")
    parser.add_argument("--merge",default=False,action="store_true",help="merge to full checkpoint ")
    parser.add_argument("--base_model_test",default=False,action="store_true",help="test base model  ")
    args=parser.parse_args()
    if args.merge:
        merge(args.model_path,args.merge_path)
    else:
        if args.base_model_test:
            #test base model
            model,tokenizer=load_base_model(args.model_path)
            response = llama_generate(
                model,
                tokenizer,
                build_prompt(args.prompt),
                max_new_tokens=85,
                temperature=args.temperature,
            )

            print(response)
            del model
            print("\n")

        model,tokenizer=load_peft(args.model_path)
        response = llama_generate(
            model,
            tokenizer,
            build_prompt(args.prompt),
            max_new_tokens=200,
            temperature=args.temperature,
        )

        print(response)


if __name__=='__main__':
    main()
