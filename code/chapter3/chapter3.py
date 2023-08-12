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

"""Module is main module use LLM """
import os
import sys
import argparse
from typing import List

import torch
from halo import Halo
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_data_from_pdf,get_text_chunks,build_embedding,search_embedding
#init console spinner
spinner = Halo(text='Agent Thinking......', spinner='dots')



B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\nUse the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer. Don't make up new terms which are not available in the context"""

PROMPT_TEMPLATE="""\
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{context}
{prompt} [/INST]
"""

example_dialogs = [
        {"role": "user", "content": "I am going to Paris, what should I see?"},
       ]

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


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
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

def load_model(model_name=None,interaction=False,pdf_path="",top_k=5):
    """
    Load  LLM (etc. llama2-7b-chat-hf) model though HuggingFace AutoTokenizer, AutoModelForCausalLM
    """
    if model_name is None:
        raise TypeError("Missing required parameter: model_name")
    if pdf_path == "":
        raise TypeError("Missing required parameter: pdf_path")

    print(f"{model_name=}")

    tokenizer = AutoTokenizer.from_pretrained(model_name,legcy=False,use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    if interaction is False:
        print("pdf model only support interaction model")
        return
    #init pdf
    data_from_pdf=load_data_from_pdf(pdf_path=pdf_path)
    pdf_chunks= get_text_chunks(data_from_pdf)
    pdf_embeddings=build_embedding(pdf_chunks)

    while interaction:
        user_input = input("Enter a value (type 'Q' to quit): ")
        if user_input.strip() == "":
            continue
        if user_input in("q","Q"):
            break
        print("User entered:", user_input)
        spinner.start()
        #search embedding
        result=search_embedding(pdf_chunks,pdf_embeddings,[user_input],top_k)

        prompt_dialogs=build_dialogs(user_input,context=result)
        prompt = llama_v2_prompt(prompt_dialogs)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        output = model.generate(
            input_ids,
            max_length=2048,
            temperature=0.6,
            do_sample=True,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
        )
        spinner.stop()
        output_text = tokenizer.decode(output[0],skip_special_tokens=True)
        new_output_text=output_text[output_text.index('[/INST]')+len("[/INST]"):]
        print(new_output_text)

def main():
    """
    main func
    """
    parser=argparse.ArgumentParser(description="This is example load llama2 with HF transformers")
    parser.add_argument('--model_name',required=True,dest='model_name',default=None,help='HF model name or local file path')
    parser.add_argument('--interaction', default=False,action="store_true", help="use agent interaction model")
    parser.add_argument('--pdf_path',required=True,dest='pdf_path',default=None,help='must setup pdf path')
    parser.add_argument('--top_k',default=1,type=int, dest='top_k',help='embedding search top_k, default 1')
    args= parser.parse_args()
    if os.path.isfile(args.pdf_path) is False:
        print(f"File {args.pdf_path} does not exist.")
        return
    # data_from_pdf=load_data_from_pdf(pdf_path=args.pdf_path)
    # pdf_chunks= get_text_chunks(data_from_pdf)
    # pdf_embeddings=build_embedding(pdf_chunks)
    # result=search_embedding(pdf_chunks,pdf_embeddings,['What is SDXL?'],top_k=args.top_k)

    load_model(args.model_name,interaction=True,pdf_path=args.pdf_path)


if __name__ == '__main__':
    main()