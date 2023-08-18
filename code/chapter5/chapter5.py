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

import openai

from utils import load_files,search_query
#init console spinner
spinner = Halo(text='Agent Thinking......', spinner='dots')

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

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

def load_model(interaction=False,file_path=".",file_ext_name=".json"):
    """
    Load  LLM (etc. llama2-7b-chat-hf) model though HuggingFace AutoTokenizer, AutoModelForCausalLM
    """
    
    
    #load pdf
    load_files(file_path,file_ext_name)
    

    while interaction:
        user_input = input("Enter a value (type 'Q' to quit): ")
        if user_input.strip() == "":
            continue
        if user_input in("q","Q"):
            break
        print("User entered:", user_input)
        
        #search embedding
        results=search_query( user_input)
        context=""
        for item in results:
            # print(f'Qdrant search result: {item.payload["desc"]}\n')
            context+=item.payload["desc"]
        spinner.start()
        prompt_dialogs=build_dialogs(user_input,context=context)
        prompt = llama_v2_prompt(prompt_dialogs)
        
        completion = openai.Completion.create(
            model="./Llama2/models/llama-2-7b-chat-hf",
            prompt=prompt,
            temperature=0.6,
            max_tokens=2048,
            )
        spinner.stop()
        print(completion["choices"][0]["text"])
         
       

def main():
    """
    main func
    """
    parser=argparse.ArgumentParser(description="This is example load llama2 with HF transformers")
    parser.add_argument('--interaction', default=False,action="store_true", help="use agent interaction model")
    parser.add_argument('--file_path',required=True,default=".",help='must setup pdf path')
    parser.add_argument('--file_ext_name',default=".pdf",help=" default .json , only support .json, .pdf")
    args= parser.parse_args()
    

    load_model(interaction=True,file_path=args.file_path,file_ext_name=args.file_ext_name)


if __name__ == '__main__':
    main()