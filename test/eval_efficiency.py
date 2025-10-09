################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import os
import sys
import torch
import gc
from termcolor import colored
from argparse import ArgumentParser, Namespace

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
os.chdir(root_dir)

from data.dataset import Dataset
from models import choose_model_class

dataset_name = "ruler/qa_2"


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="ckpt/Llama-3-8B-Instruct-Gradient-1048k")
    p.add_argument("--head_classification_path", type=str, default="select_headmask/llama3")
    p.add_argument("--datalen", type=int, default=500*1024)
    p.add_argument("--method", type=str, default="CateKV") 
    p.add_argument("--cv_threshold", type=float, default=0.4, help="Ratio of adaptive heads") 
    p.add_argument("--full_fraction", type=float, default=1.0, help="Retention ratio in adaptive head")
    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()

    model_name = args.model_name
    length = args.datalen
    method = args.method
    head_classification_path = args.head_classification_path
    cv_threshold = args.cv_threshold # r
    full_fraction = args.full_fraction # eta

    sparse_budget = 2048
    temperature = 1.0
    baseline_bsz = 1 # baseline batchsize
    cv_bsz = 1 # catekv batchsize

    chunk_size = 8
    last_q = 64
    init_tokens = sparse_budget // 32
    recent_tokens = sparse_budget // 8
    minference = False

    dataset_maxlen = 122 * 1024

    ##################### Baseline #####################
    LLM = choose_model_class(model_name)
    llm_baseline = LLM(
        model_name=model_name,
        head_classification_path=head_classification_path,
        batch_size=baseline_bsz,        
        device='cuda:0',
        max_length=length + 1024,
        attn_mode='full',                
        sparse_budget=sparse_budget,
        chunk_size=chunk_size,
        last_q=last_q,
        init_tokens=init_tokens,
        recent_tokens=recent_tokens,
        cv_threshold=cv_threshold,
        full_fraction=full_fraction,
        minference=minference
    )
    dataset = Dataset(dataset_name, llm_baseline.tokenizer, 128*1024, 50)
    total_sample_needed = length // dataset_maxlen
    remaining_length = length % dataset_maxlen
    input_ids_list = []
    for i in range(total_sample_needed):
        input_ids_list.append(dataset[i][0][:,:dataset_maxlen])
    if remaining_length > 0:
        input_ids_list.append(dataset[total_sample_needed][0][:,:remaining_length])
    input_ids = torch.cat(input_ids_list,dim=1)

    assert input_ids.shape[-1] == length

    _, throughput_baseline, peak_memory_baseline = llm_baseline.batch_generate(
        input_ids.to(llm_baseline.device),
        gen_len=100,
        benchmark=True,
        temperature=temperature
    )
    print(colored(f"[Baseline] Throughput: {throughput_baseline} tokens/s", 'red'))

    del llm_baseline.kv_cache
    del llm_baseline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    ################# catekv #####################

    llm_cv = LLM(
        model_name=model_name,
        head_classification_path=head_classification_path,
        batch_size=cv_bsz,            
        device='cuda:0',
        max_length=length + 1024, 
        attn_mode=method,            
        sparse_budget=sparse_budget,
        chunk_size=chunk_size,
        last_q=last_q,
        init_tokens=init_tokens,
        recent_tokens=recent_tokens,
        cv_threshold=cv_threshold,
        full_fraction=full_fraction,
        minference=minference
    )
    dataset = Dataset(dataset_name, llm_cv.tokenizer, 128 * 1024, 10)
    total_sample_needed = length // dataset_maxlen
    remaining_length = length % dataset_maxlen
    input_ids_list = []
    for i in range(total_sample_needed):
        input_ids_list.append(dataset[i][0][:,:dataset_maxlen])
    if remaining_length > 0:
        input_ids_list.append(dataset[total_sample_needed][0][:,:remaining_length])
    input_ids = torch.cat(input_ids_list,dim=1)

    _, throughput_cv, peak_memory_cv = llm_cv.batch_generate(
        input_ids.to(llm_cv.device),
        gen_len=1000,
        benchmark=True,
        temperature=temperature
    )
    print(colored(f"[cv] Throughput: {throughput_cv} tokens/s", 'red'))
    
    # speed up
    print(colored(f"Speedup: {throughput_cv / throughput_baseline:.2f}x", 'red'))
    print(colored(f"Memory Reduction: {peak_memory_baseline / peak_memory_cv:.2f}x", 'red'))
