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
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import warnings
warnings.filterwarnings("ignore")

import torch
import gc
from termcolor import colored
from argparse import ArgumentParser, Namespace

import torch.distributed as dist
import datetime

class DistConfig:
    def __init__(self, is_distributed, rank, world_size, device, master_process):
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_process = master_process

def init_dist():
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=60*90))
        world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{rank}" 
        torch.cuda.set_device(device)
        master_process = (
            rank == 0
        )
    else:
        device = "cuda:0"
        world_size = 1
        master_process = True

    if master_process:
        print(colored(f"[Dist init] world_size={world_size}", 'cyan'))
    
    return DistConfig(is_distributed, rank, world_size, device, master_process)

def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="ckpt/Llama-3-8B-Instruct-Gradient-1048k")
    p.add_argument("--head_classification_path", type=str, default="select_headmask/llama3")
    p.add_argument("--dataset_name", type=str_to_list, default=["ruler/niah_single_2"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=8*1024, help="The length of the context.")
    p.add_argument("--method", type=str, default="full")
    p.add_argument("--sparse_budget", default=2048)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--last_q", type=int, default=64)
    p.add_argument("--init_tokens", type=int, default=64)
    p.add_argument("--recent_tokens", type=int, default=256)
    p.add_argument("--top_threshold", type=float, default=1.0)
    p.add_argument("--top_fraction", type=float, default=0.995)
    p.add_argument("--cv_threshold", type=float, default=0.5)
    p.add_argument("--full_fraction", type=float, default=1.0)
    p.add_argument("--minference", action='store_true', default=False)

    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    dataset_names = args.dataset_name
    num_samples = args.num_samples
    datalen = args.datalen
    sparse_budget = args.sparse_budget
    dtype = torch.bfloat16
    rank = args.rank
    chunk_size = args.chunk_size
    minference = args.minference
    last_q = args.last_q
    init_tokens = args.init_tokens
    recent_tokens = args.recent_tokens
    top_threshold = args.top_threshold
    top_fraction = args.top_fraction
    cv_threshold = args.cv_threshold
    full_fraction = args.full_fraction
    head_classification_path = args.head_classification_path

    dist_config = init_dist()
    
    from evaluator import Evaluator
    from models import choose_model_class
    from data.dataset import Dataset
    
    evaluator = Evaluator(dist_config)
    
    if dist_config.master_process:
        print(colored(f"data_names: {dataset_names}", 'cyan'))
    
    LLM = choose_model_class(model_name)

    llm = LLM(model_name=model_name, head_classification_path=head_classification_path, batch_size=batch_size, device=dist_config.device, max_length=datalen+1024, attn_mode=args.method, dtype=dtype, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size,last_q=last_q,init_tokens=init_tokens,recent_tokens=recent_tokens,top_threshold=top_threshold,top_fraction=top_fraction,cv_threshold=cv_threshold,full_fraction=full_fraction,minference=minference)

    if dist_config.master_process:
        llm.print_kv_stats()

    for dataset_name in dataset_names:
        dataset = Dataset(dataset_name, llm.tokenizer, datalen, num_samples, evaluator.dist_config.rank, evaluator.dist_config.world_size)
        if args.method == 'CateKV_dynamic':
            if minference == True:
                evaluator.test(llm, dataset, f"archive/{model_name.split('/')[-1]}/{dataset_name}_{datalen}_{args.method}_{sparse_budget}_{rank}_{chunk_size}_top{top_threshold}_lastq{last_q}_init{init_tokens}_recent{recent_tokens}_cv{cv_threshold}_minfernece.jsonl", args.method)
            else:
                evaluator.test(llm, dataset, f"archive/{model_name.split('/')[-1]}/{dataset_name}_{datalen}_{args.method}_{sparse_budget}_{rank}_{chunk_size}_top{top_threshold}_lastq{last_q}_init{init_tokens}_recent{recent_tokens}_cv{cv_threshold}.jsonl", args.method)
        elif args.method == 'CateKV':
            if minference == True:
                evaluator.test(llm, dataset, f"archive/{model_name.split('/')[-1]}/{dataset_name}_{datalen}_{args.method}_{sparse_budget}_{rank}_{chunk_size}_top{top_threshold}_lastq{last_q}_init{init_tokens}_recent{recent_tokens}_cv{cv_threshold}_fullfraction{full_fraction}_minfernece.jsonl", args.method)
            else:
                evaluator.test(llm, dataset, f"archive/{model_name.split('/')[-1]}/{dataset_name}_{datalen}_{args.method}_{sparse_budget}_{rank}_{chunk_size}_top{top_threshold}_lastq{last_q}_init{init_tokens}_recent{recent_tokens}_cv{cv_threshold}_fullfraction{full_fraction}.jsonl", args.method)
        else:
            if minference == True:
                evaluator.test(llm, dataset, f"archive/{model_name.split('/')[-1]}/{dataset_name}_{datalen}_{args.method}_{sparse_budget}_{rank}_{chunk_size}_minference{note}.jsonl", args.method)
            else:
                evaluator.test(llm, dataset, f"archive/{model_name.split('/')[-1]}/{dataset_name}_{datalen}_{args.method}_{sparse_budget}_{rank}_{chunk_size}{note}.jsonl", args.method)
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    evaluator.summarize()