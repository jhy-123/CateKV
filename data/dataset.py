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

from datasets import load_dataset
from termcolor import colored
import random
import numpy as np

# RULER
from .metrics import needle_score, string_match_part, multi_number, multi_words

# NIAH
from data.utils import generate_random_number, read_context_files, create_contexts, NIAH_TEMPLATE, RANDOM_NEEDLE_CITIES

METRICS_FN = {
    'niah': needle_score,
    'multi': multi_number,
    'vt': multi_words,
    'cwe': multi_words,
    'fwe': multi_words,
    'qa': string_match_part,
}

GEN_LEN = {
    'niah': 64,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32,
}

DATADIR = {
    'ruler': 'data/ruler/data',
    'niah': 'data/niah/data',
}

class Dataset:
    def __init__(self, dataset_name, tokenizer, datalen, num_samples, rank=0, world_size=1):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.datalen = datalen
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.is_sharded = False

        if dataset_name == 'niah':
            self.tokenized_prompts, self.gt, self.ctx_len, self.depth_pct = self.get_dataset()
        else:
            self.tokenized_prompts, self.gt = self.get_dataset()
        
        self.num_samples = len(self.tokenized_prompts)
        self.gen_len = self.get_gen_len()
        self.metric = self.get_metric()

    def __str__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __len__(self) -> int:
        return self.num_samples

    def shard(self, rank, world_size):
        if world_size > 1:
            shard_size = self.num_samples // world_size
            start = rank * shard_size
            end = start + shard_size if rank != world_size - 1 else self.num_samples
            shard_tokenized_prompts, shard_gt = self.tokenized_prompts[start:end], self.gt[start:end]
            self.tokenized_prompts = shard_tokenized_prompts
            self.gt = shard_gt
            self.num_samples = len(shard_tokenized_prompts)

        self.is_sharded = True

    def get_gen_len(self):
        if 'niah' == self.dataset_name:
            return 10
        elif 'niah' in self.dataset_name:
            return 128
        elif 'vt' in self.dataset_name:
            return 30
        elif 'cwe' in self.dataset_name:
            return 120
        elif 'fwe' in self.dataset_name:
            return 50
        elif 'qa' in self.dataset_name:
            return 32
        else:
            raise Exception("Gen len not found")

    def __getitem__(self, idx):
        if 'persona' in self.dataset_name:
            return self.tokenized_prompts[idx], self.queries[idx], self.gt[idx]
        return self.tokenized_prompts[idx], self.gt[idx]

    def get_metric(self):
        if 'multiquery' in self.dataset_name or 'multivalue' in self.dataset_name:
            return METRICS_FN['multi']
        elif 'niah' in self.dataset_name:
            return METRICS_FN['niah']
        elif 'vt' in self.dataset_name:
            return METRICS_FN['vt']
        elif 'cwe' in self.dataset_name:
            return METRICS_FN['cwe']
        elif 'fwe' in self.dataset_name:
            return METRICS_FN['fwe']
        elif 'qa' in self.dataset_name:
            return METRICS_FN['qa']
        else:
            raise Exception("Metric not found")

    def get_dataset(self):
        if 'ruler' in self.dataset_name: # ruler/xxx
            task = self.dataset_name.split('/')[-1]
            assert self.datalen in [8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024], "Only support datalen of 16k, 32k, 64k, 128k"

            if 'llama-3' in self.tokenizer.name_or_path.lower():
                model_dir = 'llama-3'
            elif 'yi' in self.tokenizer.name_or_path.lower():
                model_dir = 'yi'
            elif 'lwm' in self.tokenizer.name_or_path.lower():
                model_dir = 'lwm'
            elif 'glm' in self.tokenizer.name_or_path.lower():
                model_dir = 'glm'
            elif 'qwen' in self.tokenizer.name_or_path.lower():
                model_dir = 'qwen'
            elif 'phi' in self.tokenizer.name_or_path.lower():
                model_dir = 'phi'
            else:
                raise Exception("Model not found", self.tokenizer.name_or_path)

            dataset = load_dataset("json", data_files=f'{DATADIR["ruler"]}/{model_dir}/{self.datalen}/{task}/validation.jsonl', split='train')
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            gt = []

            for i in range(self.num_samples):
                input_text = dataset[i]['input']
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
                tokenized_prompts.append(input_ids)
                gt.append(dataset[i]['outputs'])

            return tokenized_prompts, gt

        elif self.dataset_name == 'niah':
            print(colored(f"[Warning] NIAH dataset cannot set # samples, it is up to world_size, which is set to {self.world_size}", 'red'))
            
            haystack_file = f'{DATADIR["niah"]}/pg19_mini.jsonl'
            context_lengths_min = 16*1024
            context_lengths_max = self.datalen
            n_context_length_intervals = 15
            n_document_depth_intervals = 10  # position of the needle in the haystack
            n_rounds = 1 # max(1, 4 // self.world_size) # 8 rounds in total assume we have 8xGPUs
            needle = "\nThe special magic {city} number is: {rnd_number}\n"
            retrieval_question="What is the special magic {} number?"
            rnd_number_digits = 7

            context_lengths = np.round(
                np.linspace(
                    context_lengths_min,
                    context_lengths_max,
                    num=n_context_length_intervals,
                    endpoint=True,
                )
            ).astype(int)

            document_depth_percents = np.round( # we use linear scale here
                np.linspace(
                    0,
                    100,
                    num=n_document_depth_intervals,
                    endpoint=True,
                )
            ).astype(int)

            self.is_sharded = True # we shard the data during init dataset
            
            full_contexts = read_context_files(n=n_rounds, context_lengths=context_lengths, haystack_file=haystack_file, tokenizer=self.tokenizer)
            full_tokens = [
                self.tokenizer.encode(full_context, add_special_tokens=False) for full_context in full_contexts
            ]

            tokenized_prompts = []
            gt = []
            ctx_len = []
            depth_pct = []

            for context_length in context_lengths:
                trim_contexts = [
                    self.tokenizer.decode(full_token[:context_length], skip_special_tokens=True)
                    for full_token in full_tokens
                ]
                contexts = []
                for depth_percent in document_depth_percents:
                    for i in range(n_rounds):
                        random_city = random.choice(RANDOM_NEEDLE_CITIES)
                        insert_needle = True
                        needle_rnd_number = str(generate_random_number(rnd_number_digits))
                        context = create_contexts(
                            needle_rnd_number=needle_rnd_number,
                            insert_needle=insert_needle,
                            random_city=random_city,
                            trim_context=trim_contexts[i],
                            context_length=context_length,
                            depth_percent=depth_percent,
                            needle=needle,
                            retrieval_question=retrieval_question,
                            tokenizer=self.tokenizer,
                            final_context_length_buffer=32,
                        )
                        contexts.append(context)

                for context in contexts:
                    prompt = NIAH_TEMPLATE.format(
                        context=context["context"], question=context["question"]
                    )
                    input_tensor = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
                    tokenized_prompts.append(input_tensor.input_ids)
                    gt.append(context["needle_rnd_number"])
                    ctx_len.append(context["context_length"])
                    depth_pct.append(context["depth_percent"])
            
            return tokenized_prompts, gt, ctx_len, depth_pct

        else:
            raise ValueError(f"Dataset {self.dataset_name} not found, please choose in ruler, persona, infini_bench, needle, niah, long_bench")