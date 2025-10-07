################################################################################
#
# Copyright 2048 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

import torch
import math
import gc
from torch import nn
from models.tensor_op import repeat_kv
import os
import matplotlib.pyplot as plt
import random
import string
import numpy as np
import pandas as pd
import torch.nn.functional as F
import time

class KV_Cache:
    """Full Attention"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:

        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device='cpu',
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device='cpu',
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

        # batch prefill record
        self.prefilled_batch = 0
        self.batch_size = batch_size

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):

        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]

        if bsz == self.batch_size:
            self.prefilled_batch = 0

        self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        key = self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]
        value = self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]

        if incoming > 1: # prefill
            key = key.to('cuda:0')
            value = value.to('cuda:0')

        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming
        
        return key.to('cuda:0'), value.to('cuda:0')
    
    def print_stats(self):
        print(f"KVCache | max_length {self.max_length} | dtype {self.dtype} | cached {self.kv_offset}")

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache = self.k_cache.to(self.device)
        self.v_cache = self.v_cache.to(self.device)

    def clear(self):
        self.kv_offset = 0
        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset

class ShadowKVCache:
    """ShadowKV, only for accuracy measurement and understanding, not for efficiency, please refer to ShadowKV_CPU for the efficient implementation"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
        rank=160,
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.rank = rank
        self.local_chunk = 4
        self.outlier_chunk = 48

        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        self.selected_chunk_idx = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget // self.chunk_size,
            device=self.device,
            dtype=torch.long
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )


        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def get_svd(self, new_k_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0:
            # init U, SV
            self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device=self.device, dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.rank, self.head_dim, device=self.device, dtype=self.dtype)
        
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)
        # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
        self.U[layer_idx].copy_(u[:, :, :self.rank].to(self.dtype)) # [bsz, 128k, 160]
        self.SV[layer_idx].copy_(torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(self.batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)) # [bsz, 8, 160, 128]
    
    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            query: torch.Tensor=None
            ):
        
        incoming = new_v_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        self.select_sets = self.sparse_budget // self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1
        # print(query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim).shape, self.k_landmark[layer_idx].transpose(2, 3).shape)
        # [bsz, 8, 4, q_len, 128] * [bsz, 8, 128, chunks] --> [bsz, 8, 4, q_len, chunks]
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), self.k_landmark[layer_idx].transpose(2, 3)).squeeze(2) / math.sqrt(128)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
        chunk_attn = chunk_attn.sum(dim = -2) # [bsz, 8, 4, chunks]
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(256)]

        # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]

        # this is chunk idx, which can be used to offload value cache and decide if the cache hits
        self.selected_chunk_idx[layer_idx].copy_(selected_chunks, non_blocking=True)

        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(self.batch_size, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        return position_ids
        
    def get_value_cache(self, layer_idx, position_ids):
        # gather value cache
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        # gather key cache and rope them
        u = self.U[layer_idx] # [bsz, 128k, rank]
        sv = self.SV[layer_idx] # [bsz, 8, rank, 128]

        # indexing, [bsz, 8, sparse_budget, rank]
        index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1)) # [bsz, 8, sparse_budget, rank]
        u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1) # [bsz, 8, 128k, rank]
        U_head = torch.gather(u_expand, 2, index_expanded)

        # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
        result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # rope the key cache
        result = rope_func(result, position_ids)

        # send to buffer
        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.selected_chunk_idx.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
    
    def H2D(self):
        pass

    def get_kv_len(self):
        return self.kv_offset

class QuestCache:
    """Use Max and Min landmarks for retrieval"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=16,
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size

        self.k_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark_max = []
        self.k_landmark_min = []

    def print_stats(self):
        print(f"QuestCache | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | cached {self.kv_offset}")

    def register_k_landmark(self, k_landmark_max, k_landmark_min):
        self.k_landmark_max.append(k_landmark_max.clone())
        self.k_landmark_min.append(k_landmark_min.clone())

    def prefill_kv_cache(self,
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):
        
        incoming = new_k_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()
        self.k_cache_cpu[layer_idx][:, :, :incoming] = new_k_cache.clone()

        self.chunks = incoming // self.chunk_size - 32 // self.chunk_size
        self.select_sets = self.sparse_budget // self.chunk_size
        
        self.chunk_end = self.chunks * self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        key_states_roped_ctx = new_k_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        
        k_landmark_max = key_states_roped_ctx.min(dim=-2).values
        k_landmark_min = key_states_roped_ctx.max(dim=-2).values

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(k_landmark_max, k_landmark_min)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def collect_kv(self, layer_idx, query_states):
        self.incoming_q_len = query_states.shape[-2] # 1
        min_cache = repeat_kv(self.k_landmark_min[layer_idx], self.num_key_value_groups)
        max_cache = repeat_kv(self.k_landmark_max[layer_idx], self.num_key_value_groups)
        min_value = min_cache * query_states
        max_value = max_cache * query_states

        heuristic = torch.max(min_value, max_value)
        heuristic = heuristic.sum(dim=-1)
        
        heuristic = heuristic.reshape(1, self.num_key_value_heads, self.num_key_value_groups, -1)
        heuristic = heuristic.sum(dim=-2, keepdim=True)
        
        topk_chunk = heuristic.topk(k=self.select_sets, dim=-1).indices

        position_ids = (topk_chunk.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=topk_chunk.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(1, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        key_ = self.k_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        ret_k = torch.cat([key_, self.k_cache_cpu[layer_idx][:,:,self.chunk_end:self.prefill+gen_offset]], dim = 2)
        ret_v = torch.cat([value_, self.v_cache_cpu[layer_idx][:,:,self.chunk_end:self.prefill+gen_offset]], dim = 2)

        return ret_k, ret_v
        
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.k_cache_cpu[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache_cpu[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache_cpu.zero_()
        self.v_cache_cpu.zero_()
        self.k_landmark_max = []
        self.k_landmark_min = []

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache_cpu = self.k_cache_cpu.to(self.device)
        self.v_cache_cpu = self.v_cache_cpu.to(self.device)

    def get_kv_len(self):
        return self.kv_offset


class CateKV_dynamic:
    """Use Max and Min landmarks for retrieval"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=16,
        last_q=64,
        init_tokens=64,
        recent_tokens=256,
        top_threshold=0.6,
        top_fraction=0.995,
        cv_threshold=0.7,
        model_name='llama-3'
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.last_q = last_q
        self.init_tokens = init_tokens
        self.recent_tokens = recent_tokens
        self.top_threshold = top_threshold
        self.top_fraction = top_fraction
        self.cv_threshold = cv_threshold
        self.model_name = model_name

        self.k_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark_max = []
        self.k_landmark_min = []

        self.chunk_top_num = []
        self.top_cv = []

    def print_stats(self):
        print(f"TokenCache | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | cached {self.kv_offset}")

    def register_k_landmark(self, k_landmark_max, k_landmark_min):
        self.k_landmark_max.append(k_landmark_max.clone())
        self.k_landmark_min.append(k_landmark_min.clone())

    def prefill_kv_cache(self,
            query :torch.Tensor,
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            minference_pattern
            ):
        
        incoming = new_k_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()
        self.k_cache_cpu[layer_idx][:, :, :incoming] = new_k_cache.clone()

        self.chunks = incoming // self.chunk_size - 32 // self.chunk_size
        
        self.chunk_end = self.chunks * self.chunk_size

        self.select_sets = self.sparse_budget // self.chunk_size - self.recent_tokens // self.chunk_size - self.init_tokens // self.chunk_size

        self.middle_chunks = self.chunks - self.init_tokens // self.chunk_size - self.recent_tokens //self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget - self.recent_tokens - self.init_tokens, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget - self.recent_tokens - self.init_tokens}"

        key_states_roped_ctx = new_k_cache[:,:,self.init_tokens:self.chunk_end-self.recent_tokens].view(self.batch_size, self.num_key_value_heads, self.middle_chunks, self.chunk_size, self.head_dim)

        k_landmark_max = key_states_roped_ctx.min(dim=-2).values
        k_landmark_min = key_states_roped_ctx.max(dim=-2).values
        
        q_last = query[:,:,-self.last_q:,:]
        k_trans = repeat_kv(new_k_cache[:,:,self.init_tokens:self.chunk_end-self.recent_tokens,:],self.num_key_value_groups).transpose(2,3)
        attn_last_ori = torch.matmul(q_last, k_trans) / math.sqrt(self.head_dim)
        attn_last_ori = torch.softmax(attn_last_ori,dim=-1).squeeze()
        attn_last_gqa = attn_last_ori.view(self.num_key_value_heads,self.num_key_value_groups,-1,attn_last_ori.shape[-1])
        attn_last_gqa = attn_last_gqa.mean(dim=1)
        attn_last = attn_last_gqa

        q_last_num = attn_last.shape[1]
        attn_last_flat = attn_last.view(attn_last.shape[0], -1).to(torch.float) 
        vmin = torch.quantile(attn_last_flat, 1-self.top_fraction, dim=1, keepdim=True).unsqueeze(2)
        vmax = torch.quantile(attn_last_flat, self.top_fraction, dim=1, keepdim=True).unsqueeze(2)
        vmin = vmin.repeat(1, q_last_num, 1)
        vmax = vmax.repeat(1, q_last_num, 1)

        threshold = vmin + (vmax - vmin) * self.top_threshold
        threshold = torch.clamp(threshold, min=1e-9)
        data_gt_threshold = attn_last > threshold

        chunks = data_gt_threshold.view(data_gt_threshold.size(0),data_gt_threshold.size(1),-1,self.chunk_size)
        chunk_mask = chunks.max(dim=3).values # (head_num,last_token,chunk_num)

        chunk_top_num = chunk_mask.sum(dim=1).float() # (head_num,chunk_num)

        
        # calculate cv
        self.chunk_top_num.append(chunk_top_num)
        top_mean_num = torch.mean(chunk_top_num,dim=1)
        top_std_num = torch.std(chunk_top_num,dim=1)
        top_mean_num[top_mean_num == 0] = 1e-10
        top_cv = top_std_num / top_mean_num 
        self.top_cv.append(top_cv)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(k_landmark_max, k_landmark_min)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming
            assert len(self.top_cv) == self.num_layers
            self.cv_threshold_one = torch.quantile(torch.stack(self.top_cv),self.cv_threshold)
            
            head_cls = torch.zeros(self.num_layers, self.num_key_value_heads)
            for layer_id in range(self.num_layers):
                for head_id in range(self.num_key_value_heads):
                    head_cls[layer_id][head_id] = self.top_cv[layer_id][head_id].item()
            head_cls_np = head_cls.numpy()

            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

            # save head_cls_np to csv
            save_head_dir = f'./select_headmask/headmask_new{self.model_name}/vt_fraction{self.top_fraction}_top{self.top_threshold}_lastq{self.last_q}'
            os.makedirs(save_head_dir, exist_ok=True)
            head_cls_file_path = os.path.join(save_head_dir, f"{random_string}_head_cls.csv")
            # 使用 pandas 保存为 CSV
            head_cls_df = pd.DataFrame(head_cls_np)
            head_cls_df.to_csv(head_cls_file_path, index=False, header=False)

    def collect_kv(self, layer_idx, query_states):
        self.incoming_q_len = query_states.shape[-2] 
        
        cv = self.top_cv[layer_idx]
        
        min_cache = repeat_kv(self.k_landmark_min[layer_idx], self.num_key_value_groups)
        max_cache = repeat_kv(self.k_landmark_max[layer_idx], self.num_key_value_groups)
        
        min_value = min_cache * query_states
        max_value = max_cache * query_states

        heuristic = torch.max(min_value, max_value)
        heuristic = heuristic.sum(dim=-1)

        # [1, num_key_value_heads, num_key_value_groups, -1]
        heuristic = heuristic.reshape(1, self.num_key_value_heads, self.num_key_value_groups, -1)
        heuristic = heuristic.sum(dim=-2, keepdim=True)

        topk_chunk_head = []
        head_type = torch.zeros(self.num_key_value_heads)
        for head_idx in range(self.num_key_value_heads):
            if cv[head_idx] < self.cv_threshold_one:
                topk_chunk_quest = heuristic.topk(k=self.select_sets, dim=-1).indices
                topk_chunk_head.append(topk_chunk_quest.squeeze()[head_idx,:])
            else:
                chunk_top_num = self.chunk_top_num[layer_idx][head_idx]
                topk_chunk_head.append(chunk_top_num.topk(k=self.select_sets,dim=0).indices)
                head_type[head_idx] = 1
        topk_chunk = torch.stack(topk_chunk_head).unsqueeze(0).unsqueeze(2)

        position_ids = (topk_chunk.unsqueeze(-1) * self.chunk_size + 
                        torch.arange(self.chunk_size, device=topk_chunk.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)) \
                        .view(1, self.num_key_value_heads, -1)  # [bsz, 8, select_sets * chunk_size]
        position_ids += self.init_tokens

        position_ids = position_ids.to(self.device)

        key_ = self.k_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)) # [1, 8, 1728, 128]
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len
        
        ret_k = torch.cat([self.k_cache_cpu[layer_idx][:,:,:self.init_tokens], key_, self.k_cache_cpu[layer_idx][:,:,self.chunk_end-self.recent_tokens:self.prefill+gen_offset]], dim=2)
        ret_v = torch.cat([self.v_cache_cpu[layer_idx][:,:,:self.init_tokens], value_, self.v_cache_cpu[layer_idx][:,:,self.chunk_end-self.recent_tokens:self.prefill+gen_offset]], dim=2)

        return head_type, ret_k.to('cuda:0'), ret_v.to('cuda:0'), self.k_cache_cpu[layer_idx][:,:,:self.prefill+gen_offset].to('cuda:0'),self.v_cache_cpu[layer_idx][:,:,:self.prefill+gen_offset].to('cuda:0')
        
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.k_cache_cpu[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache_cpu[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache_cpu.zero_()
        self.v_cache_cpu.zero_()
        self.k_landmark_max = []
        self.k_landmark_min = []
        self.chunk_top_num = []
        self.top_cv = []

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.cv_threshold_one = 0

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache_cpu = self.k_cache_cpu.to(self.device)
        self.v_cache_cpu = self.v_cache_cpu.to(self.device)

    def get_kv_len(self):
        return self.kv_offset



class CateKV:
    """Use Max and Min landmarks for retrieval"""
    def __init__(self, 
        config :object,
        head_classification_path: str = 'wikitext/headmask/Llama-3-8B-Instruct-Gradient-1048k/wikitext2_5000_100/percentile_csvs',
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cpu',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=16,
        last_q=64,
        init_tokens=64,
        recent_tokens=256,
        top_threshold=0.6,
        cv_threshold=0.7,
        full_fraction = 0.5,
        is_retrieval = True
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.last_q = last_q
        self.init_tokens = init_tokens
        self.recent_tokens = recent_tokens
        self.top_threshold = top_threshold
        self.cv_threshold = cv_threshold
        self.full_fraction = full_fraction
        self.is_retrieval = is_retrieval

        self.num_layers = config.num_hidden_layers

        head_cls_file_path = os.path.join(head_classification_path, f"head_type_p{int(cv_threshold*100)}.csv")
        head_cls_df = pd.read_csv(head_cls_file_path, header=None)  # Assuming CSV has no header
        self.head_cls = torch.tensor(head_cls_df.to_numpy(), dtype=torch.long, device=self.device)  # Shape: (num_layers, num_heads)

        self.full_head_indices = {}   # Dict[layer_idx] = Tensor of full head indices
        self.sparse_head_indices = {} # Dict[layer_idx] = Tensor of sparse head indices
        self.full_query_head_indices = {}   # Dict[layer_idx] = Tensor of full head indices
        self.sparse_query_head_indices = {} # Dict[layer_idx] = Tensor of sparse head indices

        self.k_cache_full = {}   
        self.v_cache_full = {}   
        self.k_cache_sparse = {} 
        self.v_cache_sparse = {}

        for layer in range(self.num_layers):
            self._initialize_layer_caches(layer)

        self.full_kv_offset = 0
        self.sparse_kv_offset = 0 
        self.prefill = 0
        self.gen_offset = 0
        self.prefilled_batch = 0

        # self.k_landmark_max = []
        # self.k_landmark_min = []

    def _initialize_layer_caches(self, layer_idx: int):
        head_types = self.head_cls[layer_idx]  # Shape: (num_heads,)
        repeated_head_type = torch.repeat_interleave(head_types, repeats=self.num_key_value_groups)
        sparse_heads = (head_types == 0).nonzero(as_tuple=True)[0]
        full_heads = (head_types == 1).nonzero(as_tuple=True)[0]
        sparse_query_heads = (repeated_head_type == 0).nonzero(as_tuple=True)[0]
        full_query_heads = (repeated_head_type == 1).nonzero(as_tuple=True)[0]

        self.full_head_indices[layer_idx] = full_heads.to('cuda:0')
        self.sparse_head_indices[layer_idx] = sparse_heads.to('cuda:0')
        self.full_query_head_indices[layer_idx] = full_query_heads.to('cuda:0')
        self.sparse_query_head_indices[layer_idx] = sparse_query_heads.to('cuda:0')

        if len(full_heads) > 0:
            self.k_cache_full[layer_idx] = torch.zeros(
                self.batch_size,
                len(full_heads),
                int((self.max_length - 1024) * self.full_fraction + (self.recent_tokens + self.init_tokens) * (1 - self.full_fraction) + 2048),
                self.head_dim,
                device=self.device,
                dtype=self.dtype
            )
            self.v_cache_full[layer_idx] = torch.zeros(
                self.batch_size,
                len(full_heads),
                int((self.max_length - 1024) * self.full_fraction + (self.recent_tokens + self.init_tokens) * (1 - self.full_fraction) + 2048),
                self.head_dim,
                device=self.device,
                dtype=self.dtype
            )
        else:
            # Initialize empty tensors for layers without full heads
            self.k_cache_full[layer_idx] = torch.empty(
                self.batch_size,
                0,  # Zero full heads
                int((self.max_length - 1024) * self.full_fraction + (self.recent_tokens + self.init_tokens) * (1 - self.full_fraction) + 2048),
                self.head_dim,
                device=self.device,
                dtype=self.dtype
            )
            self.v_cache_full[layer_idx] = torch.empty(
                self.batch_size,
                0,  # Zero full heads
                int((self.max_length - 1024) * self.full_fraction + (self.recent_tokens + self.init_tokens) * (1 - self.full_fraction) + 2048),
                self.head_dim,
                device=self.device,
                dtype=self.dtype
            )
        
        if len(sparse_heads) > 0:
            self.k_cache_sparse[layer_idx] = torch.zeros(
                self.batch_size,
                len(sparse_heads),
                self.sparse_budget + 2048,  # Adjusted budget
                self.head_dim,
                device=self.device,
                dtype=self.dtype
            )
            self.v_cache_sparse[layer_idx] = torch.zeros(
                self.batch_size,
                len(sparse_heads),
                self.sparse_budget + 2048,  # Adjusted budget
                self.head_dim,
                device=self.device,
                dtype=self.dtype
            )
        else:
            # Initialize empty tensors for layers without sparse heads
            self.k_cache_sparse[layer_idx] = torch.empty(
                self.batch_size,
                0,  # Zero sparse heads
                self.sparse_budget + 2048,
                self.head_dim,
                device=self.device,
                dtype=self.dtype
            )
            self.v_cache_sparse[layer_idx] = torch.empty(
                self.batch_size,
                0,  # Zero sparse heads
                self.sparse_budget + 2048,
                self.head_dim,
                device=self.device,
                dtype=self.dtype
            )

    def print_stats(self):
        print(f"TokenCache | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | cached {self.full_kv_offset}")

    def register_k_landmark(self, k_landmark_max, k_landmark_min):
        self.k_landmark_max.append(k_landmark_max.clone())
        self.k_landmark_min.append(k_landmark_min.clone())

    def prefill_kv_cache(self,
            query :torch.Tensor,
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            minference_pattern
            ):
        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]
        if bsz == self.batch_size:
            self.prefilled_batch = 0
        
        
        sparse_heads = self.sparse_head_indices[layer_idx]
        full_heads = self.full_head_indices[layer_idx]

        self.chunks = incoming // self.chunk_size - 32 // self.chunk_size
        
        self.chunk_end = self.chunks * self.chunk_size

        self.select_sets = self.sparse_budget // self.chunk_size - self.recent_tokens // self.chunk_size - self.init_tokens // self.chunk_size

        self.middle_chunks = self.chunks - self.init_tokens // self.chunk_size - self.recent_tokens //self.chunk_size
        assert self.select_sets * self.chunk_size == self.sparse_budget - self.recent_tokens - self.init_tokens, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget - self.recent_tokens - self.init_tokens}"
        
        q_last = query[:,:,-self.last_q:,:]
        k_trans = repeat_kv(new_k_cache[:,:,self.init_tokens:self.chunk_end-self.recent_tokens,:],self.num_key_value_groups).transpose(2,3)
        attn_last_ori = torch.matmul(q_last, k_trans) / math.sqrt(self.head_dim) # [b,32,64,n]
        attn_last_ori = torch.softmax(attn_last_ori,dim=-1)
        attn_last_gqa = attn_last_ori.view(attn_last_ori.size(0),self.num_key_value_heads,self.num_key_value_groups,-1,attn_last_ori.size(-1))
        attn_last = attn_last_gqa.mean(dim=2)

        chunks = attn_last.view(attn_last.size(0),attn_last.size(1),attn_last.size(2),-1,self.chunk_size)
        chunks = chunks.max(dim=-1).values # (bs,head_num,last_token,chunk_num)
        chunk_top_num = chunks.sum(dim=2).float() # (bs,head_num,chunk_num)
        full_cache_num = self.init_tokens + int(self.middle_chunks*self.full_fraction)*self.chunk_size + incoming - self.chunk_end +self.recent_tokens
        sparse_cache_num = self.sparse_budget + incoming - self.chunk_end
        if len(full_heads) > 0:
            full_chunk_top_num = chunk_top_num[:,full_heads,:]
            chunk_top_indices = full_chunk_top_num.topk(k=int(self.middle_chunks*self.full_fraction),dim=-1).indices # (b,h k)
            position_ids = (chunk_top_indices.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_top_num.device)) 
            position_ids = position_ids.view(position_ids.size(0), len(full_heads), -1) + self.init_tokens # (b,h,k*c)
            full_new_k_cache = new_k_cache[:,full_heads,:,:] # (b,h,n,128)
            full_new_v_cache = new_v_cache[:,full_heads,:,:]
            position_ids_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, full_new_k_cache.size(-1))  
            middle_k_cache = full_new_k_cache.gather(dim=2, index=position_ids_expanded)
            middle_v_cache = full_new_v_cache.gather(dim=2, index=position_ids_expanded)
            k = torch.cat([full_new_k_cache[:, :, :self.init_tokens, :], middle_k_cache, full_new_k_cache[:, :, self.chunk_end-self.recent_tokens:, :]],dim=2).clone()
            v = torch.cat([full_new_v_cache[:, :, :self.init_tokens, :], middle_v_cache, full_new_v_cache[:, :, self.chunk_end-self.recent_tokens:, :]],dim=2).clone()
            self.k_cache_full[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :full_cache_num, :] = k
            self.v_cache_full[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :full_cache_num, :] = v
        
        if len(sparse_heads) > 0:
            sparse_chunk_top_num = chunk_top_num[:,sparse_heads,:]
            chunk_top_indices = sparse_chunk_top_num.topk(k=self.select_sets,dim=-1).indices # (b,h,k)
            position_ids = (chunk_top_indices.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_top_num.device)) 
            position_ids = position_ids.view(position_ids.size(0), len(sparse_heads), -1) + self.init_tokens
            sparse_new_k_cache = new_k_cache[:,sparse_heads,:,:]
            sparse_new_v_cache = new_v_cache[:,sparse_heads,:,:]
            position_ids_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, sparse_new_k_cache.size(-1))  
            middle_k_cache = sparse_new_k_cache.gather(dim=2, index=position_ids_expanded)
            middle_v_cache = sparse_new_v_cache.gather(dim=2, index=position_ids_expanded)
            self.k_cache_sparse[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :sparse_cache_num, :] = torch.cat([sparse_new_k_cache[:, :, :self.init_tokens, :], middle_k_cache, sparse_new_k_cache[:, :, self.chunk_end-self.recent_tokens:, :]],dim=2).clone()
            self.v_cache_sparse[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :sparse_cache_num, :] = torch.cat([sparse_new_v_cache[:, :, :self.init_tokens, :], middle_v_cache, sparse_new_v_cache[:, :, self.chunk_end-self.recent_tokens:, :]],dim=2).clone()

        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            assert self.sparse_budget < incoming
            if self.prefilled_batch == self.batch_size:
                self.full_kv_offset += full_cache_num
                self.sparse_kv_offset += sparse_cache_num
                self.prefill = incoming

    def collect_kv(self, layer_idx, query_states):

        self.incoming_q_len = query_states.shape[-2]  

        full_head_query_index = self.full_query_head_indices[layer_idx]
        sparse_head_query_index = self.sparse_query_head_indices[layer_idx]

        full_kv_offset = self.full_kv_offset if layer_idx == self.num_layers - 1 else self.full_kv_offset + self.incoming_q_len
        full_k = self.k_cache_full[layer_idx][:,:,:full_kv_offset]
        full_v = self.v_cache_full[layer_idx][:,:,:full_kv_offset]
        sparse_kv_offset = self.sparse_kv_offset if layer_idx == self.num_layers - 1 else self.sparse_kv_offset + self.incoming_q_len
        sparse_k = self.k_cache_sparse[layer_idx][:,:,:sparse_kv_offset]
        sparse_v = self.v_cache_sparse[layer_idx][:,:,:sparse_kv_offset]

        return full_head_query_index, sparse_head_query_index, full_k.to('cuda:0'), full_v.to('cuda:0'), sparse_k.to('cuda:0'), sparse_v.to('cuda:0')

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]

        sparse_heads = self.sparse_head_indices[layer_idx]
        full_heads = self.full_head_indices[layer_idx]

        full_new_k_cache = new_k_cache[:,full_heads,:,:]
        full_new_v_cache = new_v_cache[:,full_heads,:,:]
        sparse_new_k_cache = new_k_cache[:,sparse_heads,:,:]
        sparse_new_v_cache = new_v_cache[:,sparse_heads,:,:]
        
        self.k_cache_full[layer_idx][:, :, self.full_kv_offset:self.full_kv_offset + incoming].copy_(full_new_k_cache)
        self.v_cache_full[layer_idx][:, :, self.full_kv_offset:self.full_kv_offset + incoming].copy_(full_new_v_cache)
        self.k_cache_sparse[layer_idx][:, :, self.sparse_kv_offset:self.sparse_kv_offset + incoming].copy_(sparse_new_k_cache)
        self.v_cache_sparse[layer_idx][:, :, self.sparse_kv_offset:self.sparse_kv_offset + incoming].copy_(sparse_new_v_cache)

        if layer_idx == self.num_layers - 1:
            self.full_kv_offset += incoming
            self.sparse_kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache_full = {}   
        self.v_cache_full = {}   
        self.k_cache_sparse = {} 
        self.v_cache_sparse = {}
        self.k_landmark_max = []
        self.k_landmark_min = []

        self.full_kv_offset = 0
        self.sparse_kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefilled_batch = 0

        for layer in range(self.num_layers):
            self._initialize_layer_caches(layer)

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # self.k_cache_full = {k: v.to('cuda:0') for k, v in self.k_cache_full.items()}
        # self.v_cache_full = {k: v.to('cuda:0') for k, v in self.v_cache_full.items()}
        # self.k_cache_sparse = {k: v.to('cuda:0') for k, v in self.k_cache_sparse.items()}
        # self.v_cache_sparse = {k: v.to('cuda:0') for k, v in self.v_cache_sparse.items()}

    def get_kv_len(self):
        return self.prefill + self.gen_offset