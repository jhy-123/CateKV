export CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 1 test/eval_acc.py \
    --model_name ckpt/Llama-3-8B-Instruct-Gradient-1048k \
    --datalen 131072 \
    --method full \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2,ruler/niah_multikey_3" \
    --sparse_budget 2048 \
    --rank 160 \
    --chunk_size 8 \
    # --minference

# OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 1 test/eval_acc.py \
#     --model_name ckpt/Llama-3-8B-Instruct-Gradient-1048k \
#     --datalen 131072 \
#     --method snapkv \
#     --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2,ruler/niah_multikey_3" \
#     --sparse_budget 52428 \
#     --recent_token 32 \