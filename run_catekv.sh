export CUDA_VISIBLE_DEVICES=5
OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 1 test/eval_acc.py \
    --datalen 131072 \
    --model_name ./ckpt/Llama-3-8B-Instruct-Gradient-1048k \
    --head_classification_path ./select_headmask/llama3 \
    --method CateKV \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2,ruler/niah_multikey_3" \
    --sparse_budget 2048 \
    --rank 160 \
    --chunk_size 8 \
    --last_q 64 \
    --init_tokens 64 \
    --recent_tokens 256 \
    --cv_threshold 0.4 \
    --full_fraction 1.0 \
