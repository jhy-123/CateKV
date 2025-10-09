export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=48

percentile_threshold=(0.998 0.996 0.994 0.992 0.990 0.988 0.986 0.984)
scaling_factor=(1.0 0.8)

for percentile in "${percentile_threshold[@]}"; do
    for factor in "${scaling_factor[@]}"; do
        echo "with top_fraction = $top_frac, top_threshold = $threshold"

        torchrun --standalone --nnodes=1 --nproc_per_node=1 test/eval_acc.py \
            --model_name ckpt/Llama-3-8B-Instruct-Gradient-1048k \
            --datalen 131072 \
            --method CateKV_dynamic \
            --dataset_name "ruler/vt" \
            --num_samples 50 \
            --sparse_budget 2048 \
            --rank 160 \
            --chunk_size 8 \
            --last_q 64 \
            --init_tokens 64 \
            --recent_tokens 256 \
            --top_threshold "$percentile" \
            --top_fraction "$factor" \
            --cv_threshold 0.3 
    done
done