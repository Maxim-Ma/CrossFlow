CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 train_t2i.py \
            --config=configs/small_one_dimr.py