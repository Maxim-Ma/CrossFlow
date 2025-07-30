export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1 
export NCCL_SHM_DISABLE=1 
export NCCL_BLOCKING_WAIT=1
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision bf16 train_mfdit_t2i.py \
            --config=configs/smaller_mfdit_2gpu.py

# CUDA_VISIBLE_DEVICES=7 \
# accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 train_mfdit_t2i.py \
#             --config=configs/smaller_mfdit_1gpu.py