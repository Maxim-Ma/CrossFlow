CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 demo_t2i.py \
          --config=configs/distill_student.py \
          --nnet_path=checkpoints/3/best.ckpt/nnet_ema.pth \
          --img_save_path=student_demo3 \
          --prompt='A beautiful landscape with mountains and a river.' \

# accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 demo_t2i.py \
#           --config=configs/distill_teacher.py \
#           --nnet_path=checkpoints/28000.ckpt/nnet_ema.pth \
#           --img_save_path=teacher_demo \
#           --prompt='A beautiful landscape with mountains and a river.' \