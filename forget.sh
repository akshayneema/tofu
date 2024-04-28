split=forget10
model_path='./checkpoints/ft_epoch5_lr1e-05_phi_full_wd0.01'
lr=1e-5
master_port=18765
model=phi
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_path=$model_path  lr=${lr}
# python dpo.py \
#     --model_name_or_path ./checkpoints/ft_epoch5_lr1e-05_phi_full_wd0.01 \
#       --per_device_train_batch_size 4 \
#       --gradient_accumulation_steps 2 \
#       --learning_rate 1e-5 \
#       --report_to wandb \
#       --run-id unlearn-forget10 \
#       --max_seq_length 4096 \
#       --num_train_epochs 5 \
#       --logging_strategy steps \
#       --log_steps 2 \
#       --logging_first_step \
#       --save_strategy epoch \
#       --lora_rank 64 \
#       --lora_alpha 32 \
#       --lora_dropout 0.05 \
#       --output_dir ./checkpoints/ft_epoch5_lr1e-05_phi_full_wd0.01 \
#       --data-path  ./data/dpo-forget10.json