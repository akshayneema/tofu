
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}

python dpo.py \
    --model_name_or_path ./checkpoints/ft_epoch5_lr1e-05_phi_full_wd0.01 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 2 \
      --learning_rate 1e-5 \
      --report_to wandb \
      --run-id unlearn \
      --max_seq_length 4096 \
      --num_train_epochs 5 \
      --logging_strategy steps \
      --log_steps 100 \
      --logging_first_step \
      --save_strategy epoch \
      --lora_rank 64 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --output_dir ./checkpoints/ft_epoch5_lr1e-05_phi_full_wd0.01 \
      --dataset_dir  ./data/dpo-forget01.json