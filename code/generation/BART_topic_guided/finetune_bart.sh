python finetune.py \
    --data_dir dummy_data \
    --model_name_or_path facebook/bart-base \
    --warmup_model_path <best_checkpoint_of_SCAG_training> \
    --output_dir output \
    --num_train_epochs 25 \
    --max_source_length 512 \
    --max_target_length 32 \
    --val_max_target_length 32 \
    --test_max_target_length 32 \
    --num_labels 3 \
    --learning_rate 5e-6 \
    --fp16 \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --dataloader_num_workers 6 \
