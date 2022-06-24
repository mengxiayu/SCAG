python bart_text_classification_pytorch.py \
--data_dir dummy_data \
--output_dir output \
--model_name_or_path facebook/bart-base \
--num_train_epochs 20 \
--max_source_length 256 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--learning_rate 1e-5 \
--overwrite_output_dir \
