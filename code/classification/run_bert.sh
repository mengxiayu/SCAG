python bert_text_classification_pytorch.py \
--data_dir dummy_data \
--output_dir output \
--model_name_or_path bert-base-uncased \
--num_train_epochs 15 \
--max_source_length 256 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 32 \
--learning_rate 5e-5 \
--overwrite_output_dir \

