

from transformers import BartTokenizer, BartForSequenceClassification
from transformers import TrainingArguments
from utils import SeqClassificationDatset
from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments
)

from torch.optim import AdamW

from typing import Callable, Dict, Iterable, List, Tuple, Union, Optional


from torch.utils.data import DataLoader
import numpy as np
from datasets import load_metric
import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
from dataclasses import dataclass, field
import json
import os
import shutil

# from datasets import load_dataset
# dataset = load_dataset("codyburker/yelp_review_sampled")
# print(dataset['train'][0])

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Ensure output dir is not existed
if (
    os.path.exists(training_args.output_dir)
    and os.listdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

def create_output_dir(dir):
    # if os.path.exists(dir):
    #     shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
create_output_dir(training_args.output_dir)

tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path)
special_tokens = ["$SEP$", "#CITE#"] # special tokens for CiteExplainer 
tokenizer.add_tokens(special_tokens)


train_dataset = SeqClassificationDatset(
    tokenizer=tokenizer,
    type_path="train",
    data_dir=data_args.data_dir,
    n_obs=data_args.n_train if data_args.n_train != -1 else None,
    max_source_length=data_args.max_source_length
    )
val_dataset = SeqClassificationDatset(
    tokenizer=tokenizer,
    type_path="val",
    data_dir=data_args.data_dir,
    n_obs=data_args.n_val if data_args.n_val != -1 else None,
    max_source_length=data_args.max_source_length
    )
test_dataset = SeqClassificationDatset(
    tokenizer=tokenizer,
    type_path="test",
    data_dir=data_args.data_dir,
    n_obs=data_args.n_test if data_args.n_test != -1 else None,
    max_source_length=data_args.max_source_length
    )
print(len(train_dataset), len(val_dataset), len(test_dataset))

model = BartForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=3)
model.resize_token_embeddings(len(tokenizer))

from sklearn.metrics import f1_score
def compute_metrics(predictions, labels):
    micro_f1 = f1_score(predictions, labels, average='micro')
    macro_f1 = f1_score(predictions, labels, average='macro')
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

def evaluate(training_args, model, val_dataloader):
    eval_loss = 0.0
    epoch_logits = []
    epoch_labels = []
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        eval_loss += outputs.loss
        epoch_logits.append(outputs.logits)
        epoch_labels.append(batch["labels"])
    epoch_logits = torch.cat(epoch_logits).cpu()
    epoch_labels = torch.cat(epoch_labels).cpu()
    predictions = np.argmax(epoch_logits, axis=-1)
    metrics = compute_metrics(predictions, epoch_labels)
    metrics["eval_loss"] = eval_loss.cpu().item()
    return predictions, metrics

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=training_args.per_device_train_batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size)
test_dataloader = DataLoader(test_dataset,  batch_size=training_args.per_device_eval_batch_size)


optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

num_epochs = int(training_args.num_train_epochs)
num_training_steps = num_epochs * len(train_dataset)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
model.to(device)

# progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for batch in train_dataloader:
        
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # progress_bar.update(1)
    with torch.no_grad():
        with open(training_args.output_dir + "/training_loss.txt", 'a') as loss_out:
            loss_out.write(f"epoch {epoch}, loss: {loss.cpu().numpy()}\n")
        # val
        predictions, eval_metrics = evaluate(training_args, model, val_dataloader)
        
        with open(training_args.output_dir + '/epoch_val.json', 'a') as epoch_eval:
            json.dump(eval_metrics, epoch_eval, indent=1)
        with open(training_args.output_dir + '/eval_output_epoch_{}.txt'.format(epoch+1), 'w') as epoch_out:
            for pred in predictions:
                epoch_out.write(f"{pred}\n")
        # test (train)
        predictions, test_metrics = evaluate(training_args, model, test_dataloader)
        print(len(predictions))
        with open(training_args.output_dir + '/epoch_test.json', 'a') as epoch_eval:
            json.dump(test_metrics, epoch_eval, indent=1)
        with open(training_args.output_dir + '/test_output_epoch_{}.txt'.format(epoch+1), 'w') as epoch_out:
            for pred in predictions:
                epoch_out.write(f"{pred}\n")
        print(f"epoch {epoch}, training loss: {loss.item():.3f}, eval metrics: {eval_metrics}, test metrics: {test_metrics}")
        # save_model_dir = training_args.output_dir + f'/checkpoint_epoch_{epoch}'
        # if os.path.exists(save_model_dir):
        #     shutil.rmtree(save_model_dir)
        # os.makedirs(save_model_dir)
        # model.save_pretrained(save_model_dir)
        # tokenizer.save_pretrained(save_model_dir)
        # print(f"epoch {epoch} :model saved in {save_model_dir} ")
    

