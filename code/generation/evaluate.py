import json
from rouge import Rouge
from nlgeval import NLGEval

def load_lines(fn):
    with open(fn) as f:
        return [line.strip() for line in f]

def load_TY_lines(fn):
    with open(fn) as f:
        return [line.strip().split('\t')[1] for line in f]

rouge = Rouge()
nlgeval = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True)  # loads the models

def get_scores(hyps, refs):
    assert len(hyps) == len(refs)
    rouge_scores = rouge.get_scores(hyps, refs, avg=True)
    metrics_dict = nlgeval.compute_metrics([refs], hyps)
    for k,v in rouge_scores.items():
        metrics_dict[k] = v

    return metrics_dict

model_dir = "<your_model_dir>"
ref_dir = "<your_ref_dir>"

val_refs = load_lines(ref_dir + "val.target")
test_refs = load_lines(ref_dir + "test.target")


epoch_range = range(1)
val_epoch_scores = []
test_epoch_scores = []

for e in epoch_range:
    val_hyps = load_lines(f"{model_dir}eval_generations.txt")
    test_hyps = load_lines(f"{model_dir}test_generations.txt")
    # val_hyps = load_lines(f"{model_dir}eval_output_e_{e}.txt")
    # test_hyps = load_lines(f"{model_dir}test_output_epoch_{e}.txt")
    val_epoch_scores.append(get_scores(val_hyps, val_refs))
    test_epoch_scores.append(get_scores(test_hyps, test_refs))
    
with open (model_dir + "val_metrics_overlap.txt", 'w') as f:
    f.write("Epoch\tBleu_1\tBleu_2\tBleu_3\tBleu_4\tROUGE_L\tROUGE_1-R\tROUGE_1-F\tROUGE_2-R\tROUGE_2-F\tROUGE_L-R\tROUGE_L-F\tCIDEr\tMETEOR\n")
    for i in range(len(val_epoch_scores)):
        e = list(epoch_range)[i]
        x = val_epoch_scores[i]
        # print(f"epoch {i}, {x}")
        f.write(f"{e}\t{x['Bleu_1']:.6f}\t{x['Bleu_2']:.6f}\t{x['Bleu_3']:.6f}\t{x['Bleu_4']:.6f}\t{x['ROUGE_L']:.6f}\t{x['rouge-1']['r']:.6f}\t{x['rouge-1']['f']:.6f}\t{x['rouge-2']['r']:.6f}\t{x['rouge-2']['f']:.6f}\t{x['rouge-l']['r']:.6f}\t{x['rouge-l']['f']:.6f}\t{x['CIDEr']:.6f}\t{x['METEOR']:.6f}\n")

with open (model_dir + "test_metrics_overlap.txt", 'w') as f:
    f.write("Epoch\tBleu_1\tBleu_2\tBleu_3\tBleu_4\tROUGE_L\tROUGE1-R\tROUGE1-F\tROUGE2-R\tROUGE2-F\tROUGEL-R\tROUGEL-F\tCIDEr\tMETEOR\n")
    for i in range(len(test_epoch_scores)):
        e = list(epoch_range)[i]
        x = test_epoch_scores[i]
        # print(f"epoch {i}, {x}")
        f.write(f"{e}\t{x['Bleu_1']:.6f}\t{x['Bleu_2']:.6f}\t{x['Bleu_3']:.6f}\t{x['Bleu_4']:.6f}\t{x['ROUGE_L']:.6f}\t{x['rouge-1']['r']:.6f}\t{x['rouge-1']['f']:.6f}\t{x['rouge-2']['r']:.6f}\t{x['rouge-2']['f']:.6f}\t{x['rouge-l']['r']:.6f}\t{x['rouge-l']['f']:.6f}\t{x['CIDEr']:.6f}\t{x['METEOR']:.6f}\n")
        
        