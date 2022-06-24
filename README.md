# Scientific Comparative Argument Generation

## Data
We provide data in JSONLine and TSV format. Please access the data from this [Google Drive link](https://drive.google.com/drive/folders/10YrvtT6k5F7wlrQI8g0Jv-8yC4vUB9vB?usp=sharing).
-  `data/jsonl_files/` The detailed data information of the full dataset, containing SCAG and SCAGml. Please find an example in the Appendix.
-  `data/SCAG/` The SCAG data in TSV format. Each file has the following columns:
   -   "Y": the $CmpArg$ sentences.
   -   "A": the abstract of citing paper $A$
   -   "B": the abstract and introduction of cited paper $B$
   -   "idA": the id (unique identifier in S2ORC) of $A$
   -   "idB": the id of $B$
-   `data/SCAGml/` The SCAGml data in TSV format. Besides the columns that are the same as SCAG, the SCAG files have one more column:
    -   "T": the topic id of the $CmpArg$ sentence.



## Code
We provide codes, dummy datasets, and scripts for $CmpArg$ generation and output topic classification. We also provide the evaluation script for generation.
- `code/generation/BART` Generation with BART, used by our M1, M2, and M5.
- `code/generation/BART_topic_guided` Generation with BART and topic guidance, used by our M3, M4, M6, M7.
- `code/classification/` Topic classification with BART and BERT.
- `code/evaluate.py` Evluation script for generation. We calculate ROUGE scores with the [rouge](https://github.com/pltrdy/rouge) library, and calculate BLEU and METEOR with the [nlgeval](https://github.com/Maluuba/nlg-eval/tree/master/nlgeval).

## Requirements
- Python >= 3.6
- Huggingface == 3.3.0
- Pytorch == 1.7



## Appendix: An example of the jsonl data
In the jsonl files, each line is a JSON. The main keys are:

- "doc_A": The citing paper $A$. We collect the abstract.
- "doc_B": The cited paper(s) $B$. We collect each paper's abstract and introduction (first section of the main body).
- "summ_texts": The $Smry$.
- "comp_texts:": The $CmpArg$.
- "annotation_rule": The annotation rule. Only used for data collection.


Here is a data example pretty printed for better illustration.
```json
{
    "doc_A": {
        "id": "54555710",
        "text": {
            "abstract": [
                "We explore the performance of latent variable models for conditional text generation in the context of neural machine translation . Similar to #CITE# , we augment the encoder-decoder neural machine translation paradigm by introducing a continuous latent variable to model features of the translation process. We extend this model with a co-attention mechanism motivated by #CITE# in the inference network. Compared to the vision domain, latent variable models for text face additional challenges due to the discrete nature of language, namely posterior collapse #CITE# . We experiment with different approaches to mitigate this issue. We show that our conditional variational model improves upon both discriminative attention-based translation and the variational baseline presented in #CITE# . Finally, we present some exploration of the learned latent space to illustrate what the latent variable is capable of capturing. This is the first reported conditional variational model for text that meaningfully utilizes the latent variable without weakening the translation model."
            ]
        }
    },
    "doc_B": [
        {
            "id": "11212020",
            "text": {
                "abstract": [
                    "Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder-decoders and encode a source sentence into a fixed-length vector from which a decoder generates a translation. In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and propose to extend this by allowing a model to automatically search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly. With this new approach, we achieve a translation performance comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. Furthermore, qualitative analysis reveals that the alignments found by the model agree well with our intuition."
                ],
                "introduction": [
                    "Neural machine translation is a newly emerging approach to machine translation, recently proposed by Kalchbrenner and Blunsom #CITE# , Sutskever et al. #CITE# and Cho et al. . Unlike the traditional phrase-based translation system #CITE# which consists of many small sub-components that are tuned separately, neural machine translation attempts to build and train a single, large neural network that reads a sentence and outputs a correct translation.",
                    "Most of the proposed neural machine translation models belong to a family of encoderdecoders #CITE# , with an encoder and a decoder for each language, or involve a language-specific encoder applied to each sentence whose outputs are then compared #CITE# . An encoder neural network reads and encodes a source sentence into a fixed-length vector. A decoder then outputs a translation from the encoded vector. The whole encoder-decoder system, which consists of the encoder and the decoder for a language pair, is jointly trained to maximize the probability of a correct translation given a source sentence.",
                    "A potential issue with this encoder-decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. This may make it difficult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus. Cho et al. showed that indeed the performance of a basic encoder-decoder deteriorates rapidly as the length of an input sentence increases.",
                    "In order to address this issue, we introduce an extension to the encoder-decoder model which learns to align and translate jointly. Each time the proposed model generates a word in a translation, it searches for a set of positions in a source sentence where the most relevant information is concentrated. The model then predicts a target word based on the context vectors associated with these source positions and all the previous generated target words.",
                    "The most important distinguishing feature of this approach from the basic encoder-decoder is that it does not attempt to encode a whole input sentence into a single fixed-length vector. Instead, it encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation. This frees a neural translation model from having to squash all the information of a source sentence, regardless of its length, into a fixed-length vector. We show this allows a model to cope better with long sentences.",
                    "In this paper, we show that the proposed approach of jointly learning to align and translate achieves significantly improved translation performance over the basic encoder-decoder approach. The improvement is more apparent with longer sentences, but can be observed with sentences of any length. On the task of English-to-French translation, the proposed approach achieves, with a single model, a translation performance comparable, or close, to the conventional phrase-based system. Furthermore, qualitative analysis reveals that the proposed model finds a linguistically plausible alignment between a source sentence and the corresponding target sentence."
                ]
            }
        }
    ],
    "summ_texts": [
        [
            [
                "11212020"
            ],
            [
                "The attention mechanism introduced in #CITE# enhances this model by aligning source and target words using the encoder RNN hidden states."
            ]
        ]
    ],
    "comp_texts": [
        "However, it has been shown that this type of models struggles to learn smooth, interpretable global semantic features #CITE# ."
    ],
    "annotation_rule": "this"


```


