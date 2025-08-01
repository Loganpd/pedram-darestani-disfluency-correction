# Project description
This project aims at speech disfluency correction through fine-tuning LLMs using the PEFT QLoRA method.
The models used for this project are of the Google T5 family, but of different sizes/number of parameters.

## Requirements
- torch == 2.7.0
- pandas == 2.3.0
- transformers == 4.52.4
- evaluate == 0.4.5
- sentence-transformers == 5.0.0
- matplotlib == 3.10.3
- peft == 0.16.0
- jupyter == 1.1.1
- ipykernel == 6.29.5
- datasets == 3.6.0
- absl-py == 2.3.1
- nltk == 3.9.1
- rouge-score == 0.1.2
- bitsandbytes == 0.46.1

## Installation

Clone this repository and navigate to the project directory via the GitHub CLI command shown below:

```bash
gh repo clone Loganpd/pedram-darestani-disfluency-correction
```

Then you can install the project dependencies provided in the requirements.txt file.

## Usage

- The T5-small, T5-base, and T5-large models were fine-tuned on the provided Disfl-QA datasets with the use of early stopping and best model loading.
- You can find the fine-tuned and evaluated versions of the T5 models in the fine_tuning.ipynb. 
- You can also run this notebook replicate the training and testing results.
- The training error vs step plots are also available inside model directories.


## Evaluation
The metrics used for evaluation of the fine-tuned models on speech correction are shown below.
Also,in order to evaluate the semantic similarity of the model outputs with their intended values, a cosine 
similarity is calculated between the embedded version of the two versions. Note that their embeddings were
computed by using the all-MiniLM-L6-v2 model.
- BLEU
- ROGUE
- METEOR
- Semantic (Cosine) Similarity  

The best model was chosen based on its performance on the validation data through consideration of
the ROGUE L-sum metric. The said metric was calculated on the validation set for the said three models
and is shown below.


| Models   | Rogue 1  | Rogue 2  | Rogue L  | Rogue L sum | Meteor    | Bleu      | Semantic Sim. |
|----------|----------|----------|----------|-------------|-----------|-----------|---------------|
| T5-small | 0.936700 | 0.888030 | 0.924673 | 0.919297    | 0.941396  | 0.863850  | 0.961594      |
| T5-base  | 0.953543 | 0.916162 | 0.943145 | 0.942610    | 0.956973  | 0.897622  | 0.974522      |
| T5-large | 0.947475 | 0.906822 | 0.936606 | 0.936586    | 0.955478  | 0.881500  | 0.971367      |

Based on the final metrics, the best model is selected as the T5-base model, which has 220 million parameters.
Finally, the performance of this model is evaluated on the test set and is shown below.

- rouge1:              0.943906
- Rouge2:              0.898524
- RougeL:              0.932193
- RougeLsum:           0.932230
- Meteor:              0.944948
- Bleu:                0.877530
- Semantic similarity: 0.968002


Note that due to hardware limitations, the training of the T5-large model was stopped before it is adequately 
trained. Thus, it is possible that this model would out-perform the two other models given the right number of
training epochs.

## Proprietary dataset evaluation
- In order to perform your proprietary dataset evaluation, you can use the test.ipynb file.
- Simply provide the path to your dataset JSON file and the model you would like to evaluate (one of t5-small, t5-base, or t5-large model names), and run the cells.
- You will be provided with the list of metrics evaluating the performance of the model on the given JSON file.
- You can also get the output of the model for any singular prompt, by entering your prompt and running the generate_one function.

