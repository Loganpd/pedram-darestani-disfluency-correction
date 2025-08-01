import json
import torch
import evaluate
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _load_file(filename):
    """
    Loads the provided Disfl-QA dataset JSON file.
    :param filename: the name of the JSON file in the data/ directory
    :return: a Transformers Dataset object containing the input data.
    """
    with open(f"../data/{filename}", 'r') as file:
        data =  json.load(file)
    inputs = [data[key]['disfluent'] for key in data.keys()]
    targets = [data[key]['original'] for key in data.keys()]
    return Dataset.from_dict({'inputs': inputs, 'targets': targets})


def load_data():
    """
    loads the training, validation and test datasets into a Transformers Dataset object.
    :return: train, validation, and test datasets.
    """
    train_data = _load_file('train.json')
    validation_data = _load_file('dev.json')
    test_data = _load_file('test.json')
    return train_data, validation_data, test_data


def prepare_data(data, tokenizer, max_input_length=128, max_output_length=128):
    """
    Carries out preprocessing steps required to prepare the data for fine-tuning the T5 models.
    :param data: the input data.
    :param tokenizer: the tokenizer that will be used to tokenize the data.
    :param max_input_length: maximum input length to be considered.
    :param max_output_length: maximum output length to be considered.
    :return: The preprocessed data.
    """
    def _prepare_data(record):
        inputs = record["inputs"]
        targets = record["targets"]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=max_output_length, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        return model_inputs
    return data.map(_prepare_data, batched=True)


def plot_error(log_history, model_name):
    """
    Plots the model loss vs training steps for both training and validation datasets and saves it.
    :param log_history: the history that contains the model losses and their respective steps.
    :param model_name: name of the model for saving the plots with proper naming.
    :return: None. Only saves a plot to disk.
    """
    df = pd.DataFrame(log_history)

    train_loss = df[df['loss'].notna()][['step', 'loss']]
    eval_loss = df[df['eval_loss'].notna()][['step', 'eval_loss']]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss["step"], train_loss["loss"], label="Training Loss", marker='o')
    plt.plot(eval_loss["step"], eval_loss["eval_loss"], label="Eval Loss", marker='x')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.ylim(0, )
    plt.xlim(0, )
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./tuned_models/{model_name}/loss-vs-steps_{model_name}.png")
    plt.show()


def generate_one(prompt, model, tokenizer, add_instructions=False):
    """
    Generates the output of the model for a single prompt.
    :param prompt: The prompt to be fed into the model for output generation.
    :param model: The model that generates the outputs.
    :param tokenizer: The proper tokenizer for the model.
    :param add_instructions: To be used for prompt engineering purposes with instructional LLMs.
    :return: The output of the model.
    """
    if add_instructions:
        instructions = ("Below you will receive a question that contains speech correction. "
                        "Please extract and return the fluent version of the question. "
                        "Do not provide any extra information other than the fluent version.\n")
        prompt = instructions + prompt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def evaluate_outputs(outputs, targets):
    """
    Function for evaluating the outputs of a model against their target values.
    :param outputs: The LLM outputs to be checked for their quality.
    :param targets: The expected values of the LLM outputs.
    :return: A dictionary containing metrics about the quality of the outputs.
    """
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    bleu_score = bleu_metric.compute(predictions=outputs, references=[[l] for l in targets])
    rouge_result = rouge_metric.compute(predictions=outputs, references=targets)
    meteor_result = meteor_metric.compute(predictions=outputs, references=targets)
    semantic_similarity_score = semantic_similarity(outputs, targets)

    return {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "rougeLsum": rouge_result["rougeLsum"],
        "meteor": meteor_result["meteor"],
        "bleu": bleu_score["bleu"],
        "semantic similarity": semantic_similarity_score
    }


def semantic_similarity(outputs, targets):
    """
    Calculated the pair-wise cosine similarity between outputs and targets.
    :param outputs: The LLM outputs to be checked for their quality.
    :param targets: The expected values of the LLM outputs.
    :return: the pair-wise cosine similarity between outputs and targets.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    output_embeddings = model.encode(outputs)
    target_embeddings = model.encode(targets)
    similarities = model.similarity_pairwise(output_embeddings, target_embeddings)
    return similarities.mean().item()


def load_model(model_name, device):
    """
    Loads the saved fine-tuned model from disk.
    :param model_name: model's name to be loaded.
    :param device: GPU or CPU device to be used.
    :return: the tokenizer and the model for the LLM.
    """
    tokenizer = AutoTokenizer.from_pretrained(f"./tuned_models/{model_name}/{model_name}-qlora-final")
    model = AutoModelForSeq2SeqLM.from_pretrained(f"./tuned_models/{model_name}/{model_name}-qlora-final").to(device)
    return tokenizer, model

def mini_batch_inference(input_texts, model, tokenizer, batch_size, device):
    """
    Carries out the inference in batch mode in order to deal with memory limitations of GPU.
    :param input_texts: The input texts that are to be used for inference and generate outputs.
    :param batch_size: The batch size for inference.
    :param model: The model that generates the outputs.
    :param tokenizer: The proper tokenizer for the model.
    :return: A list containing all the decoded outputs of the transformer model.
    """
    input_texts = tokenizer(input_texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    model.eval()
    outputs = []
    for i in range(0, len(input_texts['input_ids']), batch_size):
        batch = input_texts['input_ids'][i:i + batch_size].to(device)
        with torch.no_grad():
            outputs += tokenizer.batch_decode(model.generate(input_ids=batch,
                                                            max_length=128,
                                                            ),
                                              skip_special_tokens=True)
        torch.cuda.empty_cache()
    return outputs

def load_test_data(file_path):
    """
    Loads the proprietary test data that you would like to provide
    :param file_path: The path to the proprietary test data.
    """
    with open(file_path, 'r') as file:
        data =  json.load(file)
    inputs = [data[key]['disfluent'] for key in data.keys()]
    targets = [data[key]['original'] for key in data.keys()]
    return Dataset.from_dict({'inputs': inputs, 'targets': targets})

def proprietary_evaluation(file_path, model_name, batch_size=128):
    """
    Evaluates a fine-tuned model against the proprietary test data.
    :param file_path: The path to the proprietary test data.
    :param model_name: The name of the model. Can be one of t5-small, t5-base, or t5-large.
    :param batch_size: The batch size for inference. Useful if the data is large and cannot fit into memory.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data= load_test_data(file_path)
    tokenizer, model = load_model(model_name, device)
    model.eval()

    test_outputs = mini_batch_inference(data['inputs'], model, tokenizer, batch_size, device)
    final_metrics = evaluate_outputs(test_outputs, data['targets'])
    print(f"Evaluation metrics of the fine tuned {model_name} model on the test set:")
    print(final_metrics)
    return tokenizer, model