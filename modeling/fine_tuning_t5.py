import torch
import evaluate
import numpy as np
from modules import prepare_data, plot_error
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
                          AutoTokenizer, AutoModelForSeq2SeqLM, EarlyStoppingCallback, BitsAndBytesConfig)


class T5QLoRA:
    """
    The class used for fine-tuning T5 models on the Disfl-QA dataset.
    Contains methods for model and data preparation.
    The order of use for the methods is as follows:
        1- define_model(...)
        2- preprocess_data(...)
        3- prepare_trainer(...)
        4- fine_tune(...)
        5- plot_training_error(...)
    """
    def __init__(self, model_name):
        """
        Initializes the T5 model based on the given name.
        :param model_name: T5 model name, which can be one of the following: t5-small, t5-base, t5-large, t5-3b, t5-11b
        """
        if model_name not in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
            raise ValueError(f"Invalid model name: {model_name}")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.bnb_config = None
        self.trainer = None
        self.data_collator = None
        self.log_history = None

        self.bleu_metric = evaluate.load("bleu", verbose=False)
        self.rouge_metric = evaluate.load("rouge", verbose=False)
        self.meteor_metric = evaluate.load("meteor", verbose=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def define_model(self):
        """
        Defines the model with proper configurations for PEFT fine-tuning with QLoRA.
        :return: None. It only adjusts the 'model' and 'tokenizer' class attributes.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "v"],  # T5 uses q, v in its attention layers
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def preprocess_data(self, train_data, validation_data, test_data):
        """
        Preprocesses and tokenizes the input data.
        :param train_data: The training data that is to be tokenized.
        :param validation_data: The validation data that is to be tokenized.
        :param test_data: The test data that is to be tokenized.
        :return: the tokenized version of the input data in this order: training, validation, test
        """
        if self.tokenizer is None:
            raise ValueError("Please define tokenizer first, by calling 'define_model' with the proper model name.")
        tokenized_train = prepare_data(train_data, self.tokenizer)
        tokenized_validation = prepare_data(validation_data, self.tokenizer)
        tokenized_test = prepare_data(test_data, self.tokenizer)
        return tokenized_train, tokenized_validation, tokenized_test

    def compute_metrics(self, eval_preds):
        """
        Method for computing the metrics during training for logging and selecting the best model.
        :param eval_preds: The outputs of the model that are to be evaluated with the given metrics.
        :return: a dictionary containing the computed metrics.
        """
        if self.tokenizer is None:
            raise ValueError("Please define tokenizer first, by calling 'define_model' with the proper model name.")

        preds, labels = eval_preds
        preds = np.where(preds != -100,preds,self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100,labels,self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu_score = self.bleu_metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        rouge_result = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_result = self.meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {
                "rouge1": rouge_result["rouge1"],
                "rouge2": rouge_result["rouge2"],
                "rougeL": rouge_result["rougeL"],
                "rougeLsum": rouge_result["rougeLsum"],
                "meteor": meteor_result["meteor"],
                "bleu": bleu_score["bleu"]
                }

    def prepare_trainer(self, tokenized_train_data, tokenized_validation_data, num_train_epochs=3):
        """
        Configures the Transformers trainer and prepares it for fine-tuning.
        :param tokenized_train_data: The tokenized training data that is to be used in the trainer.
        :param tokenized_validation_data: The tokenized validation data that is to be used in the trainer.
        :param num_train_epochs: number of epochs for the training.
        :return: None. Only adjusts the class attributes and prepares the model for fine-tuning.
        """
        if self.tokenizer is None:
            raise ValueError("Please define tokenizer first, by calling 'define_model' with the proper model name.")

        if self.model_name in ["t5-small", "t5-base"]:
            fp16 = True
        else:
            fp16 = False

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./tuned_models/{self.model_name}/t5-qlora-outputs",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=100,
            warmup_steps=100,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            fp16=fp16,
            eval_strategy="steps",
            save_total_limit=2,
            logging_dir=f"./tuned_models/{self.model_name}/logs",
            metric_for_best_model="rougeLsum",
            greater_is_better=True,
            load_best_model_at_end=True,
            save_strategy="steps",
        )

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_validation_data,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    def fine_tune(self, save=True):
        """
        Carries out the QLoRA fine-tuning task.
        :param save: whether to save the model or not.
        :return: None. Only adjusts model parameters through fine-tuning.
        """
        if self.trainer is None:
            raise ValueError("Please define trainer first, by calling 'prepare_trainer' with the proper training"
                             "and validation data.")
        self.trainer.train()
        if save:
            self.model.save_pretrained(f"./tuned_models/{self.model_name}/{self.model_name}-qlora-final")
            self.tokenizer.save_pretrained(f"./tuned_models/{self.model_name}/{self.model_name}-qlora-final")

    def plot_training_error(self):
        """
        Plots the model loss vs training steps for both training and validation datasets, and saves it.
        :return: None. Only saves a plot to disk.
        """
        if self.trainer is None:
            raise ValueError("Please define trainer first, by calling 'prepare_trainer' with the proper training"
                             "and validation data.")
        if len(self.trainer.state.log_history) == 0:
            raise ValueError("Please fine tune the model first in order to plot the error plot.")

        self.log_history = self.trainer.state.log_history
        plot_error(self.log_history, self.model_name)

    def mini_batch_inference(self, input_texts, batch_size=128):
        """
        Carries out the inference in batch mode in order to deal with memory limitations of GPU.
        :param input_texts: The input texts that are to be used for inference and generate outputs.
        :param batch_size: The batch size for inference.
        :return: A list containing all the decoded outputs of the transformer model.
        """
        input_texts = self.tokenizer(input_texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        self.model.eval()
        outputs = []
        for i in range(0, len(input_texts['input_ids']), batch_size):
            batch = input_texts['input_ids'][i:i + batch_size].to(self.device)
            with torch.no_grad():
                outputs += self.tokenizer.batch_decode(self.model.generate(input_ids=batch,
                                                                 max_length=128,
                                                                 ),
                                                  skip_special_tokens=True)
        return outputs
