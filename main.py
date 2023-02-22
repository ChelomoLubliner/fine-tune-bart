from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import numpy as np
import warnings
import pandas as pd
import plotly.express as px
warnings.filterwarnings("ignore")
import evaluate
rouge_score = evaluate.load("rouge")
import json

#initialization

access_token = 'YOU_USER_ACCESS_TOKEN'
#!python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('YOU_USER_ACCESS_TOKEN')"
#Change with your path
data_folder = 'YOUR_PATH_WITH_JSON_FILE'
max_input_length = 1024
max_target_length = 256
metric = load_metric("rouge")
model_checkpoint = "facebook/bart-base"

def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["abs_summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract the median scores
    result = {key: value for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def get_data(data_path):
    data = open(data_path)
    MediQA_data = json.load(data)
    gen_keys = MediQA_data.keys()
    final_data = []
    for gen_key in gen_keys:
        ans_keys = MediQA_data[gen_key]['answers'].keys()
        for ans_key in ans_keys:
            dic = {'article': MediQA_data[gen_key]['answers'][ans_key]['article'],
                   'abs_summary': MediQA_data[gen_key]['answers'][ans_key]['answer_abs_summ']}
            final_data.append(dic)
    final_data_df = pd.DataFrame(final_data)
    return final_data_df


def load_file():
    MediQA_data = get_data(data_folder + 'question_driven_answer_summarization_primary_dataset.json')
    print(MediQA_data.shape)
    print(MediQA_data.head())
    MediQA_data.iloc[:442].to_csv(data_folder + 'train.csv')
    MediQA_data.iloc[:442].to_csv(data_folder + 'test.csv')
    return MediQA_data

def df_visualization(MediQA_data):
    df = MediQA_data.copy()
    print(len(df['article'].iloc[0].split()))
    df["words_article"] = df["article"].apply(lambda n: len(n.split()))
    df["words_summary"] = df["abs_summary"].apply(lambda n: len(n.split()))
    print(df.head())
    fig = px.histogram(df, x="words_article",  marginal="box",  hover_data=df.columns)
    fig.show()
    fig = px.histogram(df, x="words_summary",  marginal="box",  hover_data=df.columns)
    fig.show()

def get_args(model_name, batch_size):
    args = Seq2SeqTrainingArguments(
        f"{model_name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,  ## la en mettre 3
        predict_with_generate=True,
        push_to_hub=True)
    return args


if __name__ == '__main__':
    MediQA_data = load_file()
    df_visualization(MediQA_data)
    data_files = {"train": data_folder + "train.csv", "test": data_folder + "test.csv"}
    raw_datasets = load_dataset(data_folder, data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, use_auth_token=access_token)
    args = get_args(model_name='summarizer_MediQA', batch_size=8)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
                model,
                 args,
                 train_dataset=tokenized_datasets["train"],
                 eval_dataset=tokenized_datasets["test"],
                 data_collator=data_collator,
                 tokenizer=tokenizer,
                 compute_metrics=compute_metrics)

    #predictions before fine-tune
    predictions_before = trainer.predict(tokenized_datasets["test"])
    print(predictions_before.metrics)
    np.save('predictions_before_metrics.npy', predictions_before.metrics)

    trainer.train()
    trainer.evaluate()

    predictions = trainer.predict(tokenized_datasets["test"])
    print(' ------------------')
    print('metrics before finetune')
    print(predictions_before.metrics)
    print('metrics after finetune')
    print(predictions.metrics)
    np.save('predictions_after_metrics.npy', predictions.metrics)

    trainer.save_model("./my_model")

