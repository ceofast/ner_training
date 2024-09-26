import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from datasets import load_dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score
from transformers import AutoTokenizer, XLMRobertaForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

def parse_args():
    parser = argparse.ArgumentParser(description="NER Training Script")
    parser.add_argument("--languages", nargs="+", default=["de", "fr", "it", "en"], help="Languages to use")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.629, 0.229, 0.084, 0.059], help="Fractions of data to use for each language")
    parser.add_argument("--model_name", default="xlm-roberta-base", help="Model name to use")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("--output_dir", default="multilingual-xlm-roberta-for-ner", help="Output directory for the model")
    parser.add_argument("--plot_output", default="f1_score_plot.png", help="Output file for F1 score plot")
    return parser.parse_args()

def load_and_process_datasets(langs, fracs):
    panx_ch = defaultdict(DatasetDict)
    for lang, frac in zip(langs, fracs):
        ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
        for split in ds:
            panx_ch[lang][split] = ds[split].shuffle(seed=0).select(range(int(frac*ds[split].num_rows)))
    return panx_ch

def create_tag_names(batch, tags):
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def encode_panx_dataset(corpus, tokenizer):
    return corpus.map(lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True, remove_columns=["langs", "ner_tags", "tokens"])

def align_predictions(predictions, label_ids, index2tag):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    return preds_list, labels_list

def compute_metrics(eval_pred, index2tag):
    y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids, index2tag)
    return {"f1": f1_score(y_true, y_pred)}

def get_f1_score(trainer, dataset):
    return trainer.predict(dataset).metrics["test_f1"]

def evaluate_lang_performance(lang, trainer, panx_ch, tokenizer):
    panx_ds = encode_panx_dataset(panx_ch[lang], tokenizer)
    return get_f1_score(trainer, panx_ds["test"])

def train_on_subset(dataset, num_samples, model_init, training_args, data_collator, compute_metrics, tokenizer):
    train_ds = dataset["train"].shuffle(seed=42).select(range(num_samples))
    valid_ds = dataset["validation"]
    test_ds = dataset["test"]
    training_args.logging_steps = len(train_ds) // training_args.per_device_train_batch_size
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    f1_score = get_f1_score(trainer, test_ds)
    return pd.DataFrame.from_dict(
        {"num_samples": [len(train_ds)], "f1_score": [f1_score]})

def main():
    args = parse_args()
    
    panx_ch = load_and_process_datasets(args.languages, args.fractions)
    
    xlmr_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    tags = panx_ch["de"]["train"].features["ner_tags"].feature
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}
    num_labels = tags.num_classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xlmr_model = XLMRobertaForTokenClassification.from_pretrained(args.model_name, num_labels=num_labels, id2label=index2tag, label2id=tag2index).to(device)
    
    panx_de_encoded = encode_panx_dataset(panx_ch["de"], xlmr_tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        log_level="error",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_steps=1e6,
        weight_decay=0.01,
        logging_steps=len(panx_de_encoded["train"]) // args.batch_size,
        report_to="none",
        push_to_hub=False
    )
    
    data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)
    
    def model_init():
        return xlmr_model
    
    compute_metrics_func = lambda eval_pred: compute_metrics(eval_pred, index2tag)
    
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
        train_dataset=panx_de_encoded["train"],
        eval_dataset=panx_de_encoded["validation"],
        tokenizer=xlmr_tokenizer,
    )
    
    trainer.train()
    
    f1_scores = defaultdict(dict)
    f1_scores["de"]["de"] = get_f1_score(trainer, panx_de_encoded["test"])
    print(f"F1-score of [de] model on [de] dataset: {f1_scores['de']['de']:.3f}")
    
    for lang in ["fr", "it", "en"]:
        f1_scores["de"][lang] = evaluate_lang_performance(lang, trainer, panx_ch, xlmr_tokenizer)
        print(f"F1-score of [de] model on [{lang}] dataset: {f1_scores['de'][lang]:.3f}")
    
    panx_fr_encoded = encode_panx_dataset(panx_ch["fr"], xlmr_tokenizer)
    
    metrics_df = pd.DataFrame()
    for num_samples in [250, 500, 1000, 2000, 4000]:
        metrics_df = metrics_df._append(train_on_subset(panx_fr_encoded, num_samples, model_init, training_args, data_collator, compute_metrics_func, xlmr_tokenizer), ignore_index=True)
    
    fig, ax = plt.subplots()
    ax.axhline(f1_scores["de"]["fr"], ls="--", color="r")
    metrics_df.set_index("num_samples").plot(ax=ax)
    plt.legend(["Zero-shot from de", "Fine-tuned on fr"], loc="lower right")
    plt.ylim((0,1))
    plt.xlabel("Number of Training Samples")
    plt.ylabel("F1 Score")
    plt.savefig(args.plot_output)
    plt.close()

if __name__ == "__main__":
    main()
