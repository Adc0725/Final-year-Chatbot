import json
import torch
import numpy as np

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -----------------------------
# Load dataset (SEPARATE FILES)
# -----------------------------
def load_dataset_file(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    return texts, labels


# -----------------------------
# Prepare dataset (NO SPLITTING)
# -----------------------------
def prepare_datasets(train_path, val_path):

    train_texts, train_labels = load_dataset_file(train_path)
    val_texts, val_labels = load_dataset_file(val_path)

    label_list = sorted(list(set(train_labels)))

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    train_encoded = [label2id[l] for l in train_labels]
    val_encoded = [label2id[l] for l in val_labels]

    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_encoded
    })

    val_dataset = Dataset.from_dict({
        "text": val_texts,
        "label": val_encoded
    })

    return train_dataset, val_dataset, label2id, id2label


# -----------------------------
# Tokenization
# -----------------------------
def tokenize(dataset, tokenizer):

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    return dataset.map(tokenize_fn, batched=True)


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# -----------------------------
# Main
# -----------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 🔥 IMPORTANT: Use separate datasets
    train_dataset, val_dataset, label2id, id2label = prepare_datasets(
        "intent_train.json",
        "intent_val.json"
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = tokenize(train_dataset, tokenizer)
    val_dataset = tokenize(val_dataset, tokenizer)

    # 🔥 Shuffle training data (improves generalization)
    train_dataset = train_dataset.shuffle(seed=42)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 🔥 Add dropout to reduce overfitting
    model.config.dropout = 0.3
    model.config.attention_dropout = 0.3

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./intent_model",

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        num_train_epochs=3,   # 🔥 Reduced epochs

        weight_decay=0.01,

        logging_steps=50,

        fp16=torch.cuda.is_available(),

        # 🔥 IMPORTANT: enable evaluation properly
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()

    # -----------------------------
    # Evaluate
    # -----------------------------
    results = trainer.evaluate()

    print("\nFinal Results:\n", results)

    # -----------------------------
    # Save model
    # -----------------------------
    trainer.save_model("models/intent_classifier")
    tokenizer.save_pretrained("models/intent_classifier")


if __name__ == "__main__":
    main()