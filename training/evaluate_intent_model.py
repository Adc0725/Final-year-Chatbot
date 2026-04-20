import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import json


# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(path="intent_dataset.json"):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    label_list = sorted(list(set(labels)))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    encoded_labels = [label2id[l] for l in labels]

    dataset = Dataset.from_dict({
        "text": texts,
        "label": encoded_labels
    })

    return dataset, label_list, label2id, id2label


# -----------------------------
# Predict
# -----------------------------
def predict(model, tokenizer, dataset, device):

    all_preds = []
    all_probs = []
    all_labels = []

    for sample in dataset:

        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy()[0])
        all_preds.append(np.argmax(probs.cpu().numpy()))
        all_labels.append(sample["label"])

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


# -----------------------------
# Confusion Matrix
# -----------------------------
def plot_confusion_matrix(labels, preds, label_names):

    cm = confusion_matrix(labels, preds)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=label_names,
                yticklabels=label_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Intent Confusion Matrix")
    plt.show()


# -----------------------------
# ROC Curve (ALL ON ONE GRAPH)
# -----------------------------
def plot_roc(all_labels, all_probs, label_names):

    binarized = label_binarize(all_labels, classes=list(range(len(label_names))))

    plt.figure()

    for i in range(len(label_names)):
        fpr, tpr, _ = roc_curve(binarized[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (All Intents)")
    plt.legend()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, label_names, label2id, id2label = load_dataset()

    tokenizer = AutoTokenizer.from_pretrained("models/intent_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("models/intent_classifier")

    model.to(device)
    model.eval()

    preds, probs, labels = predict(model, tokenizer, dataset, device)

    print("\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=label_names))

    plot_confusion_matrix(labels, preds, label_names)
    plot_roc(labels, probs, label_names)


if __name__ == "__main__":
    main()