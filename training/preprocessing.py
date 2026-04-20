import json
import os
import random
import re
from collections import Counter


# -----------------------------
# Load emotion mapping
# -----------------------------
def load_mapping():
    mapping_path = os.path.join("data", "emotion_mapping.json")
    with open(mapping_path, "r") as f:
        return json.load(f)


# -----------------------------
# Load label schema
# -----------------------------
def load_label_schema():
    label_path = os.path.join("data", "labels.json")
    with open(label_path, "r") as f:
        labels_data = json.load(f)

    id2label = labels_data["id2label"]
    label_schema = [id2label[str(i)] for i in range(len(id2label))]

    return label_schema


# -----------------------------
# Normalize text
# -----------------------------
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"http\S+", "", text)
    text = text.strip()
    return text


# -----------------------------
# Clean GoEmotions labels
# -----------------------------
def clean_goemotions_labels(labels, mapping_dict):
    labels = [l for l in labels if l in mapping_dict]

    if "neutral" in labels and len(labels) > 1:
        labels.remove("neutral")

    if len(labels) > 3:
        return None

    if len(labels) == 0:
        return None

    return labels


# -----------------------------
# Encode labels to vector
# -----------------------------
def encode_labels(original_labels, mapping_dict, label_schema):
    vector = [0.0] * len(label_schema)
    label2id = {label: i for i, label in enumerate(label_schema)}

    for orig in original_labels:
        mapped_label = mapping_dict[orig]

        if mapped_label in label2id:
            index = label2id[mapped_label]
            vector[index] = 1.0

    return vector


# -----------------------------
# Main preprocessing pipeline
# -----------------------------
def preprocess_dataset(dataset, tokenizer):

    mapping = load_mapping()
    label_schema = load_label_schema()

    goemotion_names = dataset["train"].features["labels"].feature.names

    # -----------------------------
    # Synonym augmentation
    # -----------------------------
    synonym_dict = {
        "happy": ["joyful", "glad", "delighted"],
        "sad": ["unhappy", "down", "depressed"],
        "angry": ["mad", "furious", "irritated"],
        "stressed": ["overwhelmed", "pressured"],
        "anxious": ["nervous", "worried"]
    }

    def synonym_augmentation(text):
        words = text.split()
        new_words = []

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)

            if clean_word in synonym_dict and random.random() < 0.3:
                new_words.append(random.choice(synonym_dict[clean_word]))
            else:
                new_words.append(word)

        return " ".join(new_words)

    # -----------------------------
    # FILTER STEP (IMPORTANT FIX)
    # -----------------------------
    def filter_invalid(example):
        original_labels = [goemotion_names[i] for i in example["labels"]]
        cleaned_labels = clean_goemotions_labels(original_labels, mapping)
        return cleaned_labels is not None

    filtered_dataset = dataset.filter(filter_invalid)

    # -----------------------------
    # TOKENIZATION STEP
    # -----------------------------
    def tokenize_and_encode(example):

        text = normalize_text(example["text"])

        if random.random() < 0.2:
            text = synonym_augmentation(text)

        original_labels = [goemotion_names[i] for i in example["labels"]]
        cleaned_labels = clean_goemotions_labels(original_labels, mapping)

        tokenized = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_token_type_ids=False  # DistilBERT fix
        )

        tokenized["labels"] = encode_labels(
            cleaned_labels,
            mapping,
            label_schema
        )

        return tokenized

    encoded_dataset = filtered_dataset.map(
        tokenize_and_encode,
        remove_columns=dataset["train"].column_names
    )

    return encoded_dataset


# -----------------------------
# Emotion keyword detection
# -----------------------------
emotion_keywords = [
    "worried",
    "anxious",
    "nervous",
    "overwhelmed",
    "stressed",
    "panic"
]


def contains_emotion_keyword(text):
    for word in emotion_keywords:
        if word in text:
            return True
    return False


# -----------------------------
# Emotion-aware oversampling
# -----------------------------
def oversample_dataset(dataset, label_schema, target_size=2000):

    emotion_counts = Counter()

    for example in dataset:
        for i, value in enumerate(example["labels"]):
            if value == 1.0:
                emotion_counts[label_schema[i]] += 1

    augmented_data = list(dataset)

    # Keyword-based augmentation
    for example in dataset:
        text = example.get("text", "")
        if contains_emotion_keyword(text):
            augmented_data.append(example)

    # Class balancing
    for emotion, count in emotion_counts.items():

        if count < target_size:

            needed = target_size - count
            emotion_index = label_schema.index(emotion)

            emotion_samples = [
                ex for ex in dataset
                if ex["labels"][emotion_index] == 1.0
            ]

            if emotion_samples:
                for _ in range(needed):
                    augmented_data.append(random.choice(emotion_samples))

    return augmented_data