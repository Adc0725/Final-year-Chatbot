import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from dialogpt_metrics import DialogPTEvaluator


# -----------------------------
# Load trained model
# -----------------------------
model_path = "models/dialogpt_empathetic"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)


# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("empathetic_dialogues")


def format_dialogue(example):
    user_input = example["utterance"]
    emotion = example.get("context", "neutral")

    text = f"[EMOTION={emotion}] User: {user_input}\nBot:"
    return {"text": text}


dataset = dataset.map(
    format_dialogue,
    remove_columns=dataset["train"].column_names
)


# -----------------------------
# Evaluation
# -----------------------------
evaluator = DialogPTEvaluator(model, tokenizer, device)

print("\nRunning evaluation...\n")

# Perplexity
perplexity = evaluator.compute_perplexity(dataset["validation"])
print("Perplexity:", perplexity)

# Generate responses
preds, refs = evaluator.generate_responses(dataset["validation"])

# Semantic similarity
similarity = evaluator.compute_semantic_similarity(preds, refs)
print("Semantic Similarity:", similarity)

# Quality
quality = evaluator.response_quality(preds)
print("Quality:", quality)

# Plot
evaluator.plot_metrics(perplexity, similarity, quality)