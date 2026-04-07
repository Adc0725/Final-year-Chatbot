import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)



# -----------------------------
# Device setup (SAFE)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

use_fp16 = torch.cuda.is_available()

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("empathetic_dialogues")

# -----------------------------
# Format dataset PROPERLY
# -----------------------------
def format_dialogue(example):
    """
    Proper conversational format with emotion conditioning
    """

    user_input = example["utterance"]
    emotion = example.get("context", "neutral")

    # Add emotion conditioning (VERY IMPORTANT)
    text = f"[EMOTION={emotion}] User: {user_input}\nBot:"

    return {"text": text}


dataset = dataset.map(
    format_dialogue,
    remove_columns=dataset["train"].column_names
)

# -----------------------------
# Load tokenizer & model
# -----------------------------
model_name = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fix padding token (CRITICAL for GPT)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

model.to(device)

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

# -----------------------------
# Data collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------------
# Training arguments (STABLE)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./models",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=5e-5,  # slightly higher for GPT fine-tuning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=2,
    num_train_epochs=4,

    warmup_ratio=0.1,
    max_grad_norm=1.0,

    logging_dir="./logs",
    logging_steps=100,

    save_total_limit=2,

    fp16=use_fp16,  # SAFE GPU usage
    report_to="none"  # avoids wandb issues
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save model
# -----------------------------
model.save_pretrained("models/dialogpt_empathetic")
tokenizer.save_pretrained("models/dialogpt_empathetic")


