import json
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# Import your models
# -----------------------------
from inference.emotion_predictor import EmotionPredictor
from inference.intent_predictor import IntentPredictor


# -----------------------------
# Initialize models
# -----------------------------
print("Loading Emotion and Intent models...")

emotion_model = EmotionPredictor()
intent_model = IntentPredictor()


# -----------------------------
# Build multi-turn samples
# -----------------------------
def build_multi_turn_samples(conversation, max_turns=3):
    """
    Extracts multi-turn conversational samples with memory.
    """

    samples = []
    turns = conversation.get("conversations", [])

    history = []

    for i in range(len(turns) - 1):

        if turns[i]["from"] == "human" and turns[i+1]["from"] == "gpt":

            user = turns[i]["value"]
            bot = turns[i+1]["value"]

            # Append to running history
            history.append({"user": user, "bot": bot})

            # Keep only last N turns
            recent_history = history[-max_turns:]

            # Build context (exclude current turn)
            context = ""
            for h in recent_history[:-1]:
                context += f"User: {h['user']}\nAssistant: {h['bot']}\n"

            current_user = recent_history[-1]["user"]
            current_response = recent_history[-1]["bot"]

            samples.append((context.strip(), current_user, current_response))

    return samples


# -----------------------------
# Format training sample
# -----------------------------
def format_sample(context, user, response, emotion, intent):
    """
    Formats sample into instruction-tuning format
    """

    # Crisis-aware system instruction
    if intent == "crisis":
        system_rule = (
            "You are a mental health assistant. The user may be in emotional distress. "
            "Respond with strong empathy, validate their feelings, and encourage seeking help. "
            "Do NOT provide harmful or unsafe advice."
        )
    else:
        system_rule = (
            "You are a mental health support assistant. "
            "Be empathetic, supportive, and conversational. "
            "Offer helpful guidance where appropriate."
        )

    prompt = f"""### Instruction:
{system_rule}

User Emotion: {emotion}
User Intent: {intent}

Conversation history:
{context}

User: {user}

### Response:
{response}
"""

    return {"text": prompt.strip()}


# -----------------------------
# Main dataset builder
# -----------------------------
def build_dataset(
    output_path="llm_training_data.json",
    max_samples=50000,
    max_turns=3
):

    print("Loading victunes dataset...")

    dataset = load_dataset(
        "victunes/nart-100k-synthetic-buddy-mixed-names",
        split="train"
    )

    final_data = []

    print("\nProcessing conversations...\n")

    for item in tqdm(dataset):

        try:
            samples = build_multi_turn_samples(item, max_turns=max_turns)

            for context, user, response in samples:

                try:
                    # -----------------------------
                    # Emotion prediction
                    # -----------------------------
                    emotions = emotion_model.predict_emotions(user)

                    if not emotions:
                        continue

                    emotions_sorted = sorted(
                        emotions,
                        key=lambda x: x["confidence"],
                        reverse=True
                    )

                    primary_emotion = emotions_sorted[0]["emotion"]

                    # -----------------------------
                    # Intent prediction
                    # -----------------------------
                    intent_result = intent_model.predict_intent(user)

                    if isinstance(intent_result, dict):
                        intent = intent_result.get("intent", "unknown")
                    else:
                        intent = str(intent_result)

                    # -----------------------------
                    # Format sample
                    # -----------------------------
                    formatted = format_sample(
                        context,
                        user,
                        response,
                        primary_emotion,
                        intent
                    )

                    final_data.append(formatted)

                    # Stop early if limit reached
                    if len(final_data) >= max_samples:
                        break

                except Exception as inner_error:
                    continue

            if len(final_data) >= max_samples:
                break

        except Exception as outer_error:
            continue

    # -----------------------------
    # Save dataset
    # -----------------------------
    print(f"\nSaving {len(final_data)} samples to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print("\nDataset build complete!")


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":

    build_dataset(
        output_path="llm_training_data.json",
        max_samples=50000,   # adjust based on GPU
        max_turns=3          # memory size (recommended: 3)
    )