import json
import random

intents = ["venting", "seeking_advice", "asking_question", "greeting", "crisis"]

emotion_words = [
    "stressed", "overwhelmed", "anxious", "sad",
    "frustrated", "tired", "confused"
]

advice_phrases = [
    "what should I do",
    "how do I handle this",
    "any advice",
    "what can help",
    "what do you suggest"
]

question_starters = ["why", "how", "what", "is it normal that"]

greetings = ["hi", "hello", "hey", "good evening"]

crisis_phrases = [
    "I can't go on",
    "I want to disappear",
    "I feel hopeless",
    "there is no point anymore"
]

def generate_text(intent):

    if intent == "venting":
        return random.choice([
            f"I feel {random.choice(emotion_words)} lately",
            f"Everything is making me feel {random.choice(emotion_words)}",
            f"I've been really {random.choice(emotion_words)} recently"
        ])

    elif intent == "seeking_advice":
        return random.choice([
            f"I'm {random.choice(emotion_words)}, {random.choice(advice_phrases)}?",
            f"{random.choice(advice_phrases)} about feeling {random.choice(emotion_words)}?",
            f"I feel {random.choice(emotion_words)}, {random.choice(advice_phrases)}"
        ])

    elif intent == "asking_question":
        return random.choice([
            f"{random.choice(question_starters)} I feel {random.choice(emotion_words)}?",
            f"{random.choice(question_starters)} does stress affect people?",
            f"{random.choice(question_starters)} is this happening to me?"
        ])

    elif intent == "greeting":
        return random.choice(greetings)

    elif intent == "crisis":
        return random.choice(crisis_phrases)


def add_noise(text):

    noise = [
        "",
        " honestly",
        " lately",
        " and I don’t know why",
        " and it's getting worse"
    ]

    return text + random.choice(noise)


def generate_dataset(size=10000):

    data = []

    for _ in range(size):
        intent = random.choice(intents)
        text = generate_text(intent)
        text = add_noise(text)

        data.append({
            "text": text.strip(),
            "label": intent
        })

    return data


if __name__ == "__main__":

    train = generate_dataset(8000)
    val = generate_dataset(2000)

    with open("intent_train.json", "w") as f:
        json.dump(train, f, indent=2)

    with open("intent_val.json", "w") as f:
        json.dump(val, f, indent=2)

    print("Generated large dataset!")