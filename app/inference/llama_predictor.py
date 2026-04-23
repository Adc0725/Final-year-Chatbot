import torch
import re
from unsloth import FastLanguageModel


class LlamaPredictor:

    def __init__(self, model_path="models/llama_mental_health", max_history=6):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=3048,
            load_in_4bit=True,
        )

        self.model.to(self.device)

        # Conversation memory
        self.chat_history = []
        self.max_history = max_history

    # -----------------------------------
    # Clean response
    # -----------------------------------
    def clean_response(self, text):

        text = re.sub(r"\s+", " ", text).strip()

        # Remove prompt leakage
        if "### Response:" in text:
            text = text.split("### Response:")[-1].strip()

        # Ensure proper ending
        if not text.endswith((".", "!", "?")):
            text += "."

        # Remove repetition
        sentences = text.split(".")
        seen = []
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                seen.append(s)

        return ". ".join(seen).strip()

    # -----------------------------------
    # Build prompt (UPDATED WITH INTENT)
    # -----------------------------------
    def build_prompt(self, user_input, primary_emotion, secondary_emotions=None, intent="general"):

        history = ""

        for turn in self.chat_history[-self.max_history:]:
            history += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

        # -----------------------------
        # EMOTION TONE
        # -----------------------------
        if primary_emotion == "joy":
            tone = "Celebrate the user's positive feelings and reinforce happiness."

        elif primary_emotion == "sadness":
            tone = "Be gentle, empathetic, and comforting."

        elif primary_emotion == "anxiety":
            tone = "Be calming and reassuring. Reduce worry."

        elif primary_emotion == "anger":
            tone = "Remain calm and help the user process anger constructively."

        else:
            tone = "Be supportive, friendly, and conversational."

        # -----------------------------
        # SECONDARY EMOTION
        # -----------------------------
        secondary_tone = ""

        if secondary_emotions:
            if "anxiety" in secondary_emotions:
                secondary_tone += " Add reassurance."

            if "sadness" in secondary_emotions:
                secondary_tone += " Show deeper empathy."

            if "anger" in secondary_emotions:
                secondary_tone += " Avoid confrontation."

            if "joy" in secondary_emotions:
                secondary_tone += " Reinforce positivity."

        # -----------------------------
        # INTENT CONTROL
        # -----------------------------
        if intent == "venting":
            intent_instruction = (
                "Let the user express themselves. Focus on listening and validating feelings."
            )

        elif intent == "seeking_advice":
            intent_instruction = (
                "Provide gentle and practical coping suggestions."
            )

        elif intent == "crisis":
            intent_instruction = (
                "Respond with urgency, empathy, and encourage seeking real-world help."
            )

        elif intent == "greeting":
            intent_instruction = (
                "Respond warmly and invite conversation."
            )

        else:
            intent_instruction = (
                "Provide a balanced supportive response."
            )

        # -----------------------------
        # FINAL PROMPT
        # -----------------------------
        prompt = f"""### Instruction:
You are a mental health support assistant.

Rules:
- Be empathetic and supportive
- Do NOT provide medical or clinical advice
- Do NOT encourage harmful behavior
- Always complete your thoughts clearly
- Keep responses natural and human-like (3–6 sentences)

Emotion Tone:
{tone}
{secondary_tone}

Intent Guidance:
{intent_instruction}

Conversation History:
{history}

User: {user_input}

### Response:
"""

        return prompt

    # -----------------------------------
    # Generate response
    # -----------------------------------
    def generate_response(
        self,
        user_input,
        primary_emotion="neutral",
        secondary_emotions=None,
        intent="general"
    ):

        prompt = self.build_prompt(
            user_input,
            primary_emotion,
            secondary_emotions,
            intent
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3048
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=180,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = self.clean_response(decoded)

        # -----------------------------
        # FALLBACK FOR BROKEN OUTPUTS
        # -----------------------------
        if len(response.split()) < 5:
            response = (
                "I'm really sorry you're feeling this way. "
                "Do you want to talk more about what's been going on?"
            )

        # -----------------------------
        # SAVE MEMORY
        # -----------------------------
        self.chat_history.append({
            "user": user_input,
            "bot": response
        })

        return response