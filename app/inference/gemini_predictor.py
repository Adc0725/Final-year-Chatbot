import os
from dotenv import load_dotenv
import google.generativeai as genai


# -----------------------------
# LOAD API KEY
# -----------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file.")

genai.configure(api_key=api_key)


class GeminiPredictor:

    def __init__(self, model_name="gemini-2.5-pro", max_history=6):

        self.model = genai.GenerativeModel(model_name)
        self.chat_history = []
        self.max_history = max_history

    # -----------------------------------
    # Tone mapping (PRIMARY emotion)
    # -----------------------------------
    def get_primary_tone(self, emotion):

        tone_map = {
            "joy": (
                "Celebrate the user's positive feelings. Reinforce their happiness "
                "and encourage continued positive behavior."
            ),
            "sadness": (
                "Be gentle, empathetic, and comforting. Acknowledge their feelings "
                "and provide emotional support."
            ),
            "anxiety": (
                "Be calming and reassuring. Help reduce worry and suggest grounding "
                "or relaxation techniques."
            ),
            "anger": (
                "Remain calm and non-judgmental. Help the user process their anger "
                "and guide them toward constructive coping strategies."
            ),
            "neutral": (
                "Be supportive, friendly, and conversational while maintaining empathy."
            )
        }

        return tone_map.get(emotion, tone_map["neutral"])

    # -----------------------------------
    # Secondary tone (WEAKER emotions)
    # -----------------------------------
    def get_secondary_tone(self, secondary_emotions):

        secondary_tone = ""

        if not secondary_emotions:
            return secondary_tone

        if "anxiety" in secondary_emotions:
            secondary_tone += " Also be slightly reassuring and reduce worry."

        if "sadness" in secondary_emotions:
            secondary_tone += " Show deeper empathy and emotional validation."

        if "anger" in secondary_emotions:
            secondary_tone += " Remain calm and avoid sounding confrontational."

        if "joy" in secondary_emotions:
            secondary_tone += " Reinforce positivity where appropriate."

        return secondary_tone

    # -----------------------------------
    # Build prompt
    # -----------------------------------
    def build_prompt(
        self,
        user_input,
        primary_emotion,
        secondary_emotions,
        intent,
        entities,
        history
    ):

        primary_tone = self.get_primary_tone(primary_emotion)
        secondary_tone = self.get_secondary_tone(secondary_emotions)

        prompt = f"""
You are a professional mental health support assistant.

=====================
CORE RULES (STRICT)
=====================
- You MUST be empathetic, supportive, and human-like
- You MUST NOT provide medical, clinical, or diagnostic advice
- You MUST NOT suggest medication or treatments
- You MUST NOT replace a licensed therapist or professional
- If serious distress is detected, gently encourage seeking professional help

=====================
SCOPE CONTROL
=====================
- Keep the conversation focused on emotional wellbeing and mental health support
- If the user tries to redirect to unrelated topics, gently guide it back
- Do NOT engage in harmful, unethical, or unsafe discussions

=====================
RESPONSE STYLE
=====================
- Be clear and moderately detailed (not too short, not too long)
- Avoid repetition and robotic phrasing
- Use natural, conversational language

=====================
EMOTIONAL GUIDANCE
=====================
Primary Emotion: {primary_emotion}
{primary_tone}

Secondary Emotional Signals: {secondary_emotions}
{secondary_tone}

=====================
USER CONTEXT
=====================
Intent: {intent}
Key Details: {entities}

=====================
CONVERSATION HISTORY
=====================
{history}

=====================
USER MESSAGE
=====================
{user_input}

=====================
RESPONSE
=====================
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
        intent="unknown",
        entities=None
    ):

        # Build history
        history_text = ""
        for turn in self.chat_history[-self.max_history:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

        prompt = self.build_prompt(
            user_input,
            primary_emotion,
            secondary_emotions,
            intent,
            entities,
            history_text
        )

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 200,
            }
        )

        text = response.text.strip()

        # Save memory
        self.chat_history.append({
            "user": user_input,
            "bot": text
        })

        return text