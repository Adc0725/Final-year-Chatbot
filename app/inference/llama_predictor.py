import re
import requests
from dotenv import load_dotenv
import os

load_dotenv()


class LlamaPredictor:

    def __init__(self, api_key, max_history=10):

        self.api_key = api_key

        self.url = (
            "https://api.mistral.ai/v1/chat/completions"
        )

        # Conversation memory
        self.chat_history = []

        self.max_history = max_history


    # -----------------------------------
    # Clean response
    # -----------------------------------
    def clean_response(self, text):

        text = re.sub(r"\s+", " ", text).strip()

        if "### Response:" in text:

            text = text.split(
                "### Response:"
            )[-1].strip()

        if not text.endswith((".", "!", "?")):
            text += "."

        sentences = text.split(".")

        seen = []

        for s in sentences:

            s = s.strip()

            if s and s not in seen:
                seen.append(s)

        return ". ".join(seen).strip()


    # -----------------------------------
    # Build prompt
    # -----------------------------------
    def build_prompt(
        self,
        user_input,
        primary_emotion,
        secondary_emotions=None,
        intent="general",
        personalization_context=None
    ):

        history = ""

        for turn in self.chat_history[-self.max_history:]:

            history += (
                f"User: {turn['user']}\n"
                f"Assistant: {turn['bot']}\n"
            )

        # -----------------------------
        # Emotion Tone
        # -----------------------------
        if primary_emotion == "joy":

            tone = (
                "Celebrate the user's positive feelings "
                "and reinforce happiness."
            )

        elif primary_emotion == "sadness":

            tone = (
                "Be gentle, empathetic, and comforting."
            )

        elif primary_emotion == "anxiety":

            tone = (
                "Be calming and reassuring."
            )

        elif primary_emotion == "anger":

            tone = (
                "Remain calm and help the user "
                "process anger constructively."
            )

        else:

            tone = (
                "Be supportive, friendly, and conversational."
            )

        # -----------------------------
        # Secondary emotions
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
        # Intent guidance
        # -----------------------------
        if intent == "venting":

            intent_instruction = (
                "Focus on listening and validating feelings."
            )

        elif intent == "seeking_advice":

            intent_instruction = (
                "Provide gentle and practical coping suggestions."
            )

        elif intent == "crisis":

            intent_instruction = (
                "Respond carefully and encourage seeking real-world support."
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
        # Personalization
        # -----------------------------
        personalization_text = ""

        if personalization_context:

            style = personalization_context["response_style"]

            time_context = personalization_context["time_context"]

            emotion_trend = personalization_context["emotion_trend"]

            personalization_text += (
                f"\nConversation Style: {style}"
            )

            personalization_text += (
                f"\nTime Context: {time_context}"
            )

            personalization_text += (
                f"\nEmotion Trend: {emotion_trend}"
            )

        prompt = f"""

You are an AI mental wellness support assistant designed ONLY for supportive emotional wellness conversations.

Core Purpose:
- Provide emotional support
- Encourage healthy coping habits
- Offer reflective listening
- Maintain supportive and empathetic conversations
- Help users feel heard and emotionally supported

STRICT RULES:
- Do NOT provide medical advice
- Do NOT provide therapy or diagnosis
- Do NOT claim to be a licensed professional
- Do NOT provide legal, financial, political, or technical advice
- Do NOT engage in coding, hacking, cybersecurity, or software engineering discussions
- Do NOT discuss explicit, sexual, violent, or dangerous topics
- Do NOT provide instructions for harmful activities
- Do NOT roleplay unsafe scenarios
- Do NOT generate off-topic educational content
- Do NOT continue conversations unrelated to emotional wellness support
- If a user goes off-topic, gently redirect the conversation back toward wellbeing, emotions, stress management, or supportive discussion

Conversation Style Rules:
- Be warm, calm, empathetic, and supportive
- Validate emotions naturally
- Keep responses conversational and human-like
- Avoid sounding robotic or overly clinical
- Keep responses between 3–6 sentences
- Avoid repetitive phrasing
- Maintain emotional safety at all times

Emotion Tone:
{tone}

Additional Tone:
{secondary_tone}

Intent Guidance:
{intent_instruction}

Personalization Context:
{personalization_text}

Conversation History:
{history}

Current User Message:
{user_input}

Assistant Response:
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
        intent="general",
        personalization_context=None
    ):

        prompt = self.build_prompt(
            user_input,
            primary_emotion,
            secondary_emotions,
            intent,
            personalization_context
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {

            "model": "mistral-small",

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            "temperature": 0.7,

            "max_tokens": 200
        }

        try:

            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=15
            )

            result = response.json()

            if "error" in result:

                print(
                    f"[Mistral API Error] {result['error']}"
                )

                generated_text = ""

            elif not result.get("choices"):

                print(
                    f"[Mistral API] Unexpected response format: {result}"
                )

                generated_text = ""

            else:

                generated_text = (
                    result["choices"][0]["message"]["content"]
                )

        except requests.exceptions.Timeout:

            print("[Mistral API] Request timed out.")

            generated_text = ""

        except Exception as e:

            print(
                f"[Mistral API] Unexpected error: {e}"
            )

            generated_text = ""

        response = self.clean_response(generated_text)

        # -----------------------------
        # Fallbacks
        # -----------------------------
        if (
            not generated_text.strip()
            or len(response.split()) < 5
        ):

            fallbacks = {

                "joy": (
                    "That's wonderful to hear. "
                    "I'm glad things are going well for you."
                ),

                "sadness": (
                    "I'm sorry you're feeling this way. "
                    "I'm here to listen if you'd like to share more."
                ),

                "anxiety": (
                    "That sounds overwhelming right now. "
                    "Take things one step at a time."
                ),

                "anger": (
                    "It's understandable to feel frustrated sometimes. "
                    "I'm here to listen."
                )
            }

            response = fallbacks.get(
                primary_emotion,
                "I'm here to chat and support you."
            )

        # Save conversation memory
        self.chat_history.append({

            "user": user_input,

            "bot": response
        })

        return response