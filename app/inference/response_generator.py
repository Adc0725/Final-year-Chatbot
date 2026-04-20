import json
import random

from inference.emotion_predictor import EmotionPredictor
from inference.intent_predictor import IntentPredictor
from inference.gemini_predictor import GeminiPredictor
from inference.entity_extractor import EntityExtractor
from inference.safety_filter import SafetyFilter
from inference.response_cleaner import ResponseCleaner


class ResponseGenerator:

    def __init__(self):

        self.emotion_model = EmotionPredictor()
        self.intent_model = IntentPredictor()
        self.dialog_model = GeminiPredictor()
        self.entity_extractor = EntityExtractor()
        self.safety_filter = SafetyFilter()
        self.cleaner = ResponseCleaner()

        with open("data/coping_strategies.json", "r", encoding="utf-8") as f:
            self.coping_strategies = json.load(f)

    def generate(self, user_input):

        # -----------------------------
        # 1. EMOTION DETECTION
        # -----------------------------
        emotions = self.emotion_model.predict_emotions(user_input)

        sorted_emotions = sorted(
            emotions,
            key=lambda x: x["confidence"],
            reverse=True
        )

        primary_emotion = sorted_emotions[0]["emotion"]

        #  NEW: Extract secondary emotions (top 2 excluding primary)
        secondary_emotions = [
            e["emotion"] for e in sorted_emotions[1:3]
        ] if len(sorted_emotions) > 1 else []

        # -----------------------------
        # 2. INTENT DETECTION
        # -----------------------------
        intent = self.intent_model.predict_intent(user_input)

        # -----------------------------
        # 3. ENTITY EXTRACTION
        # -----------------------------
        entities = self.entity_extractor.extract(user_input)

        # -----------------------------
        # 4. GENERATE RESPONSE (Gemini)
        # -----------------------------
        response = self.dialog_model.generate_response(
            user_input=user_input,
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,   # NEW
            intent=intent,
            entities=entities
        )

        # -----------------------------
        # 5. ADD COPING STRATEGY
        # -----------------------------
        if primary_emotion in self.coping_strategies:
            strategy = random.choice(self.coping_strategies[primary_emotion])
            response += f"\n\nYou might find this helpful: {strategy}"

        # -----------------------------
        # 6. SAFETY FILTER (FINAL)
        # -----------------------------
        response = self.safety_filter.filter_response(user_input, response)

        # -----------------------------
        # 7. CLEAN RESPONSE
        # -----------------------------
        response = self.cleaner.clean(response)

        return {
            "emotion": primary_emotion,
            "secondary_emotions": secondary_emotions,  #  useful for debugging/demo
            "intent": intent,
            "entities": entities,
            "response": response
        }