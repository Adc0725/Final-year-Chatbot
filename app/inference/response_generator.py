import json
import random

from inference.emotion_predictor import EmotionPredictor
from inference.dialogpt_predictor import DialogPredictor
from inference.safety_filter import SafetyFilter
from inference.conversation_memory import ConversationMemory
from inference.response_cleaner import ResponseCleaner

class ResponseGenerator:

    def __init__(self):

        self.emotion_model = EmotionPredictor()
        self.dialog_model = DialogPredictor()
        self.safety_filter = SafetyFilter()
        self.memory = ConversationMemory()
        self.cleaner = ResponseCleaner()

        with open("data/coping_strategies.json", "r") as f:
            self.coping_strategies = json.load(f)

    def get_primary_emotion(self, emotions):
        return emotions[0]["emotion"]

   

    def generate(self, user_input):

        # 1. Emotion detection
        emotions = self.emotion_model.predict_emotions(user_input)
        primary_emotion = self.get_primary_emotion(emotions)

        # 2. Get conversation history
        context = self.memory.get_context()

        # 3. Generate response with emotion + context
        response = self.dialog_model.generate_response(
            user_input,
            primary_emotion,
            context
        )

        # 4. Clean response
        response = self.cleaner.clean(response)

        # 5. Safety filter
        response = self.safety_filter.filter_response(user_input, response)

        # 6. Save to memory
        self.memory.add_turn(user_input, response)

        return {
            "emotion": primary_emotion,
            "response": response
        }