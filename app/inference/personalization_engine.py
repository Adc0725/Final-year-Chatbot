from datetime import datetime


class PersonalizationEngine:

    def __init__(self):

        # Session-based temporary preferences
        self.user_preferences = {
            "response_style": "balanced"   # short / balanced / detailed
        }

        # Track previously used coping strategies
        self.used_strategies = []

        # Emotional trend during current session
        self.emotion_history = []


    # -----------------------------------
    # Update Session Profile
    # -----------------------------------
    def update_profile(
        self,
        user_input,
        primary_emotion
    ):

        # Store emotion history
        self.emotion_history.append(primary_emotion)

        # Keep history lightweight
        self.emotion_history = self.emotion_history[-10:]

        text = user_input.lower()

        # -----------------------------
        # Detect explicit preferences
        # -----------------------------
        if any(word in text for word in [
            "short replies",
            "brief replies",
            "keep it short",
            "short response"
        ]):
            self.user_preferences["response_style"] = "short"

        elif any(word in text for word in [
            "detailed replies",
            "long response",
            "explain more",
            "be detailed"
        ]):
            self.user_preferences["response_style"] = "detailed"


    # -----------------------------------
    # Time-of-day context
    # -----------------------------------
    def get_time_context(self):

        hour = datetime.now().hour

        if hour < 12:
            return "morning"

        elif hour < 18:
            return "afternoon"

        return "night"


    # -----------------------------------
    # Get personalization context
    # -----------------------------------
    def get_personalization_context(self):

        time_context = self.get_time_context()

        style = self.user_preferences["response_style"]

        # Emotional trend
        if len(self.emotion_history) >= 3:

            recent = self.emotion_history[-3:]

            if recent.count("sadness") >= 2:
                trend = "persistent sadness"

            elif recent.count("anxiety") >= 2:
                trend = "persistent anxiety"

            else:
                trend = "mixed emotions"

        else:
            trend = "normal"

        return {
            "response_style": style,
            "time_context": time_context,
            "emotion_trend": trend
        }


    # -----------------------------------
    # Prevent repeated coping strategies
    # -----------------------------------
    def get_new_strategy(
        self,
        strategies
    ):

        available = [
            s for s in strategies
            if s not in self.used_strategies
        ]

        # Reset if exhausted
        if not available:
            self.used_strategies = []
            available = strategies

        strategy = available[0]

        self.used_strategies.append(strategy)

        return strategy 
        