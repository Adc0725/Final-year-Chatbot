import re


class SafetyFilter:

    def __init__(self):

        # -----------------------------
        # Crisis keywords
        # -----------------------------
        self.crisis_keywords = [
            "suicide", "kill myself", "end my life", "take my life",
            "i want to die", "i can't go on", "i cant go on",
            "self harm", "self-harm", "hurt myself", "cutting myself",
            "no reason to live", "not worth living", "better off dead",
            "want to disappear", "end it all", "overdose"
        ]

        # -----------------------------
        # Harmful instruction requests
        # -----------------------------
        self.harmful_requests = [

            "how to kill",
            "how to make poison",
            "how to self harm",
            "how to overdose",
            "best way to die", "kill someone", "hurt someone", "hurt them",
            "attack", "stab", "shoot someone", "murder"
        ]

        # -----------------------------
        # Profanity
        # -----------------------------
        self.profanity = [
            "idiot",
            "stupid",
            "hate you", "fuck", "shit", "bitch", "asshole", "bastard",
            "damn", "crap", "piss", "dick", "cunt"
        ]

        # -----------------------------
        # Off-topic detection
        # -----------------------------
        self.off_topic_keywords = [
           r"\b(write me (a |some )?(code|script|program|essay|email))\b",
            r"\b(who (is|was) (the )?(president|prime minister|ceo))\b",
            r"\b(what is the (weather|stock|price|capital))\b",
            r"\b(translate (this|to|from))\b",
            r"\b(math|calculate|solve|equation)\b",
            r"\b(recipe|how (do i|to) cook)\b",
        ]


    # -----------------------------------
    # Crisis Detection
    # -----------------------------------
    def detect_crisis(self, text):

        text = text.lower()

        return any(
            keyword in text
            for keyword in self.crisis_keywords
        )


    # -----------------------------------
    # Harmful Request Detection
    # -----------------------------------
    def detect_harmful_request(self, text):

        text = text.lower()

        return any(
            keyword in text
            for keyword in self.harmful_requests
        )


    # -----------------------------------
    # Off-topic Detection
    # -----------------------------------
    def detect_offtopic(self, text):

        text = text.lower()

        return any(
            keyword in text
            for keyword in self.off_topic_keywords
        )


    # -----------------------------------
    # Crisis Template
    # -----------------------------------
    def crisis_response(self):

        return (
             "I can hear that you're in a lot of pain right now, and I'm really glad you're talking. "
            "You're not alone — what you're feeling matters, and so do you. \n\n"
            "Please consider reaching out to someone who can truly support you:\n"
            "- Talk to a trusted friend, family member, or someone close to you.\n"
            "- Contact a mental health professional or your nearest health centre.\n"
            "- If you're in immediate danger, please call your local emergency number.\n\n"
            "You deserve real support right now. I'm here to listen, but please reach out to someone who can help."
        )


    # -----------------------------------
    # Harmful Request Template
    # -----------------------------------
    def harmful_request_response(self):

        return (
            "I can't help with harmful or dangerous activities. "
            "If something difficult is happening, I'm here to support you in a safe and positive way."
        )


    # -----------------------------------
    # Off-topic Template
    # -----------------------------------
    def offtopic_response(self):

        return (
            "I'm designed mainly for supportive wellness conversations, emotional support, and healthy coping discussions." 
            "If you'd like to talk about something else, I'm here to listen in a safe and positive way."
        )


    # -----------------------------------
    # Main Input Safety Layer
    # -----------------------------------
    def check_input(self, user_input):

        if self.detect_crisis(user_input):
            return self.crisis_response()

        if self.detect_harmful_request(user_input):
            return self.harmful_request_response()

        if self.detect_offtopic(user_input):
            return self.offtopic_response()

        return None


    # -----------------------------------
    # Output Filtering
    # -----------------------------------
    def filter_response(
        self,
        user_input,
        generated_response
    ):

        # Remove unsafe words
        generated_response = re.sub(
            r"\b(kill|die|harm)\b",
            "",
            generated_response,
            flags=re.IGNORECASE
        )

        # Remove profanity
        for word in self.profanity:

            generated_response = re.sub(
                rf"\b{word}\b",
                "",
                generated_response,
                flags=re.IGNORECASE
            )

        return generated_response.strip()