import re

class EntityExtractor:

    def __init__(self):

        self.stressors = ["exam", "school", "work", "job", "relationship", "family"]
        self.time_words = ["today", "tomorrow", "next week", "tonight"]
        self.symptoms = ["headache", "insomnia", "panic", "tired"]

    def extract(self, text):

        text = text.lower()

        entities = {
            "stressor": [],
            "time": [],
            "symptom": []
        }

        for word in self.stressors:
            if word in text:
                entities["stressor"].append(word)

        for word in self.time_words:
            if word in text:
                entities["time"].append(word)

        for word in self.symptoms:
            if word in text:
                entities["symptom"].append(word)

        return entities