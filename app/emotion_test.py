from inference.emotion_predictor import EmotionPredictor

predictor = EmotionPredictor()

text = "I feel extremely anxious about my exams and I'm scared I might fail."

result = predictor.predict(text)

print(result)