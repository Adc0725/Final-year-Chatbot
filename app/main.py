from inference.response_generator import ResponseGenerator

bot = ResponseGenerator()

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Take care of yourself. I'm here whenever you need support.")
        break

    result = bot.generate(user_input)

    print(f"Detected Emotion: {result['emotion']}")
    print(f"Bot: {result['response']}\n")