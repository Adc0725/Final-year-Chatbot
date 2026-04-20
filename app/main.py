from inference.response_generator import ResponseGenerator


def main():

    print("\n=== Mental Health Chatbot (Gemini Powered) ===")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    bot = ResponseGenerator()

    while True:

        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\nBot: Take care of yourself. I'm here whenever you need support.\n")
                break

            if not user_input:
                continue

            result = bot.generate(user_input)

            print("\n--- Analysis ---")
            print(f"Emotion : {result['emotion']}")
            print(f"Intent  : {result['intent']}")

            # Optional (only if you included entities in return)
            if "entities" in result:
                print(f"Entities: {result['entities']}")

            print("\n--- Response ---")
            print(f"{result['response']}\n")

        except KeyboardInterrupt:
            print("\n\nBot: Session ended. Take care!\n")
            break

        except Exception as e:
            print("\n[Error] Something went wrong:")
            print(str(e))
            print("Please try again.\n")


if __name__ == "__main__":
    main()