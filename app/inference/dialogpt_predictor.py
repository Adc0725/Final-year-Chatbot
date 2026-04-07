import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class DialogPredictor:

    def __init__(self, model_path="models/dialogpt_empathetic"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Dialog Model running on:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

        self.model.eval()

    def generate_response(self, user_input, emotion, context=""):

        prompt = (
            f"Emotion: {emotion}\n"
            f"{context}"
            f"User: {user_input}\nBot:"
        )

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            inputs,
            max_length=150,
            temperature=0.7,      # calmer responses
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.split("Bot:")[-1].strip()