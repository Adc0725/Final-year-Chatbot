import torch
import re
from unsloth import FastLanguageModel


class LlamaPredictor:

    def __init__(self, model_path="models/llama_mental_health", max_history=6):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model (IMPORTANT: Unsloth loader)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=3048,
            load_in_4bit=True,
        )

        self.model.to(self.device)

        # Conversation memory
        self.chat_history = []
        self.max_history = max_history

    # -----------------------------------
    # Clean response
    # -----------------------------------
    def clean_response(self, text):

        text = re.sub(r"\s+", " ", text)

        # Remove unfinished endings
        text = text.strip()

        if not text.endswith((".", "!", "?")):
            text += "."

        # Remove repetition
        sentences = text.split(".")
        seen = []
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                seen.append(s)

        return ". ".join(seen).strip()

    # -----------------------------------
    # Build prompt (LLAMA FORMAT)
    # -----------------------------------
    def build_prompt(self, user_input, primary_emotion,secondary_emotions=None):

        history = ""

        for turn in self.chat_history[-self.max_history:]:
            history += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

        # -----------------------------
        # EMOTION-AWARE TONE
        # -----------------------------
        if primary_emotion == "joy":
            tone = (
                "Celebrate the user's positive feelings. Reinforce their happiness "
                "and encourage them to continue positive behaviors."
            )

        elif primary_emotion == "sadness":
            tone = (
                "Be gentle, empathetic, and comforting. Acknowledge their feelings "
                "and offer emotional support."
            )

        elif primary_emotion == "anxiety":
            tone = (
                "Be calming and reassuring. Help reduce worry and suggest grounding "
                "or relaxation techniques."
            )

        elif primary_emotion == "anger":
            tone = (
                "Remain calm and non-judgmental. Help the user process their anger "
                "and guide them toward constructive coping strategies."
            )


        else:  # neutral
            tone = (
                "Be supportive, friendly, and conversational while maintaining "
                "empathy and clarity."
            ) 

        
        # SECONDARY EMOTION INFLUENCE
        
        secondary_tone = ""

        if secondary_emotions:
            if "anxiety" in secondary_emotions:
                secondary_tone += " Also be slightly reassuring and reduce worry."

            if "sadness" in secondary_emotions:
                secondary_tone += " Show deeper empathy and emotional validation."

            if "anger" in secondary_emotions:
                secondary_tone += " Remain calm and avoid sounding confrontational."

            
            if "joy" in secondary_emotions:
                secondary_tone += " Reinforce positivity where appropriate."

        
        # FINAL PROMPT
        
        prompt = f"""### Instruction:
    You are a professional mental health support assistant.

    Your responses must:
    - Be empathetic, supportive, and human-like
    - Be clear and easy to understand
    - Provide helpful guidance when appropriate
    - Be moderately detailed (not too short, not overly long)
    - Avoid repeating phrases or sounding robotic

    {tone} 
    {secondary_tone}

    Conversation history:
    {history}

    User: {user_input}

    ### Response:
    """

        return prompt

    
    # Generate response
    
    def generate_response(self, user_input, primary_emotion="neutral", secondary_emotions=None):

        prompt = self.build_prompt(user_input, primary_emotion,secondary_emotions)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],  
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only response
        response = decoded.split("### Response:")[-1].strip()
        response = self.clean_response(response)

        # Save memory
        self.chat_history.append({
            "user": user_input,
            "bot": response
        })

        return response