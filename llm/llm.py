from transformers import pipeline

class VanillaLLM:
    def __init__(self, model_name="google/flan-t5-large"):
        print("[INFO] Loading LLM...")
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            framework="pt",
        )

    def generate(self, prompt):   # 👈 use this unified method
        # strip leading/trailing spaces
        prompt = prompt.strip()
        response = self.generator(prompt, max_new_tokens=150)
        return response[0]['generated_text']