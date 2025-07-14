from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMHelperDeepseek:
    def __init__(self,
                 model_name="deepseek-ai/deepseek-coder-33b-instruct",
                 max_new_tokens=1024,
                 temperature=0.7,
                 device=None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
