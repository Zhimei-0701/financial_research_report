# llm_helper_qwen.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMHelperQwen:
    def __init__(
        self,
        model_name="Qwen/Qwen1.5-7B-Chat",
        device=None,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device is None else {"": device},
            trust_remote_code=True
        )
        self.model.eval()

    def chat(self, prompt: str, system_prompt: str = None):
        # 构造聊天模板格式
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_response(response)

    def _extract_response(self, raw_output: str):
        """
        提取模型返回的 assistant 响应部分。
        Qwen 返回的是完整对话文本，需截取最后一个 assistant 段落。
        """
        if "<|assistant|>" in raw_output:
            return raw_output.split("<|assistant|>")[-1].strip()
        return raw_output.strip()
