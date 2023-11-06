from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import warnings

from peft import PeftConfig, PeftModel


warnings.filterwarnings("ignore")


class InferenceLLM:
    def __init__(self, path_or_model_name, use_flash_attention_2=False):

        self.config = PeftConfig.from_pretrained(path_or_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            return_dict=True,
            load_in_4bit=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name_or_path
        )

        # Load the Lora model
        self.model = PeftModel.from_pretrained(self.model, path_or_model_name)

        # self.model = AutoModelForCausalLM.from_pretrained(path_or_model_name)
        # self.tokenizer =AutoTokenizer.from_pretrained(path_or_model_name)

    def __call__(self, prompt: str) -> Any:

        input_ids = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**input_ids, max_length=120)
        text_generated = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return text_generated


if __name__ == "__main__":

    # prompt="### Instruction: You are Yacine , an assistant virtual to help user .\n ### Question: What's your name ?\n### Answer :"

    prompt = "### Human: Écrivez un récit sous la forme d'une lettre pour Abdou Aziz DIOP .\n### Assistant : "

    generator = InferenceLLM(path_or_model_name="llama2-french")

    print(generator(prompt=prompt))
