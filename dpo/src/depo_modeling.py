import torch
from datasets import load_dataset
from trl import DPOTrainer
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments,
)
from utils import *
from peft import get_peft_model, LoraConfig


model_name = "llama2-french"

dataset = load_dataset("qanastek/frenchmedmcqa", split="train")


def formating_input(example):

    return {
        "prompt": "### Input: " + example["question"] + "\n ### Output: ",
        "chosen": example["answer_a"],
        "rejected": example["answer_c"],
    }


column_name = dataset.column_names

dataset = dataset.map(formating_input, remove_columns=column_name, batched=False)

peft_config = LoraConfig(
    lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM"
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, trust_remote_code=True
)
model_ref = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = get_peft_model(model , peft_config=peft_config)

model.print_trainable_parameters()


training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_arguments,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=1024,
    max_length=2048,
)


dpo_trainer.train()

