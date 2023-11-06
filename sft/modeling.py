from datasets import load_dataset , load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
from utils import *
from transformers import TrainingArguments
from trl import SFTTrainer , DataCollatorForCompletionOnlyLM



model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

dataset = load_dataset("Nekochu/novel17_train_alpaca_format" , split="train")

def formating_input(example):
    
    text=f"### Human : {example['instruction']}### Assistant : {example['output']}"
    
    return {
        "text": text
    }

dataset_column_names = dataset.column_names 

dataset = dataset.map(formating_input, batched=False, remove_columns=dataset_column_names )


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model , peft_config=peft_config)


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

instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

 
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments
)


with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    trainer.train()
    
trainer.save_model("llama2-french")