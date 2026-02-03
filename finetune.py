# finetune.py
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# Model configuration
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
new_model_name = "llama-3.1-8b-eldewrito"

# QLoRA configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA configuration
peft_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_dataset('json', data_files='eldewrito_commands.jsonl', split='train')

# Format function for the dataset
def format_instruction(sample):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in ElDewrito modding commands. You generate accurate command scripts for modding Halo Online.<|eot_id|><|start_header_id|>user<|end_header_id|>

{sample['instruction']}
{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sample['output']}<|eot_id|>"""

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,  # Use bfloat16 on 3090
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    warmup_steps=50,
    lr_scheduler_type="cosine",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=format_instruction,
)

# Train
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
