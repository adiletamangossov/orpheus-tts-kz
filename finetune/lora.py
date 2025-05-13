from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import wandb

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

lora_rank = 32
lora_alpha = 64
lora_dropout = 0.0

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj",  "o_proj", "gate_proj", "down_proj", "up_proj"],
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"], # Optional to train the embeddings and lm head
    task_type="CAUSAL_LM",
    use_rslora=True,
)

model = get_peft_model(model, lora_config)

MAX_LEN = 2048

def split_long_rows_batch(batch):
    out_input_ids, out_labels, out_masks = [], [], []
    for ids in batch["input_ids"]:
        for i in range(0, len(ids), MAX_LEN):
            chunk = ids[i : i + MAX_LEN]
            if chunk:
                out_input_ids.append(chunk)
                out_labels.append(chunk)
                out_masks.append([1] * len(chunk))
    return {
        "input_ids":      out_input_ids,
        "labels":         out_labels,
        "attention_mask": out_masks,
    }

raw_ds = load_dataset(
    dsn,
    split="train",
    streaming=False,
)

ds = raw_ds.map(
    split_long_rows_batch,
    batched=True,
    batch_size=1000,
    num_proc=4,
    remove_columns=raw_ds.column_names,
)

wandb.init(project=project_name, name = run_name)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True, 
    learning_rate=learning_rate,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()

merged_model = model.merge_and_unload()

merged_model.save_pretrained(f"./{base_repo_id}/merged")
tokenizer.save_pretrained(f"./{base_repo_id}/merged")
