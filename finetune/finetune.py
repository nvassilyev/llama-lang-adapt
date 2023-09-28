import os
import logging

# import fire
import torch
import transformers
import math

from peft import (
    LoraConfig,
    PeftModel, 
    PeftConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from datasets import load_dataset
from tqdm import tqdm
from itertools import chain

def train():  
    # training hyperparams
    batch_size = 32
    micro_batch_size = 16
    block_size = 512

    # llm hyperparams
    resume_from_checkpoint = None  # either training checkpoint or final adapter
    wandb_run_name = "llama-2-yoruba"
    
    base_model = "meta-llama/Llama-2-7b-hf"
    train_file = "../african-plm/data/mlm/yo/train.txt"
    validation_file = "../african-plm/data/mlm/yo/eval.txt"
    output_dir = "models/llama-2-yoruba"
    cache_dir = ".cache"

    os.environ["WANDB_API_KEY"] = '8c1bd48d33538f11315cd7578c4954f1713febce'
    os.environ['WANDB_ENTITY']='nvassilyev'
    os.environ["WANDB_PROJECT"] = "lang-adapt"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    # world_size = int(os.environ.get("WORLD_SIZE", 1))

    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size


    data_files = {}
    data_files["train"] = train_file
    data_files["validation"] = validation_file

    raw_datasets = load_dataset(
        "text",
        data_files=data_files,
        cache_dir=cache_dir
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    columns = raw_datasets["train"].features

    tokenized_datasets = raw_datasets.map(
        lambda data: tokenizer(data["text"]),
        batched=True,
        remove_columns=columns
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )


    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            use_auth_token=True,
            cache_dir=cache_dir,
        )
    
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r = 8,
        lora_alpha = 16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=31
    )

    model = get_peft_model(model, config)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        output_dir=output_dir,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=wandb_run_name,
        overwrite_output_dir=True,
    )
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
        # data_collator=transformers.default_data_collator,
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    metrics = trainer.evaluate()

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print("Perplexity:", perplexity)


if __name__ == "__main__":
    train()