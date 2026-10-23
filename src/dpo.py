import argparse
import os
import torch
from datasets import load_from_disk
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from constants import USER_DIR
from metrics_computer import MetricsComputer, MetricsComputerOlmo, UniqueCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=os.path.join(USER_DIR, "periti_2025_data/dpo_dataset/")
    )
    parser.add_argument(
        "--model",
        default=os.path.join(
            USER_DIR, "periti_2025_data/LlamaDictionary-en_ML38BI-merged/"
        ),
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(USER_DIR, "periti_2025_data/dpo_model/")
    )
    parser.add_argument("--modeltype", default="olmo")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--lr", default=1e-6, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    print(args, flush=True)
    peft_config = None
    if args.lora:
        peft_config = LoraConfig(
            r=64,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.2,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            # all Linear modules
            bias="none",
            task_type="CAUSAL_LM",
        )

    output_dir = "-".join(
        [
            args.output_dir.rstrip("/"),
#            "lora",
#            str(args.lora),
#            "lr",
#            str(args.lr),
#            "lora",
#            str(args.lora_alpha),
            "scheduler",
            args.lr_scheduler_type,
            "beta",
            str(args.beta),
            "warmup",
            str(args.warmup_steps),
        ]
    )
    training_args = DPOConfig(
        max_length=None,  # no truncation
        logging_steps=100,
        output_dir=output_dir,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=48,
        dataset_num_proc=4,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=True,
        eval_on_start=True,
        eval_steps=500,
        eval_strategy="steps",
        num_train_epochs=1,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        run_name=f"DefGen-DPO-{args.modeltype}-beta-{args.beta}-warmup-{args.warmup_steps}",
        beta=args.beta,
        batch_eval_metrics=True,
    )
    dataset = load_from_disk(args.dataset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    print("Train instance example:", flush=True)
    print(train_dataset[0], flush=True)
    print("Eval instance example:", flush=True)
    print(eval_dataset[0], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if args.modeltype == "olmo":
        metrics_computer = MetricsComputerOlmo(tokenizer=tokenizer)
        model = AutoPeftModelForCausalLM.from_pretrained(
        args.model, device_map="auto", is_trainable=True, dtype=torch.bfloat16
    )
    else:
        metrics_computer = MetricsComputer(tokenizer=tokenizer)
        model = AutoModelForCausalLM(args.model, device_map="auto", dtype=torch.bfloat16)

    trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config,  # It is not really necessary, TRL picks it up from the base model
        processing_class=tokenizer,
        compute_metrics=metrics_computer,
    )
    unique_callback = UniqueCallback(trainer)
    trainer.add_callback(unique_callback)
    trainer.train()
    trainer.save_model()
