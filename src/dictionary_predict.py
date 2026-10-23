import argparse
import json
import os
import re
from peft import AutoPeftModelForCausalLM
from smart_open import open
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

prompts = {
    "en": ". What is the definition of <TRG>?",
    "fr": ". Quelle est la définition de <TRG>?",
    "ru": ". Что такое <TRG>?",
    "no": ". Hva betyr <TRG>?",
    "de": ". Was ist die Definition von <TRG>?",
    "fi": ". Mitä tarkoittaa <TRG>?",
    "it": ". Qual è la definizione di <TRG>?",
    "es": ". ¿Cuál es la definición de <TRG>?",
}


def clean_text(text):
    if isinstance(text, str):
        # Handle patterns like #{option1|option2}#
        text = re.sub(r"#\{[^|}]+\|([^}]+)\}#?", r"\1", text)
        # Handle patterns like {option1|option2}# at the start of the sentence
        text = re.sub(r"^\{[^|}]+\|([^}]+)\}#?", r"\1", text)
    return text


def chat_template(record, modeltype="llama"):
    record = clean_text(record)
    if modeltype == "llama":
        chat = [
            {
                "role": "system",
                "content": "You are a lexicographer familiar with providing concise definitions of word meanings.",
            },
            {
                "role": "user",
                "content": f'Please provide a concise definition for the meaning of the word "{record['target']}" in the following sentence: {record['example']}',
            },
        ]
    else:
        question = record["example"].rstrip().rstrip(".").rstrip() + prompts[
            "en"
        ].replace("<TRG>", record["target"].lower())
        chat = [{"role": "user", "content": question}]
    return chat


class DictionaryDataset(Dataset):
    def __init__(self, pipe, path, modeltype):
        examples = set()
        print(path, flush=True)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    line = json.loads(line)
                    chat = chat_template(line, modeltype=modeltype)
                    instance = pipe.tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=False
                    )
                    examples.add(instance)
        self.examples = list(examples)
        self.examples.sort(key=len, reverse=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="dev_dbnary_en.jsonl")  # will run in 3 hours
    parser.add_argument("--data_path", help="Path to the directory with jsonl files")
    parser.add_argument("--model", default="")  # path to the definition generator model
    parser.add_argument("--modeltype", default="olmo", choices=["olmo", "llama"])
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--batch_size", type=int, default=32)
    # Why should we use a tokenizer different from the model?
    #    parser.add_argument('--tokenizer', default="meta-llama/Meta-Llama-3-8B-Instruct")
    return parser.parse_args()


def llama_dictionary_generation(pipe, dataset, args, outfile_name):
    pipe.tokenizer.padding_side = "left"
    pipe.tokenizer.add_special_tokens = True
    pipe.tokenizer.add_eos_token = True
    pipe.tokenizer.add_bos_token = True
    # end of sequence for stop condition
    eos_tokens = [pipe.tokenizer.eos_token_id]
    eos_tokens.extend(
        [26, 2652, 13, 662, 128009]
    )  # [';', ' ;', '.', ' .', '<|eot_id|>']

    outputs = []
    for i, out in enumerate(
        tqdm(
            pipe(
                dataset,
                batch_size=args.batch_size,
                forced_eos_token_id=eos_tokens,
                eos_token_id=eos_tokens,  # temperature = 0.00001,
                pad_token_id=pipe.tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
                min_new_tokens=6,
                padding="longest",
            ),
            total=len(dataset),
        )
     ):
        # if i < 10:
        #    print(out, flush=True)
        outputs += out
    with open(outfile_name, "w") as outfile:
        for el in outputs:
            outfile.write(el["generated_text"] + "\n")


def generation(pipe, dataset, args, outfile_name):
    tokens = [".", " ."]
    eos_tokens = [pipe.tokenizer.eos_token_id] + [
        pipe.tokenizer.encode(token, add_special_tokens=False)[0] for token in tokens
    ]

    outputs = []

    for i, out in enumerate(
        tqdm(
            pipe(
                dataset,
                forced_eos_token_id=eos_tokens,
                eos_token_id=eos_tokens,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=6,
                do_sample=False,
                batch_size=args.batch_size,
                repetition_penalty=args.repetition_penalty,
                # temperature=0.001,   # Why do we need temperature for greedy decoding?
                pad_token_id=pipe.tokenizer.eos_token_id,
                padding="longest",
            )
        )
    ):
        # if i < 10:
        #    print(out, flush=True)
        outputs += out
    print(f"Generation finished, writing to {outfile_name}")
    with open(outfile_name, "w") as outfile:
        for el in outputs:
            outfile.write(el["generated_text"] + "\n")

def main():
    args = parse_args()
    print(args, flush=True)
    data_path = args.data_path
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if args.modeltype == "olmo":
        model = AutoPeftModelForCausalLM.from_pretrained(args.model, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model.eval()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        num_workers=4,
    )
    dataset = DictionaryDataset(
        pipe, os.path.join(data_path, args.jsonl), args.modeltype
    )

    model_name_for_outfile = args.model.strip("/").replace("/", "-")
    print(model, flush=True)
    outfile_name = f"{args.jsonl}.{model_name_for_outfile}.txt.gz"
    outfile_name = os.path.join(data_path, outfile_name)
    if args.model != "FrancescoPeriti/Llama3Dictionary-merge":
        generation(pipe, dataset, args, outfile_name)
    else:
        llama_dictionary_generation(pipe, dataset, args, outfile_name)


if __name__ == "__main__":
    main()
