import argparse
import json
from pprint import pprint as print
import re
from collections import defaultdict
from datasets import Dataset, DatasetDict
from fuzzywuzzy import fuzz
from smart_open import open
from tqdm import tqdm

olmo = True

SYSTEM_PATTERN = re.compile(
    r"""<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\nYou are a lexicographer familiar with providing concise definitions of word meanings.<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n"""
)
SYSTEM_PATTERN_OLMO = """<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\n"""

TARGET_PATTERN = re.compile(
    r"Please provide a concise definition for the meaning of the word \"(.+)\" in the following sentence",
)
TO_SPLIT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

TO_SPLIT_OLMO = "<|im_end|>\n"

if olmo:
    SYSTEM_PATTERN = SYSTEM_PATTERN_OLMO
    TO_SPLIT = TO_SPLIT_OLMO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default="../train_dpo.jsonl.gz",
    )
    parser.add_argument(
        "--dev",
        default="../dev_dpo.jsonl.gz",
    )
    parser.add_argument("--gold_train", default="../train_dpo.jsonl.gz")
    parser.add_argument("--gold_dev", default="../dev_dpo.jsonl.gz")
    parser.add_argument("--out", default="dpo_dataset_model")
    return parser.parse_args()


def read_predictions(sft_predictions_file, gold_file):
    with open(sft_predictions_file, "r") as f:
        generated = f.read()

    if olmo:
        splitted = generated.split(SYSTEM_PATTERN)
    else:
        splitted = re.split(
            SYSTEM_PATTERN, generated
        )  # not really memory efficient, but we are safe of \n inside examples and such
    assert len(splitted) > 1

    by_senses = defaultdict(dict)
    by_example = defaultdict(dict)
    with open(gold_file, "r") as f:
        for line in f:
            line = json.loads(line)
            targetword = line["target"].lower()
            if olmo:
                example = line["example"].rstrip(" .").strip()
            else:
                example = line["example"]
            if line["definition"] in by_senses[targetword]:
                by_senses[targetword][line["definition"]].append(example)
            else:
                by_senses[targetword][line["definition"]] = [example]
            by_example[targetword][example] = line["definition"]
    return splitted, by_senses, by_example


def format_dpo(sft_predictions_file, gold_file, split):
    splitted, by_senses, _ = read_predictions(sft_predictions_file, gold_file)

    dataset_dict = {"prompt": [], "chosen": [], "rejected": []}
    for text in splitted:
        text = text.strip()
        if text:
            prompt, definition = text.split(TO_SPLIT)
            prompt_with_target, example = prompt.split(": ", maxsplit=1)
            target = re.match(TARGET_PATTERN, prompt_with_target).group(1)

            this_sense = ""
            this_target = by_senses[target]
            for sense, examples in this_target.items():
                for t_example in examples:
                    if t_example == example:
                        this_sense = sense
                        break
                if this_sense:
                    break
            dataset_dict["prompt"].append([{"role": "user", "content": prompt}])
            dataset_dict["chosen"].append(
                [{"role": "assistant", "content": this_sense}]
            )
            dataset_dict["rejected"].append(
                [{"role": "assistant", "content": definition}]
            )
    return Dataset.from_dict(dataset_dict, split=split)


def format_dpo_from_models_answers(sft_predictions_file, gold_file, split):
    splitted, by_senses, by_example = read_predictions(sft_predictions_file, gold_file)
    dataset_dict = {"prompt": [], "chosen": [], "rejected": []}
    by_senses_model = defaultdict(dict)
    parsed = []
    for text in tqdm(splitted):
        text = text.strip()
        if text:
            prompt, definition = text.split(TO_SPLIT)
            if olmo:
                example, target = prompt.split(
                    " What is the definition of ", maxsplit=1
                )
                target = target[:-1]  # Removing "?"
            else:
                prompt_with_target, example = prompt.split(": ", maxsplit=1)
                target = re.match(TARGET_PATTERN, prompt_with_target).group(1)
            if olmo:
                example = example.rstrip(" .").strip()
            try:
                gold_definition = by_example[target][example]
            except KeyError:
                print(by_example[target])
                print(example)
                exit()
            this_sense = by_senses_model[target].get(gold_definition)
            if this_sense is None:
                by_senses_model[target][gold_definition] = definition
            else:
                if fuzz.ratio(gold_definition, this_sense) < fuzz.ratio(
                    gold_definition, definition
                ):
                    by_senses_model[target][gold_definition] = definition
            parsed.append((target, example, prompt, definition))

    for instance in parsed:
        target, example, prompt, definition = instance
        gold_definition = by_example[target][example]
        this_sense = by_senses_model[target].get(gold_definition)
        if this_sense != definition:
            dataset_dict["prompt"].append([{"role": "user", "content": prompt}])
            dataset_dict["chosen"].append(
                [{"role": "assistant", "content": this_sense}]
            )
            dataset_dict["rejected"].append(
                [{"role": "assistant", "content": definition}]
            )
    return Dataset.from_dict(dataset_dict, split=split)


if __name__ == "__main__":
    args = parse_args()
    train = format_dpo_from_models_answers(args.train, args.gold_train, "train")
    dev = format_dpo_from_models_answers(
        args.dev,
        args.gold_dev,
        "dev",
    )
    dpo_dataset = DatasetDict({"train": train, "dev": dev})
    print(dpo_dataset["train"])
    print(dpo_dataset["train"][0])
    print(dpo_dataset["dev"])
    print(dpo_dataset["dev"][0])
    dpo_dataset.save_to_disk(args.out)
