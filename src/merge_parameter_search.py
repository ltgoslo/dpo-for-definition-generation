from smart_open import open
import re
from collections import defaultdict
import json
import logging
import argparse
from unicodedata import category
from collections import Counter
from Levenshtein import distance
import matplotlib.pyplot as plt


def normalize(text):
    text = "".join(ch for ch in text if category(ch)[0] != "P")
    text = text.strip()
    return text


def merging(targets, strategy, distance_threshold, length=4):
    # targets is a dictionary with target words as keys and lists of definitions/senses as values
    # Finding definitions similar to the most frequent one (paraphrases) for each targ_word:
    # length is the minimal length of the definition (in words) to be allowed to replace other definitions
    mappings = {targ_word: {} for targ_word in targets}

    for targ_word in targets:
        definitions = Counter(targets[targ_word]).most_common()
        # definitions = sorted(definitions)  # can be commented out to start from most frequent
        logging.debug(f"{targ_word}: {len(definitions)} unique senses before merging")
        sim_cache = {}
        def2compare = None
        for nr, source in enumerate(definitions):
            cand = source[0]
            if len(cand.split()) >= length and cand not in mappings[targ_word]:
                def2compare = cand
                mapped = 0
                for definition in definitions:
                    definition_text = definition[0]
                    if (
                        definition_text != def2compare
                        and definition_text not in mappings[targ_word]
                    ):
                        if (definition_text, def2compare) in sim_cache:
                            text_distance = sim_cache[(definition_text, def2compare)]
                        else:
                            text_distance = distance(definition_text, def2compare)
                            sim_cache[(definition_text, def2compare)] = text_distance
                            sim_cache[(def2compare, definition_text)] = text_distance
                        if text_distance < distance_threshold:
                            mappings[targ_word][definition_text] = def2compare
                            mapped += 1
                if mapped > 0:
                    logging.debug(f"{mapped} definitions mapped to {def2compare}")
            if strategy == "minimalist":
                if def2compare:
                    break
        if not def2compare:
            logging.debug(
                f"No dominant sense with a long enough definition found for {targ_word}!"
            )
    logging.debug("==================================")
    newtargets = {}
    for word in targets:
        new_defs = [
            mappings[word][d] if d in mappings[word] else d for d in targets[word]
        ]
        newtargets[word] = new_defs
        cur_definitions = Counter(newtargets[word]).most_common()
        logging.debug(f"{word}: {len(cur_definitions)} unique senses")
    return newtargets


def accuracy(preds, gold):
    match = 0
    for word in preds:
        senses = set(preds[word])
        gold_senses = set(gold[word])
        if len(senses) == len(gold_senses):
            match += 1

    return match / len(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred", type=str, help="Text file with generated definitions", required=True
    )
    parser.add_argument(
        "--gold", type=str, help="JSONL file with gold definitions", required=True
    )
    parser.add_argument(
        "--modeltype", type=str, help="Llama or OLMo?", default="olmo"
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    print(args.modeltype)
    with open(args.pred, "r") as ours:
        ours_pred = ours.read()

    if args.modeltype == "olmo":
        to_split = "<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\n"
        target_pat = re.compile("What is the definition of\s+(.+?)\?")
        to_split_prompt = "<|im_end|>\n"
    else:
        to_split = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a lexicographer familiar with providing concise definitions of word meanings.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        target_pat = re.compile(
            f'Please provide a concise definition for the meaning of the word "(.+)" in the following sentence:'
        )
        to_split_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    ours_instances = ours_pred.split(to_split)

    results_gold = defaultdict(list)

    with open(args.gold, "r") as f:
        for line in f:
            line = json.loads(line)
            target = line["target"]
            if args.modeltype == "olmo":
                target = target.lower()
            results_gold[target].append(line["definition"].strip())

    results_our = defaultdict(list)

    for i, instance_our in enumerate(ours_instances[1:]):
        prompt_our, pred_our = instance_our.split(to_split_prompt)
        target_our = re.search(target_pat, prompt_our).group(1)
        results_our[target_our].append(pred_our.strip())

    senses_our = sum(len(target) for target in results_our.values())
    senses_gold = sum(len(target) for target in results_gold.values())

    logging.info(
        f"Average senses per target in gold: {senses_gold / len(results_gold):.2f}"
    )
    logging.info(
        f"Average senses per target in predictions: {senses_our / len(results_our):.2f}"
    )

    our_ambiguous = [el for el in results_our if len(set(results_our[el])) > 1]
    gold_ambiguous = [el for el in results_gold if len(set(results_gold[el])) > 1]

    logging.info(
        f"Ambiguous targets according to gold data: {(len(gold_ambiguous) / len(results_gold)) * 100:.1f}% "
        f"({len(gold_ambiguous)} out of {len(results_gold)})"
    )
    logging.info(
        f"Ambiguous targets according to predictions: {(len(our_ambiguous) / len(results_our)) * 100:.1f}% "
        f"({len(our_ambiguous)} out of {len(results_our)})"
    )
    senses_amb_our = sum(len(results_our[target]) for target in our_ambiguous)
    senses_amb_gold = sum(len(results_gold[target]) for target in gold_ambiguous)

    logging.info(
        f"Average senses per ambiguous target in gold: {senses_amb_gold / len(gold_ambiguous):.2f}"
    )
    logging.info(
        f"Average senses per ambiguous target in predictions: {senses_amb_our / len(our_ambiguous):.2f}"
    )


    logging.info(f"Accuracy: {accuracy(results_our, results_gold):.3f}")
    logging.info("==================================")

    for strat in ["minimalist", "full"]:
        thresholds = []
        accuracies = []
        logging.info(f"Merging strategy: {strat}")
        for threshold in range(100):
            test_merging = merging(results_our, strat, threshold)

            senses_merge = sum(len(target) for target in test_merging.values())
            logging.debug(
                f"Average senses per target after merging: {senses_merge / len(results_our):.3f}"
            )
            merge_ambiguous = [
                el for el in test_merging if len(set(results_our[el])) > 1
            ]
            logging.debug(
                f"Ambiguous targets after merging: {(len(merge_ambiguous) / len(results_our)) * 100:.1f}% "
                f"({len(merge_ambiguous)} out of {len(results_our)})"
            )
            result = accuracy(test_merging, results_gold)
            thresholds.append(threshold)
            accuracies.append(result)
            logging.debug(f"Accuracy with threshold {threshold}: {result:.3f}")

        logging.info(f"Best accuracy: {max(accuracies):.3f}")

        plt.plot(thresholds, accuracies)
        plt.ylabel("Accuracy")
        plt.xlabel("Levenshtein threshold")
        plt.title(f"Merging threshold effect with {strat} strategy")
        plt.grid(True)
        plt.show()
