import re
from collections import defaultdict
import torch
from transformers import EvalPrediction, TrainerCallback


class MetricsComputerOlmo:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.target_pat = re.compile(
            "What is the definition of\s+(.+?)\?",
        )
        self.to_split_prompt = "<|im_end|>\n"
        self.results_our = defaultdict(list)

    def __call__(self, eval_preds: EvalPrediction, compute_result: bool):
        preds = torch.argmax(eval_preds.predictions, dim=2).cpu().tolist()
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(
            eval_preds.label_ids.cpu().tolist(), skip_special_tokens=False
        )
        batch_size = int(len(preds) / 2)
        for pred, label in zip(decoded_preds[:batch_size], decoded_labels[:batch_size]):
            try:
                target_orig = re.search(self.target_pat, label).group(1)
            except AttributeError:
                print(f"Target not found: {label}")
                target_orig = ""
            try:
                pred_our = pred.split(self.to_split_prompt)[-1]
            except ValueError:
                print(f"Prediction not found: {pred}")
                pred_our = pred
            self.results_our[target_orig].append(
                pred_our.replace("<|endoftext|>", "")
                .replace("<|im_start|>", "")
                .strip()
            )


class MetricsComputer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.target_pat = re.compile(
            f'Please provide a concise definition for the meaning of the word "(.+)" in the following sentence:',
        )
        self.to_split_prompt = "<|start_header_id|>assistant<|end_header_id|>"
        self.results_our = defaultdict(list)

    def __call__(self, eval_preds: EvalPrediction, compute_result: bool):
        # print(eval_preds.predictions.size())  # 2 * batch, tokens, vocabulary (first batch chosen, 2nd rejected)
        # print(eval_preds.label_ids.size()) # 2 * batch, tokens
        # print(eval_preds.inputs.size()) # inputs are None

        preds = torch.argmax(eval_preds.predictions, dim=2).cpu().tolist()
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(
            eval_preds.label_ids.cpu().tolist(), skip_special_tokens=False
        )
        batch_size = int(len(preds) / 2)
        for pred, label in zip(decoded_preds[:batch_size], decoded_labels[:batch_size]):
            try:
                target_orig = re.search(self.target_pat, label).group(1)
            except AttributeError:
                print(label)
                target_orig = ""
            try:
                prompt_our, pred_our = pred.split(self.to_split_prompt)
            except ValueError:
                print(pred)
                pred_our = pred
            self.results_our[target_orig].append(
                pred_our.replace("<|start_header_id|>", "")
                .replace("<|begin_of_text|>", "")
                .strip()
            )
            # print(pred)
            # print("+++++++")
            # print(label)
            # print("-------------------------------------------------------------")


class UniqueCallback(TrainerCallback):
    def __init__(self, trainer):
        self._trainer = trainer

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        n_unique = 0
        print("Counting unique senses")
        for i, (target, senses) in enumerate(
            self._trainer.compute_metrics.results_our.items()
        ):
            n_unique += len(set(senses))
            if i < 10:
                print(target, flush=True)
                print(senses, flush=True)
                print("------------------------------")
        unique_senses_per_word = n_unique / len(
            self._trainer.compute_metrics.results_our
        )
        custom_metrics = {
            "eval_unique_senses_per_word": unique_senses_per_word,
        }

        self._trainer.log(custom_metrics)
        print(custom_metrics)
        self._trainer.compute_metrics.results_our = defaultdict(list)
        return control
