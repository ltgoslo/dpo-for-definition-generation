import argparse
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM

from constants import CACHE_DIR, USER_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--adapter_model", default="FrancescoPeriti/LlamaDictionary-en_ML38BI")
parser.add_argument("--output_dir", default=os.path.join(USER_DIR, "periti_2025_data/LlamaDictionary-en_ML38BI-merged/"))

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.base_model, cache_dir=CACHE_DIR)
peft_model = PeftModel.from_pretrained(model, args.adapter_model, cache_dir=CACHE_DIR)
merged_model = peft_model.merge_and_unload()
print(merged_model)
merged_model.save_pretrained(args.output_dir)
