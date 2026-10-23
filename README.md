# definitions-dpo

1. data_2025.ipynb loads Francesco's dataset from HF and creates a *.jsonl. don't forget not to use labels replacement

2. src/llama_dictionary_predict.py takes jsonl and generates txt with model predictions

3. src/format_dpo.py takes that txt and creates dpo dataset in hf format. this one may be used by dpo.py for training

## Reproducing results for SemEval'20:

```shell
git clone git@github.com:ltgoslo/Definition-generation-for-LSCD.git
cp -r generated_definitions/english/periti_202*/ Definition-generation-for-LSCD/generated_definitions/english/
cp -r generated_definitions/english/dpo/ Definition-generation-for-LSCD/generated_definitions/english/
```

then follow the instructions in that repository. don't use merging