import argparse

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lemma_rule import apply_lemma_rule
from preprocessor import Preprocessor
from utils import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    return parser.parse_args()


class SentencesDataset(Dataset):
    def __init__(self, path="../all/sentences.parquet"):
        self.data = pd.read_parquet(path)
        self.sentences = self.data.sentence.to_list()
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        return self.sentences[idx]
    

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, predict_model = load_model(model_path="HPLT/hplt_gpt_bert_base_3_0_eng_Latn", device=device)
    predict_model.eval()
    preprocessor = Preprocessor(tokenizer=tokenizer)
    nouns, noun_phrases = set(), set()
    lemmas, uposes, deps, heads = [], [], [], []
    dataset = SentencesDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=7)
    for batch in tqdm(dataloader):
        batch = preprocessor.preprocess(batch)
        with torch.no_grad():
            lemma_p, upos_p, _, _, _, _, dep_p, head_p = predict_model(
                                    batch["subwords"].to(device),
                                    batch["alignment"].to(device),
                                    batch["subword_lengths"].to(device),
                                    batch["word_lengths"].to(device),
                                )
        last_upos, last_lemma = "", ""
        for i in range(len(batch["words"])):
            sentence_lemmas, sentence_upos, sentence_deps, sentence_heads = [], [], [], []
            for j, form in enumerate(batch["words"][i]):
                lemma_rule = {
                    rule_type: model["dataset"].lemma_vocab[rule_type].get(p[i, j, :].argmax().item(),
                            model["dataset"].lemma_vocab[rule_type][-1])
                    for rule_type, p in lemma_p.items()
                }
                lemma = apply_lemma_rule(form, lemma_rule)
                sentence_lemmas.append(lemma)
                
                upos = model['dataset'].upos_vocab[upos_p[i, j, :].argmax().item()]
                
                sentence_upos.append(upos)
                sentence_deps.append(model['dataset'].arc_dep_vocab[dep_p[i, j, :].argmax().item()])
                sentence_heads.append(str(head_p[i, j].item()))
                if upos == "NOUN":
                    nouns.add(lemma)
                    if last_upos == "ADJ":
                        noun_phrases.add(" ".join((last_lemma, lemma,)))
                last_upos = upos
                last_lemma = lemma
            lemmas.append(" ".join(sentence_lemmas))
            uposes.append(" ".join(sentence_upos))
            deps.append(" ".join(sentence_deps))
            heads.append(" ".join(sentence_heads))
        
    with open('../nouns.txt', 'w') as nouns_out:
        for noun in sorted(list(nouns)):
            nouns_out.write(noun + '\n')
    with open('../noun_phrases.txt', 'w') as noun_phrases_out:
        for noun_phrase in sorted(list(noun_phrases)):
            noun_phrases_out.write(noun_phrase + '\n')
    dataset.data["lemmas"] = lemmas
    dataset.data["uposes"] = uposes
    dataset.data["deps"] = deps
    dataset.data["heads"] = heads
    dataset.data.to_parquet("../all/sentences_parsed.parquet")

    # my preprocessing splits r&d I hope that's fine
    # EIA lowercased


if __name__ == "__main__":
    main()
