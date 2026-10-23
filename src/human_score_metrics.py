import argparse
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="../data/t2_clustered-apd-xld.tsv",
    )
    args = parser.parse_args()

    conv = {"no change": 0, "loss and gain": 1, "gain": 1, "loss": 1}

    df = pd.read_csv(args.input, sep="\t")
    df = df.loc[df["gold"] != "x"]
    # print(df)
    # print(df.change_type.unique())

    gold = [int(el) for el in df.gold.values]
    predicted = [conv[el] for el in df.change_type]

    assert len(gold) == len(predicted)

    print(classification_report(gold, predicted))
