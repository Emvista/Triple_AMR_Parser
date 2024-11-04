from datasets import Dataset


def load_seq2seq_data(src_path, tgt_path, max_samples=None):
    # 1. load data
    with open(src_path, "r") as src_f, open(tgt_path, "r") as tgt_f:
        src_txt = [line.rstrip() for line in src_f.readlines()]
        tgt_txt = [line.rstrip() for line in tgt_f.readlines()]

    max_samples = max_samples if max_samples is not None else len(src_txt)
    # 2. create dataset
    dataset = Dataset.from_dict({"input": src_txt[:max_samples], "label": tgt_txt[:max_samples]})
    return dataset

