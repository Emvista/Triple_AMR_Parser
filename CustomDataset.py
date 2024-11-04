# from torch.utils.data import Dataset
# from pathlib import Path
# from typing import Union
#
# class CustomDataset(Dataset):
#
#     def __init__(self, src_path: Union[str, Path], tgt_path: Union[str, Path], tokenizer, max_length=1024):
#         with open(src_path, "r") as src_f, open(tgt_path, "r") as tgt_f:
#             self.src_txt = src_f.readlines()
#             self.tgt_txt = tgt_f.readlines()
#             self.model_inputs = self.tokenize(tokenizer, max_length)
#
#
#     def __len__(self):
#         return len(self.src_txt)
#
#     def __getitem__(self, index):
#         return self.model_inputs[index]
#
#     def tokenize(self, tokenizer, max_length):
#         # tokenize the source and target text without padding
#
#         model_inputs = tokenizer(self.src_txt, text_target=self.tgt_txt,
#                                  max_length=max_length,
#                                  truncation=True)
#
#         return model_inputs

from datasets import Dataset

class CustomDataset:

    def __init__(self, src_path, tgt_path):
        self.src_txt, self.tgt_txt = self.load_data(src_path, tgt_path)

    def load_data(self, src_path, tgt_path):
        # 1. load data
        with open(src_path, "r") as src_f, open(tgt_path, "r") as tgt_f:
            src_txt = src_f.readlines()
            tgt_txt = tgt_f.readlines()

        # 2. create dataset
        dataset = Dataset.from_dict({"input": src_txt, "label": tgt_txt})
        return dataset

