from settings import FAIRESEQ_LANGUAGE_MAP


class Seq2seqTokenizer:
    def __init__(self, tokenizer, src_lang, tgt_lang, max_length):
        self.tokenizer = tokenizer
        self.set_tokenizer_lang(src_lang, tgt_lang)
        self.max_length = max_length

    def set_tokenizer_lang(self, src_lang, tgt_lang):

        self.tokenizer.src_lang = FAIRESEQ_LANGUAGE_MAP[src_lang]
        if tgt_lang == "amr":
            self.tokenizer.tgt_lang = "amr"
        else:
            self.tokenizer.tgt_lang = FAIRESEQ_LANGUAGE_MAP[tgt_lang]

    def tokenize(self, example):
        # tokenize the source and target text without padding
        model_inputs = self.tokenizer(example["input"],
                                      text_target=example["label"],
                                      max_length=self.max_length,
                                      truncation=True)

        return model_inputs