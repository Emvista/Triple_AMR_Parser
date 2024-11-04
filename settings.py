
import pathlib
import logging

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]
VALID_SUFFIXES = ["amr", "tri", "wo_invrole.tri", "wo_var.tri", "wo_var.wo_invrole.tri" ]

def get_data_path(name, split, suffix=None):
    # name should be in the format of "src_lang-tgt_task" (e.g. "fr-amr")
    suffix = "tri" if suffix is None else suffix
    src_lang, tgt_task = name.split("-")[0], name.split("-")[1]
    src_path = DATA_DIR / f"{tgt_task}" / f"{src_lang}" / f"{split}" / f"{src_lang}-{tgt_task}.{src_lang}"
    tgt_path = DATA_DIR / f"{tgt_task}" / f"{src_lang}" / f"{split}" / f"{src_lang}-{tgt_task}.{suffix}"
    return src_path, tgt_path

FAIRESEQ_LANGUAGE_MAP = {lang[:2]:lang for lang in FAIRSEQ_LANGUAGE_CODES}

logger = logging.getLogger()
logfmt = '%(asctime)s - %(levelname)s - \t%(message)s'
logging.basicConfig(format=logfmt, datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

LANGUAGE_MAPPING = {"amr": "amr",
                    "vannoord": "vnd",
                    "penman": "penman" }