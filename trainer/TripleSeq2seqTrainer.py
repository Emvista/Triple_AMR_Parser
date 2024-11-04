from transformers import Seq2SeqTrainer
from tqdm import tqdm
from typing import Optional, List
from pathlib import Path
from torch.utils.data import Dataset

from postprocess.postprocess_tri import prediction_to_penman, pass_sanity_check
from postprocess.singline_to_mutiline_amr import reformat_to_multiline
from AMR import postprocess_AMRs as pm_postprocess
from trainer.trainer_utils import get_restored_and_formatted, set_smatch_args, fix_easy_errors, add_unmatched_parenthesis, undo_ne_recat
from AMR_utils import to_single_line_amr, get_default_amr
from utils import new_cd, fill_empty_line
from settings import PROJECT_DIR, get_data_path
from AMR.smatch import smatch
from AMR.wikify_file import wikify_file
from postprocess.postprocess_penman import NamedEntPostprocessor
class TripleSeq2seqTrainer(Seq2SeqTrainer):
    def __init__(self, train_subset=None,
                 without_variables=False,
                 no_wikify=False,
                 forced_bos_token_id=None,
                 generate_max_length=None,
                 generate_num_beams=None,
                 target_format="amr",
                 eval_gold_file=None,
                 test_gold_file=None,
                 test_v3_gold_file=None,
                 silvertest_gold_file=None,
                 sub_train_gold_file=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_subset = train_subset
        self.prediction_save_path = Path(self.args.output_dir) / "predictions"
        self.restore_variable = without_variables # if True, restore variables in the prediction
        self.no_wikify = no_wikify
        self.target_format = target_format
        eval_gold_file = eval_gold_file
        test_gold_file = test_gold_file
        test_v3_gold_file = test_v3_gold_file
        silvertest_gold_file = silvertest_gold_file
        sub_train_gold_file = sub_train_gold_file
        # make a dictionary of reference multiline gold amr graphs file path
        self.reference_paths = {"eval": eval_gold_file,
                                "test": test_gold_file,
                                "silvertest": silvertest_gold_file,
                                "train_subset": sub_train_gold_file,
                                "test_v3": test_v3_gold_file,}

        # since modifying generation_config is deprecated in transformers v4.x
        self.forced_bos_token_id = forced_bos_token_id
        self.generate_max_length = generate_max_length
        self.generate_num_beams= generate_num_beams


    def write_predictions(self, metric_key_prefix:str, predictions: List[str], is_raw_predictions=True) -> Path:
        step_num = f"step_{self.state.global_step}"
        if "test" in metric_key_prefix:
            step_num  = metric_key_prefix
        file_name = "predictions.txt" if is_raw_predictions else "predictions.pm"
        line_break = "\n" if is_raw_predictions else "\n\n"
        file_path = self.prediction_save_path / step_num / metric_key_prefix / file_name
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w") as f:
            f.write(line_break.join(predictions))
        return file_path

    def write_smatch(self, metric_key_prefix:str, smatch: float):
        step_num = f"step_{self.state.global_step}"
        if "test" in metric_key_prefix:
            step_num = metric_key_prefix
        file_name = f"smatch.txt"
        file_path = self.prediction_save_path / step_num / metric_key_prefix / file_name
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, "w") as f:
            f.write(str(smatch))

    def postprocess_predictions_triplet(self, metric_key_prefix, predictions: List[str]):
        # write raw predictions to file
        self.write_predictions(metric_key_prefix, predictions, is_raw_predictions=True)

        single_line_predictions = []
        multi_lines_predictions = []
        print(f"Postprocessing predictions for {metric_key_prefix} dataset: wikify={not self.no_wikify}, restore_variable={self.restore_variable}")

        for prediction in predictions:
            pm = prediction_to_penman(prediction, restore_wiky=not self.no_wikify, do_variable_restore=self.restore_variable)
            single_line_pm = to_single_line_amr(pm)

            single_line_predictions.append(single_line_pm)
            multi_lines_predictions.append(reformat_to_multiline(single_line_pm))

        return single_line_predictions, multi_lines_predictions

    def postprocess_prediction_penman(self, metric_key_prefix:str, predictions):
        single_line_predictions = []
        # write raw predictions to file
        raw_prediction_file = self.write_predictions(metric_key_prefix, predictions, is_raw_predictions=True,)

        # enlgish text necessary for wiki restore in vannoord postprocessing
        sent_file, _ = get_data_path(name="en-amr", split=metric_key_prefix)

        # step 1.
        # psotprocess predictions to fix format errors (unmatched parenthesis)
        # and undo named entity recategorization
        postprocesed_predictions = []
        for prediction in predictions:
            post_processed = add_unmatched_parenthesis(prediction)
            post_processed = fix_easy_errors(post_processed)
            post_processed = undo_ne_recat(post_processed)

            if not pass_sanity_check(post_processed):
                post_processed = get_default_amr("pm_1line")

            postprocesed_predictions.append(post_processed)

        final_file = raw_prediction_file.parent / (raw_prediction_file.name + ".postprocessed")
        with open(final_file, "w") as f:
            f.write("\n".join(postprocesed_predictions)) # write postprocessed predictions to file

        # step2. wikify the cleaned output file
        if not self.no_wikify:
            wikify_file(final_file.as_posix(), sent_file.as_posix())
            final_file = final_file.parent / f"{final_file.name}.wiki"

        # step3. one final sanity check
        for line in open(final_file):
            line = line.strip()
            if not pass_sanity_check(line):
                line = get_default_amr("pm_1line")
            single_line_predictions.append(line.strip())

        multi_line_predictions = [reformat_to_multiline(line) for line in single_line_predictions]

        return single_line_predictions, multi_line_predictions

    def postprocess_predictions_vnd(self, metric_key_prefix:str, predictions):
        # write raw predictions to file
        raw_prediction_path = self.write_predictions(metric_key_prefix, predictions, is_raw_predictions=True, )

        # 1. undo named entity recategorization
        recat_undo_path = raw_prediction_path.parent / (raw_prediction_path.name + ".recat_undo")
        recats = []
        for pred in predictions:
            processor = NamedEntPostprocessor(pred, with_variables=False)
            recats.append(processor.undo_ne_recategorize())
        with open(recat_undo_path, "w") as f:
            f.write("\n".join(recats))

        # postprocess predictions linearized with van noord linezarzation script
        print(f"Postprocessing predictions for {metric_key_prefix} dataset: wikify={not self.no_wikify}, van noord format")
        coreference, force, no_empty_line_file, no_wiki, sent_file = self.get_input_args(metric_key_prefix, recat_undo_path)

        # 2. execute van noord postprocessing
        input_args = no_empty_line_file.as_posix(), sent_file.as_posix(), no_wiki, coreference, force
        try:
            with new_cd(PROJECT_DIR / "AMR"):
                pm_postprocess.process_file(input_list=input_args)
        except Exception as e:
            print(f"Van Noord postprocessing failed with error: {e}")
            return predictions, predictions

        # 3. read the output file and reformat the single line to multi line predictions
        restored_file = no_empty_line_file.as_posix() + '.restore.final'
        if Path(restored_file).exists():
            single_line_predictions, multi_lines_predictions = get_restored_and_formatted(restored_file)
            return single_line_predictions, multi_lines_predictions


    def get_input_args(self, metric_key_prefix, raw_prediction_path):
        # 0. set arguments to execute van noord postprocessing
        sent_file, _ = get_data_path(name="en-amr", split=metric_key_prefix) # van noord wikification requires Enlgish text source file path
        no_wiki = self.no_wikify
        coreference = 'dupl'  # co-referring nodes are duplicated in the output
        force = True  # force overwrite if the file already exists

        # 1. fill empty with '#' in generation file to avoid error during postprocessing
        no_empty_line_file = fill_empty_line(raw_prediction_path)

        return coreference, force, no_empty_line_file, no_wiki, sent_file

    def compute_smatch(self, metric_key_prefix:str, predictions: List[str]):
        """
        Compute the smatch_modified score for the predictions
        Step 1. Postprocess the predictions to the target format (penman or amr)
        Step 2. Compute the smatch_modified score
        """

        # Step 1 - preprocessing
        if self.target_format == "amr":
            single_line_predictions, multi_line_predictions = self.postprocess_predictions_triplet(metric_key_prefix, predictions)
        elif self.target_format == "vannoord":
            single_line_predictions, multi_line_predictions = self.postprocess_predictions_vnd(metric_key_prefix, predictions)
        elif self.target_format == "penman":
            single_line_predictions, multi_line_predictions = self.postprocess_prediction_penman(metric_key_prefix, predictions)
        else:
            raise ValueError(f"Invalid target format: {self.target_format}")

        pred_path = self.write_predictions(metric_key_prefix, multi_line_predictions, is_raw_predictions=False,)
        eval_path = self.reference_paths.get(metric_key_prefix)

        # Step 2
        arg_parser = set_smatch_args(pred_path, eval_path)
        with new_cd(PROJECT_DIR / "AMR" / "smatch"):
            try:
                f1_score = smatch.main(arg_parser)
            except (AttributeError, IndexError, ValueError) as e:
                print("Smatch script failed with error: ", e)
                f1_score = 0.0 # if the smatch script fails, return 0.0
        return f1_score

    def eval_loop(self, dataloader, metric_key_prefix):
        # iterate through the dataloader for evaluation
        # write predictions/smatch_modified in .txt and log the smatch_modified score
        total_predictions = []
        total_labels = []
        total_loss = 0.0

        for i, inputs in enumerate(tqdm(dataloader, desc=f"Evaluating on {metric_key_prefix} dataset")):
            loss, generated_tokens, labels = self.prediction_step(self.model,
                                                                  inputs,
                                                                  prediction_loss_only=False,
                                                                  forced_bos_token_id=self.forced_bos_token_id,
                                                                  max_length=self.generate_max_length,
                                                                  num_beams=self.generate_num_beams,)
            total_loss += loss.item()

            # decode the generated tokens and labels
            decoded_predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            labels[labels == -100] = self.tokenizer.pad_token_id  # replace -100 with pad to    ken id for decoding
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            total_predictions.extend(decoded_predictions)
            total_labels.extend(decoded_labels)

        loss = total_loss / len(dataloader)

        # saved_file_path = self.write_predictions(metric_key_prefix, total_predictions) # write raw prediction
        # compute smatch_modified score and log the results
        smatch = round(self.compute_smatch(metric_key_prefix, total_predictions), 3)
        eval_metrics = {f"{metric_key_prefix}_loss": loss, f"{metric_key_prefix}_smatch": smatch}
        self.write_smatch(metric_key_prefix, smatch)
        self.log(eval_metrics)

        return eval_metrics

    def evaluate(self, eval_dataset: Optional[Dataset]=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):

        train_subset_dataloader = self.get_eval_dataloader(self.train_subset)
        dataloader = self.get_eval_dataloader()

        if "test" in metric_key_prefix :
            train_subset_dataloader = None
            dataloader = self.get_test_dataloader(eval_dataset)

        if train_subset_dataloader is not None:
            _ = self.eval_loop(train_subset_dataloader, "train_subset") # write predictions/logging
        eval_metrics = self.eval_loop(dataloader, metric_key_prefix)
        # if validation, check for early stopping
        if metric_key_prefix =="eval":
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_metrics)

        return eval_metrics

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test", **kwargs):
        return self.evaluate(test_dataset, ignore_keys, metric_key_prefix, **kwargs)




