import datasets
import transformers
from transformers.trainer_utils import PredictionOutput, speed_metrics
from datasets.arrow_dataset import Dataset
from datasets.metric import Metric
from typing import Optional, List, NamedTuple, Dict
import collections
import time
import numpy as np
import json
from transformers import T5Tokenizer, TFT5Model
from transformers.data.data_collator import DataCollatorForSeq2Seq


class EvalPrediction(NamedTuple):
    predictions: List[str]
    label_ids: np.ndarray
    metas: List[dict]


class Seq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
            self,
            metric: Metric,
            *args,
            eval_examples: Optional[Dataset] = None,
            ignore_pad_token_for_loss: bool = True,
            target_with_db_id: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.target_with_db_id = target_with_db_id

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        raise NotImplementedError()

    def _post_process_function(
            self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        raise NotImplementedError()

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            eval_examples: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                eval_dataset,
                output.predictions,
                "eval_{}".format(self.state.epoch),
            )
            output.metrics.update(self.compute_metrics(eval_preds))

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Dataset,
            test_examples: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # We might have removed columns from the dataset so we put them back.
            if isinstance(test_dataset, Dataset):
                test_dataset.set_format(
                    type=test_dataset.format["type"],
                    columns=list(test_dataset.features.keys()),
                )

            eval_preds = self._post_process_function(
                test_examples, test_dataset, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output


class SpiderTrainer(Seq2SeqTrainer):
    def _post_process_function(
            self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = [
            {
                "query": x["query"],
                "question": x["question"],
                "context": context,
                "label": label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        if self.target_with_db_id:
            # Remove database id from all predictions
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        return self.metric.compute(predictions=predictions, references=references)


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5Model.from_pretrained('t5-small')

input_ids = tokenizer("Studies have been shown that owning a dog is good for you",
                      return_tensors="tf").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="tf").input_ids  # Batch size 1
outputs = model(input_ids, decoder_input_ids=decoder_input_ids)
# create file spider.py inside spider directory with above cell content
dataset = datasets.load.load_dataset(path="")
metric = datasets.load.load_metric(
    path="", config_name="both", test_suite_db_dir=None
)
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

training_args = {
    "run_name": "t5-spider",
    "model_name_or_path": "t5-3b",
    "dataset": "spider",
    "source_prefix": "",
    "schema_serialization_type": "peteshaw",
    "schema_serialization_randomized": False,
    "schema_serialization_with_db_id": True,
    "schema_serialization_with_db_content": True,
    "normalize_query": True,
    "target_with_db_id": True,
    "output_dir": "/train",
    "cache_dir": "/transformers_cache",
    "do_train": True,
    "do_eval": True,
    "fp16": False,
    "num_train_epochs": 3072,
    "per_device_train_batch_size": 5,
    "per_device_eval_batch_size": 5,
    "gradient_accumulation_steps": 410,
    "label_smoothing_factor": 0.0,
    "learning_rate": 1e-4,
    "adafactor": True,
    "adam_eps": 1e-6,
    "lr_scheduler_type": "constant",
    "warmup_ratio": 0.0,
    "warmup_steps": 0,
    "seed": 1,
    "report_to": ["wandb"],
    "logging_strategy": "steps",
    "logging_first_step": True,
    "logging_steps": 4,
    "load_best_model_at_end": True,
    "metric_for_best_model": "exact_match",
    "greater_is_better": True,
    "save_total_limit": 128,
    "save_steps": 64,
    "evaluation_strategy": "steps",
    "eval_steps": 64,
    "predict_with_generate": True,
    "num_beams": 1,
    "num_beam_groups": 1,
    "use_picard": False
}

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "metric": metric,
    "train_dataset": train_dataset,
    "eval_dataset": validation_dataset,
    "eval_examples": None,
    "tokenizer": tokenizer,
    "data_collator": DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=(-100),
        pad_to_multiple_of=None,
    ),
    "ignore_pad_token_for_loss": True,
    "target_with_db_id": True,
}

trainer = SpiderTrainer(**trainer_kwargs)
