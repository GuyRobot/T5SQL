from typing import Optional, Union, Dict, Any
import datasets
import T5SQL.spider.evaluation as spider_evaluation

_URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


class Spider(datasets.Metric):
    def __init__(
            self,
            config_name: Optional[str] = None,
            keep_in_memory: bool = False,
            cache_dir: Optional[str] = None,
            num_process: int = 1,
            process_id: int = 0,
            seed: Optional[int] = None,
            experiment_id: Optional[str] = None,
            max_concurrent_cache_files: int = 10000,
            timeout: Union[int, float] = 100,
            **kwargs
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )
        self.test_suite_db_dir: Optional[str] = kwargs.pop("test_suite_db_dir", None)

    def _info(self):
        if self.config_name not in [
            "exact_match",
            "test_suite",
            "both",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in " '["exact_match", "test_suite", "both"]'
            )
        return datasets.MetricInfo(
            "Spider metric",
            "Spider citation",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": {
                        "query": datasets.Value("string"),
                        "question": datasets.Value("string"),
                        "context": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                        "db_path": datasets.Value("string"),
                        "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                        "db_column_names": datasets.features.Sequence(
                            {
                                "table_id": datasets.Value("int32"),
                                "column_name": datasets.Value("string"),
                            }
                        ),
                        "db_foreign_keys": datasets.features.Sequence(
                            {
                                "column_id": datasets.Value("int32"),
                                "other_column_id": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            reference_urls=[_URL],
        )

    def _compute(self, predictions, references):
        if self.config_name == "exact_match" or self.config_name == "both":
            exact_match = compute_exact_match_metric(predictions, references)
        else:
            exact_match = dict()

        test_suite = dict()

        return {**exact_match, **test_suite}


def compute_exact_match_metric(predictions, references) -> Dict[str, Any]:
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = spider_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    evaluator = spider_evaluation.Evaluator(references[0]["db_path"], foreign_key_maps, "match")
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        _ = evaluator.evaluate_one(reference["db_id"], reference["query"], prediction)
    evaluator.finalize()
    return {
        "exact_match": evaluator.scores["all"]["exact"],
    }
