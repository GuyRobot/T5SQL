import json
import re

import datasets
from datasets import DownloadManager, DatasetInfo
from transformers import T5Tokenizer, TFT5Model
from transformers.data.data_collator import DataCollatorForSeq2Seq
import sqlite3

from T5SQL.spider.args import DataArguments, spider_add_serialized_schema
from T5SQL.spider.seq2seq_trainer import SpiderTrainer

_URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


class Spider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="spider",
            version=VERSION,
            description="Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _generate_examples(self, data_filepath, db_path):
        with open(data_filepath) as f:
            spider = json.load(f)
            for idx, sample in enumerate(spider):
                db_id = sample["db_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(db_path + "/" + db_id + "/" + db_id + ".sqlite",
                                                                   db_id)
                schema = self.schema_cache[db_id]
                yield idx, {
                    "query": sample["query"],
                    "question": sample["question"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": [
                        {"table_id": table_id, "column_name": column_name}
                        for table_id, column_name in schema["column_names_original"]
                    ],
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": [{"column_id": column_id, "other_column_id": other_column_id}
                                        for column_id, other_column_id in schema["foreign_keys"]],

                }

    def _info(self) -> DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )

        return datasets.DatasetInfo(
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        downloaded_filepath = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_filepath": downloaded_filepath + "/spider/train_spider.json",
                            "db_path": downloaded_filepath + "/spider/database"},
            ),

            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_filepath": downloaded_filepath + "/spider/dev.json",
                            "db_path": downloaded_filepath + "/spider/database"},
            )

        ]


def convert_fk_index(data):
    fk_holder = []
    for fk in data["foreign_keys"]:
        tn, col, ref_tn, ref_col = fk[0][0], fk[0][1], fk[1][0], fk[1][1]
        ref_cid, cid = None, None
        try:
            tid = data['table_names_original'].index(tn)
            ref_tid = data['table_names_original'].index(ref_tn)

            for i, (tab_id, col_org) in enumerate(data['column_names_original']):
                if tab_id == ref_tid and ref_col == col_org:
                    ref_cid = i
                elif tid == tab_id and col == col_org:
                    cid = i
            if ref_cid and cid:
                fk_holder.append([cid, ref_cid])
        except:
            pass
    return fk_holder


def dump_db_json_schema(db, f):
    """read table and column info"""

    conn = sqlite3.connect(db)
    conn.execute('pragma foreign_keys=ON')
    # noinspection SqlDialectInspection
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    data = {'db_id': f,
            'table_names_original': [],
            'table_names': [],
            'column_names_original': [(-1, '*')],
            'column_names': [(-1, '*')],
            'column_types': ['text'],
            'primary_keys': [],
            'foreign_keys': []}

    fk_holder = []
    for i, item in enumerate(cursor.fetchall()):
        table_name = item[0]
        data['table_names_original'].append(table_name)
        data['table_names'].append(table_name.lower().replace("_", ' '))
        fks = conn.execute("PRAGMA foreign_key_list('{}') ".format(table_name)).fetchall()
        # print("db:{} table:{} fks:{}".format(f,table_name,fks))
        fk_holder.extend([[(table_name, fk[3]), (fk[2], fk[4])] for fk in fks])
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            data['column_names_original'].append((i, col[1]))
            data['column_names'].append((i, col[1].lower().replace("_", " ")))
            # varchar, '' -> text, int, numeric -> integer,
            col_type = col[2].lower()
            if 'char' in col_type or col_type == '' or 'text' in col_type or 'var' in col_type:
                data['column_types'].append('text')
            elif 'int' in col_type or 'numeric' in col_type or 'decimal' in col_type or 'number' in col_type \
                    or 'id' in col_type or 'real' in col_type or 'double' in col_type or 'float' in col_type:
                data['column_types'].append('number')
            elif 'date' in col_type or 'time' in col_type or 'year' in col_type:
                data['column_types'].append('time')
            elif 'boolean' in col_type:
                data['column_types'].append('boolean')
            else:
                data['column_types'].append('others')

            if col[5] == 1:
                data['primary_keys'].append(len(data['column_names']) - 1)

    data["foreign_keys"] = fk_holder
    data['foreign_keys'] = convert_fk_index(data)

    return data


def preprocess(dataset, data_training_args):
    dataset = dataset.map(
        spider_add_serialized_schema,
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    dataset = dataset.map(
        lambda batch: spider_pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.max_target_length,
            data_training_args=data_training_args,
            tokenizer=data_training_args
        ),
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
    )

    return dataset


def load():
    dataset = datasets.load.load_dataset("./spider")
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    data_training_args = DataArguments()
    train_dataset = preprocess(train_dataset, data_training_args)
    validation_dataset = preprocess(validation_dataset, data_training_args)
    return train_dataset, validation_dataset


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))


def spider_get_input(
        question: str,
        serialized_schema: str,
        prefix: str,
) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()


def spider_get_target(
        query: str,
        db_id: str,
        normalize_query: bool,
        target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def spider_pre_process_function(
        batch: dict,
        max_source_length,
        max_target_length,
        data_training_args,
        tokenizer,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""

    inputs = [
        spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
        for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
    ]

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=data_training_args.normalize_query,
            target_with_db_id=data_training_args.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5Model.from_pretrained('t5-small')

input_ids = tokenizer("Studies have been shown that owning a dog is good for you",
                      return_tensors="tf").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="tf").input_ids  # Batch size 1
outputs = model(input_ids, decoder_input_ids=decoder_input_ids)
# create file spider.py inside spider directory with above cell content
dataset = datasets.load.load_dataset(path="./spider")
metric = datasets.load.load_metric(
    path="./spider", config_name="both", test_suite_db_dir=None
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
