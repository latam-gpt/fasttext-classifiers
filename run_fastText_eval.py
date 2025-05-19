import argparse
import logging
import os
import random
import re
import shutil
from glob import glob

from datasets import concatenate_datasets, load_dataset, load_from_disk
from fasttext import FastText, load_model
from tqdm import tqdm

# This line prevents the warning from fasttext
# (https://github.com/facebookresearch/fastText/issues/1067)

FastText.eprint = lambda *args, **kwargs: None
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# This vars prevents load the model multiple times for thread
load = False
model = ""


def delete_cache(path, cache_dir):
    for file in os.listdir(path):
        if "cache" in file:
            os.remove(os.path.join(path, file))

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)


def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"([.!?,\'\"/()])", r" \1 ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_labels(sample, model_path):
    global load
    global model

    if not load:
        model = load_model(f"{model_path}")
        load = True

    label = model.predict(
        sample["text_formatted"].replace("\n", " "), threshold=0.98, k=-1
    )

    if len(label[0]) >= 1:
        label = label[0][0]
        if "Not_" in label:
            label = "False"
        else:
            label = "True"
    else:
        label = "False"
    sample["Label"] = label
    return sample


def process_arrow(file_path, model_path, cache_dir):
    shard_dataset = load_dataset("arrow", data_files=[file_path], cache_dir=cache_dir)[
        "train"
    ]

    shard_dataset = shard_dataset.map(
        lambda sample: {
            "text": sample["text"],
            "text_formatted": preprocess_text(sample["text"]),
        },
        num_proc=100,
    )
    shard_dataset = shard_dataset.map(
        lambda sample: get_labels(sample, model_path),
        num_proc=100,
    )

    shard_dataset = shard_dataset.filter(
        lambda sample: sample["Label"] == "True",
        num_proc=100,
    )

    return shard_dataset


def combine_datasets(path):
    files = glob(f"{path}/*.arrow")
    datasets = []
    for file_path in files:
        dataset = load_from_disk(file_path)
        datasets.append(dataset)
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset


def main(args):
    delete_cache(args.dataset_path, args.cache_dir)
    files = glob(args.dataset_path + "/*.arrow")
    random.shuffle(files)
    logging.info(f"Have found {len(files)} arrows in the dataset")
    sum = 0
    for file_path in tqdm(files, desc="Processing files"):
        logging.info(f"Processing {file_path}")
        arrow = process_arrow(file_path, args.models_path)

        sum += len(arrow)

        logging.info(f"{sum} been founded in the dataset")

        arrow.save_to_disk(f"{args.output_path}_arrows/processed_{sum}.arrow")
        delete_cache(args.dataset_path, args.cache_dir)
        if sum > args.size:
            break

    final_dataset = combine_datasets(f"{args.output_path}_arrows/")
    final_dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Data/Dataset_of_the_labels",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="Data/labelled_dataset",
        help="Path to the output directory",
    )

    parser.add_argument(
        "--models_path",
        type=str,
        default="models/",
        help="Path to the models directory",
    )

    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Size of the dataset to process",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Path to the cache directory",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
