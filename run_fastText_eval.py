import argparse
import logging
import os
import shutil
from glob import glob

from datasets import concatenate_datasets, config, load_dataset, load_from_disk
from fasttext import FastText, load_model
from tqdm import tqdm

# This line prevents the warning from fasttext
# (https://github.com/facebookresearch/fastText/issues/1067)
FastText.eprint = lambda *args, **kwargs: None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load = False
models = []


def get_labels(sample, model_path):
    global load
    global models

    if not load:
        models = [
            load_model(os.path.join(model_path, model_file))
            for model_file in os.listdir(model_path)
        ]
        load = True

    index = 0
    for model_file in os.listdir(model_path):
        label_name = model_file.split("_")[1]

        current_model = models[index]
        label = current_model.predict(sample["texto"].replace("\n", " "))

        sample[label_name] = label[0][0].replace("__label__", "")
        index += 1

    return sample


def process_arrow(file_path, model_path, n_processes):
    shard_dataset = load_dataset("arrow", data_files=[file_path])

    shard_dataset = shard_dataset.map(
        lambda sample: get_labels(sample, model_path),
        num_proc=n_processes,
    )

    return shard_dataset


def delete_cache(path):
    for file in os.listdir(path):
        if "cache" in file:
            os.remove(f"{path}/{file}")

    cache_dir = config.HF_DATASETS_CACHE

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)


def concatenate_all(datasets):
    final_dataset = load_from_disk(datasets[0])["train"]
    shutil.rmtree(datasets[0])

    for dataset in datasets[1:]:
        try:
            final_dataset = concatenate_datasets(
                [final_dataset, load_from_disk(dataset)["train"]]
            )
            shutil.rmtree(dataset)

        except Exception as e:
            logging.error(f"Error concatenating datasets: {str(e)}")

    return final_dataset


def main(args):
    files = sorted(glob(f"{args.dataset_path}/*.arrow"))
    processed_files = set(sorted(glob(f"{args.output_path}/*.arrow")))

    logging.info(f"Found {len(files)} files in {args.dataset_path}")
    logging.info(f"Found {len(processed_files)} processed files in {args.output_path}")
    all_files = []

    for file_path in tqdm(files, desc="Processing files"):
        file_name = os.path.basename(file_path)
        processed_file = os.path.join(args.output_path, file_name)

        if processed_file in processed_files:
            logging.info(f"File {file_name} has already been processed")
            all_files.append(processed_file)
            continue

        logging.info(f"Processing file {file_name}")

        arrow = process_arrow(file_path, args.models_path, args.n_processes)
        if arrow:
            arrow.save_to_disk(processed_file)
            all_files.append(processed_file)

        delete_cache(args.dataset_path)

    if all_files:
        logging.info(f"Processed {len(all_files)} files")
        final_dataset = concatenate_all(all_files)

        os.rmdir(args.output_path)
        final_dataset.save_to_disk(args.output_path)

        logging.info(f"Dataset saved in {args.output_path}")
        delete_cache(args.dataset_path)
    else:
        logging.info("No files have been processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Data/dataset",
        help="Path to the directory containing the dataset",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="Data/labelled_dataset",
        help="Path to the directory where the labeled dataset will be saved",
    )

    parser.add_argument(
        "--models_path",
        type=str,
        default="models/",
        help="Path to the directory with the fastText models",
    )

    parser.add_argument(
        "--n_processes",
        type=int,
        default=20,
        help="Number of parallel processes to use",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
