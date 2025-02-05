import argparse
import json
import os

import fasttext
from datasets import Dataset, concatenate_datasets

def normalize_data(dataset, size, labels):
    subsets = []
    N = int(size / len(labels))

    for label in labels:
        subset = dataset.filter(lambda x: x['label'] == label)
        subset = subset.shuffle(seed=42).select(range(N))
        subsets.append(subset)

    return concatenate_datasets(subsets).shuffle(seed=42)



def compute_metrics(model, model_path, labels):
    valid_path = f"{model_path}/data.valid"
    with open(valid_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    for label in labels:
        label_tag = f"__label__{label.replace(' ', '-')}"
        filtered_lines = [line for line in data if label_tag in line]
        
        with open(f"{model_path}/data_valid.txt", "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)

        
        os.system(f"head -n {len(filtered_lines)} {model_path}/data_valid.txt > {valid_path}")
        result = model.test(valid_path)
        print(f"Label: {label}, Precision: {result[1]}, Recall: {result[2]}")


def main(args):
    data_raw = Dataset.load_from_disk(args.dataset_path)
    labels = data_raw.unique("label")

    dataset = normalize_data(data_raw, args.size, labels) if args.normalize else data_raw.shuffle(seed=42).select(range(args.size))
    dataset = dataset.map(lambda x: {"text": x["text"], "label": f"__label__{x['label'].replace(' ', '-')}"})
    
    if args.all:
        data_raw = data_raw.map(lambda x: {"text": x["text"], "label": f"__label__{x['label'].replace(' ', '-')}"})
        dataset_train = dataset
        train_texts = set(x['text'] for x in dataset_train)
        dataset_valid = data_raw.filter(lambda x: x['text'] not in train_texts)

        with open(f"{args.model_path}/data_train.txt", "w", encoding="utf-8") as f:
            for sample in dataset_train:
                f.write(str(sample['label']) + " " + str(sample['text'].replace("\n", " ")) + "\n")

        with open(f"{args.model_path}/data_valid.txt", "w", encoding="utf-8") as f:
            for sample in dataset_valid:
                f.write(str(sample['label']) + " " + str(sample['text'].replace("\n", " ")) + "\n")

        os.system(f"head -n {args.size} {args.model_path}/data_train.txt > {args.model_path}/data.train")
        os.system(f"head -n {len(data_raw) - args.size} {args.model_path}/data_valid.txt > {args.model_path}/data.valid")

    else:
        with open(f"{args.model_path}/data.txt", "w", encoding="utf-8") as f:
            for sample in dataset:
                f.write(str(sample['label']) + " " + str(sample['text'].replace("\n", " ")) + "\n")

        os.system(f"head -n {int(args.size * 0.8)} {args.model_path}/data.txt > {args.model_path}/data.train")
        os.system(f"tail -n {int(args.size * 0.2)} {args.model_path}/data.txt > {args.model_path}/data.valid")

    if args.automatic:
        model = fasttext.train_supervised(input=f"{args.model_path}/data.train", autotuneValidationFile=f"{args.model_path}/data.valid", autotuneDuration=600)
        model.save_model(f"{args.model_path}/auto_model_{args.size}.bin")

    else:
        model = fasttext.train_supervised(input=f"{args.model_path}/data.train", epoch = args.num_epochs, lr=args.learning_rate, wordNgrams=args.n_grams)
        model.save_model(f"{args.model_path}/model_{args.size}.bin")


    
    model_path = f"{args.model_path}/{'auto_' if args.automatic else ''}model_{args.size}.bin"
    model.save_model(model_path)

    
    print(model.test(f"{args.model_path}/data.valid"))
    compute_metrics(model, args.model_path,labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure FastText training settings")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory to save model"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Data/labelled_dataset",
        help="Path to the directory file containing the dataset"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for training"
    )

    parser.add_argument(
        "--n_grams",
        type=int,
        default=1,
        help="Number of n-grams to use"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Size of the model"
    )

    parser.add_argument(
        "--normalize",
        default=False,
        help="Normalize the dataset"
    )

    parser.add_argument(
        "--automatic",
        default=False,
        help="Use automatic hyperparameter tuning"
    )

    parser.add_argument(
        "--all",
        default=False,
        help="Use all the dataset to validate"
    )
    args = parser.parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    
    main(args)
