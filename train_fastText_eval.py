import argparse
import os
import re
import json
import fasttext
from datasets import Dataset, concatenate_datasets

def preprocess_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'([.!?,\'\"/()])', r' \1 ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main(args):

    with open(args.json_path) as f:
        data = json.load(f)
    text_dataset = Dataset.load_from_disk(args.dataset_path)
    print(f"Loaded {len(data)} labelled samples from {len(text_dataset)}")

    combined_data_list = []
    combined_antidata_list = []

    for val in range(len(data)):
    
        label = data[f"{val}"]["Label"]
        if not label:
            continue

        if args.label == label:
            combined_data_list.append({"text": text_dataset[val]["text"][:4000], "Label": args.label})
        else:
            combined_antidata_list.append({"text": text_dataset[val]["text"][:4000], "Label": "Not_" + args.label})
        
    data_label = Dataset.from_list(combined_data_list)
    size = len(data_label)
    
    antidata = Dataset.from_list(combined_antidata_list).select(range(min(len(combined_antidata_list),size)))

    data_neutra = Dataset.load_from_disk(args.neutral_path).shuffle().select(range(size-len(antidata)))
    data_neutra = data_neutra.map(lambda x: {"text":x["texto"]  ,"Label": "Not_" + args.label})

    dataset = concatenate_datasets([data_label, antidata, data_neutra])
    dataset = dataset.map(lambda x: {"text": preprocess_text(x["text"]), "Label": f"__label__{x['Label'].replace(' ', '-')}"})
    dataset = dataset.shuffle(seed=41)

    with open(f"{args.model_path}/data.txt", "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(str(sample['Label']) + " " + str(sample['text'].replace("\n", " ")) + "\n")

    os.system(f"head -n {int(size*2 * 0.8)} {args.model_path}/data.txt > {args.model_path}/data.train")
    os.system(f"tail -n {int(size*2 * 0.2)} {args.model_path}/data.txt > {args.model_path}/data.valid")

    model = fasttext.train_supervised(input=f"{args.model_path}/data.train",  lr=0.1, epoch=25, wordNgrams=2)
    model.save_model(args.model_path + f'/{args.label}{size//1000}k.bin')
    
    res =model.test(f"{args.model_path}/data.valid")
    print(f"Metrics: {res}")
    print(f"Model saved to {args.model_path}/{args.label}{size//1000}k.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure FastText training settings")

    parser.add_argument(
        "--json_path",
        type=str,
        default="Data/labelled.json",
        help="Path to the JSON file with the labelled data"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Data/Dataset_of_the_labels",
        help="Path to the dataset"
    )

    parser.add_argument(
        "--neutral_path",
        type=str,
        default="Data/Neutro",
        help="Path to the neutral dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory to save model"
    )

    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label to train the model on"
    )

    args = parser.parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    args.label = args.label.replace("_", " ")
    main(args)

