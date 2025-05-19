import argparse
import re
import json
import torch
import logging
from datasets import Dataset
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

PATTERN_DOMAIN = r"LABEL:\s*([A-Za-zÀ-ÖØ-öø-ÿ ]+)"

def load_base_prompt(prompt_path):
    with open(prompt_path) as f:
        return f.read()
    
def update_text(example, base_prompt,tokenizer,label):
    base_prompt = base_prompt.replace("<TEXT>", f"{example[args.text_column]}")
    base_prompt = base_prompt.replace("<LABEL>", f"{label}")
    chat = [
        {"role": "user", "content": base_prompt},
    ]
    base_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return example

def process_batch(llm, batch, sampling_params,label):
    outputs = llm.generate(list(batch["eval_prompt"]), sampling_params)
    results = {}
    count = 0
    for idx, (vllm_output, prompt) in enumerate(
        zip(outputs, list(batch[args.text_column]))
    ):
        generated_text = vllm_output.outputs[0].text
        match_domain = re.search(PATTERN_DOMAIN, generated_text)
    
        if match_domain:
            results[idx] = {"Label": match_domain.group(1), "Output": generated_text}
            if match_domain.group(1) == label:
                count += 1
        else:
            results[idx] = {"Label": None, "Output": generated_text}

    
    return results,count

def main(args):
    try:
        base_prompt = load_base_prompt(args.prompt_path)
        logging.info(f"Loading dataset from {args.dataset_path}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        dataset = Dataset.load_from_disk(args.dataset_path)
        dataset = dataset.map(lambda x: {"texto": x["texto"][:4000]})
        dataset = dataset.map(lambda x: update_text(x, base_prompt,tokenizer,args.label))
        sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=600,
            truncate_prompt_tokens=32000,
        )

        logging.info(f"Initializing LLM with model {args.model_path}")
        llm = LLM(
            model=args.model_path,
            download_dir=args.download_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            dtype=torch.bfloat16  
        )
        
        results_dict = {}
        curr_idx = 0
        count_f = 0
        while len(dataset) > curr_idx + args.batch_size:
            curr_batch = dataset[curr_idx : curr_idx + args.batch_size]
            batch_results,count = process_batch(llm, curr_batch, sampling_params,args.label)
            count_f += count
            for idx, result in batch_results.items():
                results_dict[curr_idx + idx] = result

            with open(args.output_path, "w", encoding="utf-8") as json_file:
                json.dump(results_dict, json_file, ensure_ascii=False, indent=4)
            curr_idx += args.batch_size
            logging.info(f"Processed {curr_idx} examples")
            logging.info(f"Count of examples with topics: {count_f}")
            if count_f > args.limit:
                break

        logging.info(f"Processing complete. Results saved to {args.output_path}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and generate scores.")
    parser.add_argument("--prompt_path", type=str, default="utils/prompts.txt", help="Path to the base prompt file.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--model_path", type=str, default="nvidia/Llama-3_3-Nemotron-Super-49B-v1", help="Path or name of the model to use.")
    parser.add_argument("--download_dir", type=str, required=True, help="Directory to download the model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument("--text_column", type=str, default="texto", help="Name of the text column in the dataset.")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
    parser.add_argument("--max_model_len", type=int, default=32000, help="Max sequence length.")
    parser.add_argument("--label", type=str, required=True, help="Label to insert in the prompt.")
    parser.add_argument("--limit", type=int, default=750000, help="Limit of examples to process.")

    args = parser.parse_args()
    args.label = args.label.replace("_"," ")
    print(args)
    main(args)
