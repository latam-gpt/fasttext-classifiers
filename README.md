# Bootstrapping FastText with LLMs

When building a classifier using the methodology described in [llm-data-eval](https://github.com/latam-gpt/llm-data-eval), we often encounter difficulties with underrepresented classes. This repo presents a bootstrapping methodology that allows us to balance these classes in a more efficient way.

---

## Quickly Gathering Data for Underrepresented Classes

When training a binary classifier to detect a specific topic that appears in only a small percentage of the corpus, we face a challenge: in our previous methodology, this low percentage is preserved in the training set, making learning ineffective.

Instead of simply increasing the dataset size by sampling more data at random, we propose a more efficient approach:
1. Train a basic model using the initial dataset.
2. Run this model over the entire corpus using a high-confidence threshold.
3. Re-evaluate the positively classified samples using an LLM to improve label precision.
4. Use this curated data to train a new, stronger model.
5. Repeat this process until the classifier reaches a satisfactory performance.

---

## Pipeline Overview

The entire pipeline is a three-step process, with each step encapsulated in its own script.

---

## Labeling with an LLM

This step is implemented in `label_with_llms.py`.

It uses a base prompt (provided in `prompt.txt`) to guide the LLM in labeling a small portion of the dataset.

**Example usage:**

```bash
python label_with_llm.py \
--prompt_path prompts/prompt.txt \
--dataset_path datasets/Chile \
--model_path nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
--download_dir workspace1/llm-models \
--label Chile \
--output_path results/results_Chile.json
```

---

## Train FastText

Implemented in `train_fastText_eval.py`.

Using the labeled dataset generated in Step 1, this script trains a FastText model. To improve the model's performance quickly, we leverage the false positive adding them to the training set to improve recall.

Optionally, you can provide a base dataset of known "neutral" data — unrelated to any of the target classes — to serve as negative examples.

In our original experiment, we progressively scaled the number of positive examples using sizes such as: 1k → 5k → 15k → 35k → 75k.

**Example usage:**

```bash
python train_fastText_eval.py \
--json_path results/results_Chile.json \
--dataset_path datasets/Chile \
--neutral_path datasets/Data_Neutra \
--model_path fastText_models/ \
--label Chile
```

---

## Run FastText Classifier

Implemented in `run_fastText_eval.py`.

Once a reasonably good classifier is trained, we run it over the full corpus and collect only the rows classified positive. These samples can then be:
- Used for evaluation,
- Labeled again with an LLM for a new iteration,

**Example usage:**

```bash
python run_fastText_eval.py \
--dataset_path big_corpus/red_pajama \
--output_path dataset_bootstrapped/Chile/ \
--model_path fastText_models/Chile.bin \
--size 75000 \
--cache_dir .cache/fastText/
```

---

## Summary

This bootstrapping pipeline allows us to iteratively improve classification of underrepresented topics in large corpus using a combination of LLMs and FastText. It provides a practical way to build balanced and accurate datasets with minimal manual labeling.
