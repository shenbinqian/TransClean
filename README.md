# TransClean: A Benchmark for Detecting and Extracting Clean Translations from Large Language Model Outputs

This repository contains the code and data of our paper submitted anonymously for peer review. The code and data here are released for inspection and review purposes. We will have  a more detailed instruction on how to configure the environment and to use our dataset for clean translation extraction after the anonymity period.

## Benchmark Construction

Use `bench_creating/generate_noisy_translations.py` to generate the synthetic subset for the selected `data/clean_samples.jsonl`.

Use `bench_creating/generate_silver_labels_*.py` to create the silver labels for the `data/curated_noisy_1100.jsonl`.

Use `bench_creating/check_silver_agreement.py` to save disagreements for human inspection, and `bench_creating/aggregate_silver_labels.py` to get the final silver labels.

## Extraction Methods

Run `methods/qe_baseline.py` for the QE-based extraction and `methods/llm_extractor.py` for the LLM-based extraction method.

`evaluate_benchmark.py` is used to compute the detection and extraction accuracies for the above methods after running and getting the results.

## Benchmark Dataset

The final benchmark dataset described in our paper (synthetic and curated) is in `data/synthetic.jsonl` and `curated_noisy_1100_with_silver.jsonl` respectively.