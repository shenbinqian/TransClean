# TransClean: A Benchmark for Detecting and Extracting Clean Translations from Large Language Model Outputs

This repository contains the code and data of our paper [TransClean: A Benchmark for Detecting and Extracting Clean Translations from Large Language Model Outputs]() presented at WMT26. The code and data are released for reproducibility purposes. Please visit our [Hugging Face repo](https://huggingface.co/datasets/shenbinqian/TransClean), if you want to use our data to detect and extract clean translations from LLM outputs. More details will be released soon.

## Environment and Reproducibility

The following instructions describe a minimal environment and common commands to reproduce the dataset creation and evaluation used in this repository.

### Python

- Recommended: Python 3.10 or newer.

### Install (minimal)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install required packages (example minimal set):

```bash
pip install --upgrade datasets huggingface-hub transformers sentencepiece tqdm
```

Optionally, you can create a `requirements.txt` file with the above packages and install with `pip install -r requirements.txt`.

### Reproducing the benchmark

- Generate the synthetic subset (uses `data/clean_samples.jsonl`):

```bash
python bench_creating/generate_noisy_translations.py
```

- Generate silver labels for the curated set:

```bash
python bench_creating/generate_silver_labels_api.py
# or
python bench_creating/generate_silver_labels_vllm.py
```

- Check agreement and aggregate labels:

```bash
python bench_creating/check_silver_agreement.py
python bench_creating/aggregate_silver_labels.py
```

### Running extraction and evaluation

- Run QE baseline extraction:

```bash
python methods/qe_baseline.py
```

- Run LLM-based extraction:

```bash
python methods/llm_extractor.py
```

- Compute detection and extraction accuracies:

```bash
python methods/evaluate_benchmark.py
```

### Loading the dataset locally

You can load the JSONL subset with the `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="data/synthetic.jsonl", split="train")
```

Or from our Hugging Face dataset:

```python
from datasets import load_dataset
ds = load_dataset("shenbinqian/TransClean")
```

### Contact

For questions or issues, please open an issue in this repository or contact the maintainers via the Hugging Face dataset page.

If you need the `curated subset`, pleaes contact us: shenbinq@ifi.uio.no.