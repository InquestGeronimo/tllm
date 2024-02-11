<h1 align="center">
CypherTune
</h1>

CypherTune is a library for fine-tuning large language models (LLMs) on text-to-Cypher datasets. 


Inspired by Neo4j's recent initiative to crowdsource the first open sourced text-to-cypher dataset, this repository offers users a simplified and seamless approach to fine-tuning LLMs with a minimal background in AI to get started.



## Features

- **Model Fine-Tuning**: Fine-tune LLMs with custom text-to-cypher datasets from 🤗.
- **Bits and Bytes**: Optimizes model performance with 4-bit quantization.
- **LoRA**: Adapter fine-tuning ability through LoRA (Low-Rank Adaptation).
- **Promp Template**: Doc conversion into a prompt template for fine-tuning.
- **Weights & Biases Integration**: Track and log your experiments using wandb.

(Under Development)
> Fully Sharded Data Parallel (FSDP)**: Utilizes FSDP for efficient training across distributed systems.

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Datasets
- Accelerate
- Peft
- bitsandbytes
- wandb (optional for logging)

# Install <img align="center" width="30" height="29" src="https://media.giphy.com/media/sULKEgDMX8LcI/giphy.gif">
<br>

```
pip install cyphertune
```

## Usage

Setting Up
First, clone the repository to your local machine:

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

# Initialize the Trainer <img align="center" width="30" height="29" src="https://media.giphy.com/media/QLcCBdBemDIqpbK6jA/giphy.gif">
<br>

Import and initialize the `LLMTrainer` class from the script:

```py
from llm_trainer import LLMTrainer

trainer = LLMTrainer(
    project_name="YourProjectName",
    model_id="ModelIdentifier",
    dataset_id="DatasetIdentifier"
)
```

