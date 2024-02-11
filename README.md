<div align="center">
    <img width="400" height="350" src="/img/cyphertune-logo.webp">
</div>
<br>

**CypherTune** is a small and simple library for fine-tuning large language models (LLMs) on text-to-Cypher datasets. [Cypher](https://neo4j.com/developer/cypher/) is the graph query language of Neo4j, designed to retrieve data efficiently from a knowledge graph. Drawing inspiration from SQL, Cypher stands out as the most user-friendly graph language, owing to its intuitive syntax and resemblance to familiar languages.

This repository is inspired by Neo4j's recent initiative to crowdsource the development of their first open-source text-to-Cypher dataset. It aims to provide a streamlined and accessible platform for fine-tuning Large Language Models (LLMs). This resource is particularly tailored for users with minimal background in AI, offering an easy entry point to get started with LLMs and graph query language integration.


# Features

- **Model Fine-Tuning**: Fine-tune LLMs with custom text-to-cypher datasets from 🤗.
- **Bits and Bytes**: Optimizes model performance with 4-bit quantization.
- **QLoRA**: Fine-tuning using LoRA, a popular and lightweight training technique that significantly reduces the number of trainable parameters.
- **Promp Template**: Doc conversion into a prompt template for fine-tuning.
- **Weights & Biases Integration**: Track and log your experiments using wandb. (optional)

Under Development

> - YAML file library storing hyperparameter configurations for select training job environments (i.e. depending on dataset size, model type/size and amount/type of compute).
> - Model eval post training
> - Fully Sharded Data Parallel (FSDP)**: Utilizes FSDP for efficient training across distributed systems.

# Install <img align="center" width="30" height="29" src="https://media.giphy.com/media/sULKEgDMX8LcI/giphy.gif">
<br>

```
pip install cyphertune
```

# Launch CypherTuner <img align="center" width="30" height="29" src="https://media.giphy.com/media/QLcCBdBemDIqpbK6jA/giphy.gif">

To start training, initialize the `CypherTuner` class from the script. Pass along your project name, Hugging Face model and dataset stubs. This trainer expects you to have a `train` and `validation` split for your dataset. Here is a placeholder [example](https://huggingface.co/datasets/zeroshot/text-to-cypher) of the dataset format.

```py
from cyphertune import CypherTuner

trainer = CypherTuner(
    project_name="CypherTraining",
    model_id="meta-llama/Llama-2-7b-hf",
    dataset_id="zeroshot/text-to-cypher"
)
```

### Load and Preprocess Datasets and Set training Configuration

```py
train_data, eval_data = trainer.load_datasets()
model, tokenizer = trainer.load_model_and_tokenizer()
train_data, eval_data = trainer.create_prompts_from_datasets(tokenizer, train_data, eval_data)
```

### Configure Training Job

```py
model = trainer.configure_lora(model)
trainer = trainer.configure_training(model, tokenizer, train_data, eval_data)
```

### Start training!

```py
trainer.train_model(trainer)
```

