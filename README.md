<div align="center">
    <img width="400" height="350" src="/img/cyphertune-logo.webp">
</div>

<h4 align="center">
  ⚠️<em>Under Active Development</em> ⚠️
</h4>

**CypherTune** is a small and simple Python library for fine-tuning large language models (LLMs) on text-to-Cypher datasets. [Cypher](https://neo4j.com/developer/cypher/) is the graph query language of Neo4j, designed to retrieve data efficiently from a knowledge graph. Drawing inspiration from SQL, Cypher stands out as the most user-friendly graph language, owing to its intuitive syntax and resemblance to familiar languages.

This repository is inspired by Neo4j's recent [initiative](https://bratanic-tomaz.medium.com/crowdsourcing-text2cypher-dataset-e65ba51916d4) to crowdsource the development of their first open-source text-to-Cypher dataset. It aims to provide a streamlined and accessible platform for fine-tuning LLMs for users with minimal background in AI, offering an easy entry point to get started with LLMs and graph query language generation.

If you want to help with the crowdsourcing initiative, don't be shy and visit their [app](https://text2cypher.vercel.app/)  💁.

# Features

CypherTune streamlines the training process by abstracting the complexities of the Hugging Face ecosystem. It currently offers the following features:

- **Model Fine-Tuning**: Fine-tune LLMs with custom text-to-cypher datasets from 🤗.
- **Bits and Bytes**: Optimizes model performance with 4-bit quantization.
- **QLoRA**: Fine-tuning using LoRA, a popular and lightweight training technique that significantly reduces the number of trainable parameters.
- **Prompt Template**: Doc conversion into a prompt template for fine-tuning.
- **Weights & Biases Integration**: Track and log your experiments using wandb. (optional)

TODO

> - Table storing hyperparameter configurations for select training job environments (i.e. depending on dataset size, model type/size and amount/type of compute).
> - Model eval functionality post-training.
> - Fully Sharded Data Parallel (FSDP): Utilizes efficient training across distributed systems.

# Install <img align="center" width="30" height="29" src="https://media.giphy.com/media/sULKEgDMX8LcI/giphy.gif">

```
pip install cyphertune
```

# Start Training <img align="center" width="30" height="29" src="https://media.giphy.com/media/QLcCBdBemDIqpbK6jA/giphy.gif">

To start training, the minimum requirement is to pass a `project name`, and your Hugging Face `model`/`dataset` stubs. The trainer is expecting to ingest a `train` and `validation` split from your dataset prior to training. Here is a placeholder [example](https://huggingface.co/datasets/zeroshot/text-2-cypher) of the dataset format the trainer is expecting to receive.

```py
from cyphertune import CypherTuner

model = CypherTuner(
    project_name="cypher-training-run",
    model_id="codellama/CodeLlama-7b-Instruct-hf",
    dataset_id="zeroshot/text-2-cypher"
)

tuner.train(model)
```