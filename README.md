<div align="center">
    <img width="400" height="350" src="/img/cyphertune-logo.webp">
</div>

<h5 align="center">
  ⚠️<em>Under Active Development</em> ⚠️
</h5>

**CypherTune** is a compact and intuitive Python library designed to fine-tune large language models (LLMs) specifically for text-2-Cypher applications. [Cypher](https://neo4j.com/developer/cypher/), the graph query language utilized by Neo4j, is renowned for its efficiency in data retrieval from knowledge graphs. Its user-friendly nature, drawing parallels with SQL, is attributed to an intuitive syntax that resonates well with those familiar with traditional programming languages.

This repository takes inspiration from Neo4j's recent [initiative](https://bratanic-tomaz.medium.com/crowdsourcing-text2cypher-dataset-e65ba51916d4) to create their inaugural open-source text-2-Cypher dataset. Our goal with CypherTune is to simplify the process of fine-tuning LLMs, making it more accessible, especially for individuals who are new to the realm of AI. We aim to provide an approachable gateway for diving into the world of LLMs and graph query language generation.

Contributions and participation in this crowdsourcing effort is welcomed! If you're interested in being a part of this exciting initiative, feel free to join and contribute to Neo4j's [application](https://text2cypher.vercel.app/) 🚀🚀.

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

tuner = CypherTuner(
    project_name="cyphertune-training-run1",
    model_id="codellama/CodeLlama-7b-Instruct-hf",
    dataset_id="zeroshot/text-2-cypher"
)

tuner.train()
```

# HyperParameter Configuration <img align="center" width="30" height="29" src="https://media.giphy.com/media/3o85xmYEd5Ml5zT6QU/giphy.gif">