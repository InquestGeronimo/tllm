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

Hyperparameter configuration and LoRA settings can be particularly challenging for those new to AI engineering. This repository aims to reduce reliance on hyperparameters, yet it's beneficial to have a solid understanding of them before training, especially if you plan to adjust them yourself.

The three primary factors influencing hyperparameters during training are dataset size, model type and size, and the type and amount of available hardware. After the Text-2-Cypher dataset has completed being crowdsourced, we will conduct multiple training runs for us to get a good understanding of baseline hyperparameters that perform well, expediting the fine-tuning process using CypherTune.

The CypherTune's hyperparameters can be found in the constructor of the `CypherTuner` class in the `train.py` module.

The first set is regarding [LoRA](https://huggingface.co/docs/peft/en/package_reference/lora) or the adapter.

```py
  # LoRA configuration settings
  r=8,                  # The size of the LoRA adjustments. It determines the level of detail in the modifications LoRA applies.
  lora_alpha=16,        # This is the scaling factor for LoRA. It controls the magnitude of the adjustments made by LoRA.
  target_modules=[      # Specifies the parts of the model where LoRA is applied. These can be components of the transformer architecture.
      "q_proj", 
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj", 
      "down_proj",
      "lm_head",
  ],
  bias="none",          # Indicates that biases are not adapted as part of the LoRA process.
  lora_dropout=0.05,    # The dropout rate for LoRA layers. It's a regularization technique to prevent overfitting.
  task_type="CAUSAL_LM" # Specifies the type of task. Here, it indicates the model is for causal language modeling.
```

The 2nd set of parameters is for the training job itself:

```py
  # Trainer configuration settings
  output_dir="./output",               # Directory where the training outputs and model checkpoints will be written.
  warmup_steps=1,                      # Number of steps to perform learning rate warmup.
  per_device_train_batch_size=32,      # Batch size per device during training.
  gradient_accumulation_steps=1,       # Number of updates steps to accumulate before performing a backward/update pass.
  gradient_checkpointing=True,         # Enables gradient checkpointing to save memory at the expense of slower backward pass.
  max_steps=1000,                      # Total number of training steps to perform.
  learning_rate=2.5e-5,                # Initial learning rate for the optimizer.
  bf16=True,                           # Use bfloat16 mixed precision training instead of the default fp32.
  optim="paged_adamw_8bit",            # The optimizer to use, here it's a variant of AdamW optimized for 8-bit computing.
  logging_dir="./logs",                # Directory to store logs.
  save_strategy="steps",               # Strategy to use for saving a model checkpoint ('steps' means saving at every specified number of steps).
  save_steps=100,                      # Number of steps to save a checkpoint after.
  evaluation_strategy="steps",         # Strategy to use for evaluation ('steps' means evaluating at every specified number of steps).
  eval_steps=2,                        # Number of training steps to perform evaluation after.
  do_eval=True,                        # Whether to run evaluation on the validation set.
  report_to="wandb",                   # Tool to use for logging and tracking (Weights & Biases in this case).
  remove_unused_columns=True,          # Whether to remove columns not used by the model when using a dataset.
  run_name=f"run-name",                # Name of the experiment run, usually containing the project name and timestamp.
```
