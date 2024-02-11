<div align="center">
    <img width="400" height="350" src="/img/cyphertune-logo.webp">
</div>

<h5 align="center">
  ⚠️<em>Under Active Development</em> ⚠️
</h5>

**CypherTune** is a compact and intuitive Python library designed to fine-tune large language models (LLMs) specifically for text-2-Cypher tasks. It uses [LoRA](https://arxiv.org/pdf/2106.09685.pdf) under the hood which allows for most of the pre-trained LLM's parameters to remain frozen during fine-tuning. Consequently, only a minimal number of parameters in the adapter are adjusted.

[Cypher](https://neo4j.com/developer/cypher/), the graph query language utilized by Neo4j, is renowned for its efficiency in data retrieval from knowledge graphs. Its user-friendly nature, drawing parallels with SQL, is attributed to an intuitive syntax that resonates well with those familiar with traditional programming languages.

This repository takes inspiration from Neo4j's recent [initiative](https://bratanic-tomaz.medium.com/crowdsourcing-text2cypher-dataset-e65ba51916d4) to create their inaugural open-source text-2-Cypher dataset. Our goal with CypherTune is to simplify the process of fine-tuning LLMs, making it more accessible, especially for individuals who are new to the realm of AI. We aim to lower the barrier of entry for Neo4j users to fine-tune LLMs using the text-2-Cypher dataset once it's released to the public.

Contributions and participation in this crowdsourcing effort is welcomed! If you're interested in being a part of this exciting initiative, feel free to join and contribute to Neo4j's [application](https://text2cypher.vercel.app/) 🚀🚀.

# Features

CypherTune streamlines the training process by abstracting several libraries from the Hugging Face ecosystem. It currently offers the following features:

- **Model Fine-Tuning**: Fine-tunes LLMs using HF's `Trainer` for custom text-to-Cypher datasets stored in the [Hub](https://huggingface.co/datasets).
- **Bits and Bytes**: Loads model with 4-bit quantization.
- **QLoRA**: Fine-tuning using LoRA, a popular and lightweight training technique that significantly reduces the number of trainable parameters. It works by quantizing the precision of the weight parameters in the pre trained LLM to 4-bit precision lowering the barrier for the amount of memory required during training. FYI, LLM fine-tuning is mostly a memory bound task.
- **Prompt Template**: Doc conversion into a prompt template for fine-tuning.
- **Weights & Biases Integration**: Track and log your experiments using wandb.

TODO

> - Table storing hyperparameter configurations for select training job environments (i.e. depending on dataset size, model type/size and amount/type of compute).
> - Model eval functionality post-training.
> - Make dataset headers dynamic, right now they are static.
> - Provide inference snippet for testing after training.
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
After training completes, the adapter will be saved in your output directory. The pre-trained model will not be saved.

# HyperParameter Configuration <img align="center" width="30" height="29" src="https://media.giphy.com/media/FhE5Og89nRTpt4QCet/giphy.gif">


Configuring hyperparameters and LoRA settings can be a complex task, especially for those new to AI engineering. Our repository is designed to lessen the dependence on hyperparameters. For instance, once the Text-2-Cypher dataset is fully crowdsourced, we will undertake multiple training sessions. This will help us establish a set of baseline hyperparameters that are known to yield good results, thereby streamlining the fine-tuning process with CypherTune. However, having a thorough understanding of these hyperparameters is still advantageous, particularly if you intend to modify them yourself during training.

Three key factors that affect hyperparameters during training:

1. The size of the dataset.
2. The type and size of the model.
3. The type and quantity of hardware.

For accessing CypherTune's hyperparameters, you can refer to the `CypherTuner` class constructor within the [train.py](https://github.com/InquestGeronimo/cyphertune/blob/main/cyphertune/train.py) module. This should provide detailed insights into the default configuration used in the tuning process.

The first set is regarding [LoRA](https://huggingface.co/docs/peft/en/package_reference/lora) or the adapter:

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

For further detail, refer to the PEFT [documentation](https://huggingface.co/docs/peft/en/package_reference/lora) on LoRA

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
  run_name="run-name",                 # Name of the experiment run, usually containing the project name and timestamp.
```
The provided parameters, while not comprehensive, cover the most critical ones for fine-tuning. Particularly, `per_device_train_batch_size` and `learning_rate` are the most sensitive and influential during this process.

# Resources

- [Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms?utm_source=substack&utm_campaign=post_embed&utm_medium=web)
- [Easily Train a Specialized LLM: PEFT, LoRA, QLoRA, LLaMA-Adapter, and More](https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft#:~:text=LoRA%20leaves%20the%20pretrained%20layers,of%20the%20model%3B%20see%20below.&text=Rank%20decomposition%20matrix.,the%20dimensionality%20of%20the%20input.)