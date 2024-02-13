<div align="center">
    <img width="400" height="350" src="/img/cyphertune-logo.webp">
</div>

<h5 align="center">
  ⚠️<em>Under Active Development</em> ⚠️
</h5>

**CypherTune** is an encapsulated Python library designed to fine-tune large language models (LLMs) specifically for text-2-Cypher tasks. [Cypher](https://neo4j.com/developer/cypher/), the graph query language utilized by Neo4j, is renowned for its efficiency in data retrieval from knowledge graphs. Its user-friendly nature, drawing parallels with SQL, is attributed to an intuitive syntax that resonates well with those familiar with traditional programming languages.

This repository takes inspiration from Neo4j's recent [initiative](https://bratanic-tomaz.medium.com/crowdsourcing-text2cypher-dataset-e65ba51916d4) to create their inaugural open-source text-2-Cypher dataset. Our goal with CypherTune is to simplify the process of fine-tuning LLMs, making it more accessible, especially for individuals who are new to the realm of AI. We aim to lower the barrier of entry for Neo4j users to fine-tune LLMs using the text-2-Cypher dataset once it's released to the public.

Contributions and participation in this crowdsourcing effort is welcomed! If you're interested in being a part of this exciting initiative, feel free to join and contribute to Neo4j's [application](https://text2cypher.vercel.app/) 🚀🚀.

# Features <img align="center" width="30" height="29" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOTBqaWNrcGxnaTdzMGRzNTN0bGI2d3A4YWkxajhsb2F5MW84Z2dxaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26tOZ42Mg6pbTUPHW/giphy.gif">

CypherTune streamlines the training process by abstracting several libraries from the Hugging Face ecosystem. It currently offers the following features:

- **LLM Fine-Tuning**: Fine-tunes LLMs using HF's `Trainer` for custom text-to-Cypher datasets stored in the [Hub](https://huggingface.co/datasets).
- **Bits and Bytes**: Loads model with 4-bit quantization.
- **PEFT**: Uses [LoRA](https://arxiv.org/pdf/2106.09685.pdf) under the hood, a popular and lightweight training technique that significantly reduces the number of trainable parameters. We combine 4-bit quantization lowering the barrier for the amount of memory required during training ([QLoRA](https://arxiv.org/abs/2305.14314)).
- **Dataset Preprocessing**: Converts dataset into a prompt template for fine-tuning.
- **Weights & Biases Integration**: Track and log your experiments using wandb.

TODO

> - Table storing hyperparameter configurations for select training job environments (i.e. depending on dataset size, model type/size and amount/type of compute).
> - Model eval functionality post-training.
> - `Max length` to be determined after dataset is released.
> - Make dataset headers dynamic, right now they are static.
> - Provide inference snippet for testing after training.
> - Fully Sharded Data Parallel (FSDP): Utilizes efficient training across distributed systems.

# Install <img align="center" width="30" height="29" src="https://media.giphy.com/media/sULKEgDMX8LcI/giphy.gif">

```
pip install cyphertune
```

# Start Training <img align="center" width="30" height="29" src="https://media.giphy.com/media/QLcCBdBemDIqpbK6jA/giphy.gif">

To start training, the only requirements are a `project name`, your Hugging Face `model`/`dataset` stubs and the path to your YAML `config_file`. This file includes the essential LoRA and training arguments for fine-tuning. Before beginning the training process, ensure you download the YAML file from this repository using either the curl or wget commands to access its contents.

```bash
curl -o config.yml https://raw.githubusercontent.com/InquestGeronimo/cyphertune/main/cyphertune/config.yml
```

```bash
wget -O config.yml https://raw.githubusercontent.com/InquestGeronimo/cyphertune/main/cyphertune/config.yml
```

The trainer expects to ingest a `train` and `validation` split from your dataset prior to training with a specific format. Here is an [example](https://huggingface.co/datasets/zeroshot/text-2-cypher) of a placeholder dataset, designed to demonstrate the expected format.

```py
from cyphertune import CypherTuner

tuner = CypherTuner(
    project_name="cyphertune-training-run1",
    model_id="codellama/CodeLlama-7b-Instruct-hf",
    dataset_id="zeroshot/text-2-cypher",
    config_file="path/to/config.yml"
)

tuner.train()
```
After training completes, the adapter will be saved in your output directory. The pre-trained model will not be saved.

# HyperParameter Knowledge <img align="center" width="30" height="29" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXV1bWFyMWxkY3JocjE1ZDMxMWZ5OHZtejFkbHpuZXdveTV3Z3BiciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bGgsc5mWoryfgKBx1u/giphy.gif">


Configuring hyperparameters and LoRA settings can be a complex task, especially for those new to AI engineering. Our repository is designed to lessen the dependence on hyperparameters. For instance, once the Text-2-Cypher dataset is fully crowdsourced, we will undertake multiple training sessions. This will help us establish a set of baseline hyperparameters known to yield good results, and we will save them to this repository, thereby streamlining the fine-tuning process for Neo4j users. However, having a thorough understanding of these hyperparameters is still advantageous, particularly if you intend to modify them yourself.

Three key factors affect hyperparameters during training:

1. The type and size of the model.
2. The type and quantity of hardware.
3. The size of the dataset.

For accessing CypherTune's hyperparameters, you can refer to the [config_file](https://github.com/InquestGeronimo/cyphertune/blob/main/cyphertune/config.yml). At the time of writing, the Text-2-Cypher dataset is not publicly available, so the parameters in the file serve as placeholders.

The first set of parameters pertains to the [LoRA](https://huggingface.co/docs/peft/en/package_reference/lora) settings:

```py
  # LoRA configuration settings
  r=8,                  # The size of the LoRA's rank. Opting for a higher rank could negate the efficiency benefits of using LoRA. The higher the rank the largar the checkpoint file is.
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

For further details, refer to the PEFT [documentation](https://huggingface.co/docs/peft/en/package_reference/lora) or read the blogs found at the end of this README.

The following set pertains specifically to the training arguments:

```py
  # Trainer configuration settings
  output_dir="./output",               # Directory where the training outputs and model checkpoints will be written.
  warmup_steps=1,                      # Number of steps to perform learning rate warmup.
  per_device_train_batch_size=32,      # Batch size per device during training.
  gradient_accumulation_steps=1,       # Number of updates steps to accumulate before performing a backward/update pass.
  gradient_checkpointing=True,         # Enables gradient checkpointing to save memory at the expense of slower backward pass.
  max_steps=1000,                      # Total number of training steps to perform.
  learning_rate=1e-4,                  # Initial learning rate for the optimizer.
  bf16=True,                           # Use bfloat16 mixed precision training instead of the default fp32.
  optim="paged_adamw_8bit",            # The optimizer to use, here it's a variant of AdamW optimized for 8-bit computing.
  logging_dir="./logs",                # Directory to store logs.
  save_strategy="steps",               # Strategy to use for saving a model checkpoint ('steps' means saving at every specified number of steps).
  save_steps=25,                       # Number of steps to save a checkpoint after.
  evaluation_strategy="steps",         # Strategy to use for evaluation ('steps' means evaluating at every specified number of steps).
  eval_steps=50,                       # Number of training steps to perform evaluation after.
  do_eval=True,                        # Whether to run evaluation on the validation set.
  report_to="wandb",                   # Tool to use for logging and tracking (Weights & Biases in this case).
  remove_unused_columns=True,          # Whether to remove columns not used by the model when using a dataset.
  run_name="run-name",                 # Name of the experiment run, usually containing the project name and timestamp.
```
The provided parameters, while not comprehensive, cover the most critical ones for fine-tuning. Particularly, `per_device_train_batch_size` and `learning_rate` are the most sensitive and influential during this process.

# Resources

- [Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms?utm_source=substack&utm_campaign=post_embed&utm_medium=web)
- [Easily Train a Specialized LLM: PEFT, LoRA, QLoRA, LLaMA-Adapter, and More](https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft#:~:text=LoRA%20leaves%20the%20pretrained%20layers,of%20the%20model%3B%20see%20below.&text=Rank%20decomposition%20matrix.,the%20dimensionality%20of%20the%20input.)