<div align="center">
    <img width="400" height="350" src="/img/logo.jpg">
</div>

**tLLM** is a comprehensive Python library that simplifies the fine-tuning of large language models (LLMs) for text generation. It is specifically designed to abstract away complexities associated with various libraries from the Hugging Face ecosystem, offering a more encapsulated and user-friendly approach.

Our goal with tLLM is to simplify the process of fine-tuning LLMs, making it more accessible, especially for individuals who are new to the realm of AI. The aim is to lower the barrier of entry for new users to fine-tune LLMs using the state-of-the-art open source stack 🚀🚀.

Fine-tuning with large language models helps tailor or constrain the LLM's output to a select downstream task's format. The more specific the format requirements of your task, the more beneficial fine-tuning will be.

# Features <img align="center" width="30" height="29" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOTBqaWNrcGxnaTdzMGRzNTN0bGI2d3A4YWkxajhsb2F5MW84Z2dxaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26tOZ42Mg6pbTUPHW/giphy.gif">

- **LLM Fine-Tuning**: Fine-tunes LLMs using HF's `Trainer` for custom datasets stored in the [Hub](https://huggingface.co/datasets).
- **Bits and Bytes**: Loads model with 4-bit quantization.
- **PEFT**: Uses [LoRA](https://arxiv.org/pdf/2106.09685.pdf) under the hood, a popular and lightweight training technique that significantly reduces the number of trainable parameters. We combine 4-bit quantization lowering the barrier for the amount of memory required during training ([QLoRA](https://arxiv.org/abs/2305.14314)).
- **Dataset Preprocessing**: Converts dataset into a prompt template for fine-tuning.
- **Weights & Biases Integration**: Track and log your experiments using wandb.

TODO

> - Table storing hyperparameter configurations for select training job environments (i.e. depending on dataset size, model type/size and amount/type of compute).
> - Model eval functionality post-training.
> - add full list of training args to yaml.
> - Provide inference snippet for testing after training.
> - Fully Sharded Data Parallel (FSDP): Utilizes efficient training across distributed systems.

# Install <img align="center" width="30" height="29" src="https://media.giphy.com/media/sULKEgDMX8LcI/giphy.gif">

```
pip install tllm
```

# Dataset Format <img align="center" width="30" height="29" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbjJmZmEyZ2hsZ2Jyd3c2cDRweWt2dTFyOWJybHp0YTFvc2Q0ZGp1bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4ZkydYFRV6CYprhvF7/giphy.gif">

The trainer expects to ingest a `train` and `validation` split from your dataset prior to training with a specific format. More specificaly, your custom dataset will require `schema` , `input` , and `output` headers. Here is an [example](https://huggingface.co/datasets/zeroshot/text-2-cypher) of a placeholder dataset, designed to demonstrate the expected format.

The following tasks are supported for downstream text generation: 

- `text2sql` : generates SQL query from question in natural language
- `text2cypher` : generates Cypher query from question in natural language
- `input2output` : custom task for generating custom output. If this is selected, you must include a `context` parameter in the Trainer constructor. This context provides guidance to the LLM on interpreting the input and output texts. For instance, if your task is to summarize news articles, the context would be defined as: "Given a news article, construct a summary paragraph ..." etc.

Prompt templates for supported tasks can be found in the [PromptHandler](https://github.com/InquestGeronimo/tllm/blob/main/tllm/utils.py) class.

# Start Training <img align="center" width="30" height="29" src="https://media.giphy.com/media/QLcCBdBemDIqpbK6jA/giphy.gif">

To start training, the only requirements are a `project name`, `task`, your Hugging Face `model`/`dataset` stubs and the path to your YAML `config_file`. This file includes the essential tokenizer, LoRA and training arguments for fine-tuning. Before beginning the training process, ensure you download the YAML file from this repository using either the curl or wget commands to access its contents. As previously stated, if your `task` is **input2output**, you'll need to add an additional `context` parameter to the constructor.

```bash
curl -o config.yml https://raw.githubusercontent.com/InquestGeronimo/tllm/main/config.yml
```

```bash
wget -O config.yml https://raw.githubusercontent.com/InquestGeronimo/tllm/main/config.yml
```

Run the trainer:

```py
from tllm import Trainer

tllm = Trainer(
    project_name="tllm-training-run1",
    task="text2cypher", # tex2sql or input2output
    model_id="codellama/CodeLlama-7b-Instruct-hf",
    dataset_id="zeroshot/text-2-cypher",
    config_file="path/to/config.yml"
)

tllm.train()
```
After training completes, the adapter will be saved in your output directory. The pre-trained model will not be saved.

# HyperParameter Knowledge <img align="center" width="30" height="29" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXV1bWFyMWxkY3JocjE1ZDMxMWZ5OHZtejFkbHpuZXdveTV3Z3BiciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bGgsc5mWoryfgKBx1u/giphy.gif">


Configuring hyperparameters and LoRA settings can be a complex task, especially for those new to AI engineering. Our repository is designed to lessen the dependence on hyperparameters. We hope to establish a set of baseline hyperparameters known to yield good results on specific tasks, and we will save them to this repository, thereby streamlining the fine-tuning process for users. However, having a thorough understanding of these hyperparameters is still advantageous, particularly if you intend to modify them yourself.

Three key factors affect hyperparameters during training:

1. The type and size of the model.
2. The type and quantity of hardware.
3. The size of the dataset.

For accessing tllm's hyperparameters, you can refer to the [config_file](https://github.com/InquestGeronimo/tllm/blob/main/tllm/config.yml). Present parameters in the serve as placeholders.

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