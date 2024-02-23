import os
import torch
import datetime
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from .utils import ErrorHandler as fixer, PromptHandler, YamlFileManager as manager
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)


class Trainer:
    """
    A class for fine-tuning an LLM using QLoRA.

    Args:
        project_name: Name of the project for organization and tracking.
        model_id: Identifier for the base model from Hugging Face.
        dataset_id: Identifier for the dataset used for training and evaluation.
        config_file: path to YAML config file.
        context: The context required for 'input2output' task.
    """

    def __init__(
        self,
        project_name: str,
        task: str,
        model_id: str,
        dataset_id: str,
        config_file: str,
        context: str = None,
    ):
        if task == "input2output" and context is None:
            raise ValueError(fixer.context_error)

        self.project_name = project_name
        self.task = task
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.context = context

        # Parse configuration file
        token_config, lora_config, trainer_config = manager.parse_yaml_file(config_file)

        # Use the parsed configurations
        self.token_config = token_config
        self.lora_config = LoraConfig(**lora_config.model_dump())
        self.args = TrainingArguments(**trainer_config.model_dump())

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.wandb_project = f"tllm-finetune_{current_time}"

    def setup_wandb(self):
        """
        Set up the Weights & Biases environment for logging and tracking.
        """
        wandb.login()
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project

    def load_datasets(self):
        """
        Load the training and evaluation datasets from the specified dataset ID.
        """
        train_dataset = load_dataset(path=self.dataset_id, split="train")
        eval_dataset = load_dataset(path=self.dataset_id, split="validation")

        return train_dataset, eval_dataset

    def load_model_and_tokenizer(self):
        """
        Load the base model and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.bnb_config,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )

        return model, tokenizer

    def create_prompts_from_datasets(self, tokenizer, train_data, eval_data):
        """
        Create and tokenize prompts from the datasets.

        Args:
            tokenizer: The tokenizer to be used for tokenizing the prompts.
            train_data: The training dataset.
            eval_data: The evaluation dataset.

        Returns:
            Tuple of tokenized training and evaluation datasets.
        """
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize(prompt):
            result = tokenizer(
                prompt,
                max_length=self.token_config.max_length,
                truncation=self.token_config.truncation,
                padding=self.token_config.padding,
            )
            result["labels"] = result["input_ids"].copy()

            return result

        def generate_and_tokenize_prompt(doc):
            if self.task == "text2cypher":
                prompt = PromptHandler.set_cypher_prompt(doc)
            elif self.task == "text2sql":
                prompt = PromptHandler.set_sql_prompt(doc)
            elif self.task == "input2output":
                prompt = PromptHandler.set_custom_prompt(doc, self.context)
            else:
                raise ValueError(fixer.handle_task(self.task))

            return tokenize(prompt)

        tokenized_train_dataset = train_data.map(generate_and_tokenize_prompt)
        tokenized_eval_dataset = eval_data.map(generate_and_tokenize_prompt)

        return tokenized_train_dataset, tokenized_eval_dataset

    def configure_lora(self, model):
        """
        Configure the model with Lora settings for fine-tuning.

        Args:
            model: The model to be configured.

        Returns:
            The configured model.
        """
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)
        model = self.accelerator.prepare_model(model)
        return model

    def configure_training(self, model, tokenizer, train_dataset, eval_dataset):
        """
        Set up the training configuration using Transformers Trainer.

        Args:
            model: The model to be trained.
            tokenizer: The tokenizer to be used during training.
            train_dataset: The dataset to be used for training.
            eval_dataset: The dataset to be used for evaluation.

        Returns:
            Configured trainer object.
        """
        if torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True
        tokenizer.pad_token = tokenizer.eos_token

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=self.args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        model.config.use_cache = False

        return trainer

    def train(self):
        """
        Prepares datasets, model, tokenizer, and configurations, and then starts the training process.
        """

        print("Preparing your training job...")

        self.setup_wandb()
        train_data, eval_data = self.load_datasets()
        model, tokenizer = self.load_model_and_tokenizer()
        train_data, eval_data = self.create_prompts_from_datasets(
            tokenizer, train_data, eval_data
        )
        model = self.configure_lora(model)
        trainer = self.configure_training(model, tokenizer, train_data, eval_data)
        trainer.train()
