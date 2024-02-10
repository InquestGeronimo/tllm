import os
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
import transformers
from peft import PeftModel

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from .utils import PromptHandler

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

class LLMTrainer:
    """
    A class for fine-tuning and generating text with the Llama 2 7B model.

    Args:
        project_name (str): Name of the project for organization and tracking.
        model_id (str): Identifier for the base model from Hugging Face.
        dataset_id (str): Identifier for the dataset used for training and evaluation.
    """

    def __init__(self, project_name, model_id, dataset_id):
        self.project_name = project_name
        self.model_id = model_id
        self.dataset_id = dataset_id
        
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.args = TrainingArguments(
            output_dir="./output",
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            max_steps=10,
            learning_rate=2.5e-5,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=10,
            evaluation_strategy="steps",
            eval_steps=2,
            do_eval=True,
            report_to="wandb",
            remove_unused_columns=True,
            run_name=f"{self.project_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        )

    def load_datasets(self):
        """
        Load the training and evaluation datasets from the specified dataset ID.
        """
        train_dataset = load_dataset(path=self.dataset_id, split="train")
        eval_dataset = load_dataset(path=self.dataset_id, split="validation")
        
        return train_dataset, eval_dataset

    def load_model_and_tokenizer(self):
        """
        Load the base model and tokenizer for the Llama 2 7B model from Hugging Face.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, quantization_config=self.bnb_config, trust_remote_code=True,
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
                truncation=True,
                max_length=340,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            
            return result
        
        def generate_and_tokenize_prompt(doc):
            return tokenize(PromptHandler.set_prompt(doc))
        
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
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
        model = accelerator.prepare_model(model)
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
        wandb.login()
        wandb_project = "tllm-finetune"
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project

        return trainer
    
    def train_model(self, trainer):
        """
        Run the fine-tuning training on LLM.

        Args:
            trainer: The trainer object configured for model training.
        """
        trainer.train()

    def load_finetuned_model(self, checkpoint_path):
        """
        Load the fine-tuned model checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint directory.
        """
        self.ft_model = PeftModel.from_pretrained(self.base_model, checkpoint_path)

    def generate_text(self, eval_prompt):
        """
        Generate text using the fine-tuned model.

        Args:
            eval_prompt (str): The prompt for text generation.

        Returns: 
            str: The generated text.
        """
        model_input = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        self.ft_model.eval()
        with torch.no_grad():
            return self.tokenizer.decode(
                self.ft_model.generate(**model_input, max_new_tokens=300)[0],
                skip_special_tokens=True,
            )
