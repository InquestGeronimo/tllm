import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
import os
import transformers
from datetime import datetime
from peft import PeftModel


class LLMTrainer:
    """
    A class for fine-tuning and generating text with the Llama 2 7B model.

    Args:
        base_model_id (str): The identifier for the base Llama 2 7B model from Hugging Face.
        output_dir (str): The directory where model checkpoints and logs will be saved.
        train_dataset_path (str): The stub of the training dataset on HF Hub.
        eval_dataset_path (str): The stub of the evaluation dataset on HF Hub.
    """

    def __init__(
        self, base_model_id, output_dir, train_dataset_path, eval_dataset_path=None
    ):
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.train_dataset_path = train_dataset_path
        self.eval_dataset_path = eval_dataset_path

    def load_datasets(self):
        """
        Load training and evaluation datasets.
        """
        train_dataset = load_dataset(path=self.train_dataset_path, split="train"
        )
        eval_dataset = load_dataset(path=self.eval_dataset_path, split="eval"
        )
        return train_dataset, eval_dataset

    def prepare_model(self):
        """
        Prepare the model for fine-tuning with QLoRA.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = prepare_model_for_kbit_training(base_model)

        config = LoraConfig(
            r=32,
            lora_alpha=64,
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
        return model, tokenizer

    def setup_training(self, model, tokenizer, train_dataset, eval_dataset=None):
        """
        Set up the training configuration using Transformers Trainer.
        """
        if torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        project = "journal-finetune"
        run_name = "llama2-7b-" + project

        tokenizer.pad_token = tokenizer.eos_token

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=TrainingArguments(
                output_dir=self.output_dir,
                warmup_steps=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                max_steps=500,
                learning_rate=2.5e-5,
                bf16=True,
                optim="paged_adamw_8bit",
                logging_dir="./logs",
                save_strategy="steps",
                save_steps=50,
                evaluation_strategy="steps",
                eval_steps=50,
                do_eval=True,
                report_to="wandb",
                run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
            ),
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
        Train the fine-tuned model.
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
