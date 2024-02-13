from pydantic import BaseModel
from typing import List

class LoraConfig(BaseModel):
    r: int
    lora_alpha: int
    target_modules: List[str]
    bias: str
    lora_dropout: float
    task_type: str


class TrainerConfig(BaseModel):
    output_dir: str
    warmup_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_steps: int
    learning_rate: float
    bf16: bool
    optim: str
    logging_dir: str
    save_strategy: str
    save_steps: int
    evaluation_strategy: str
    eval_steps: int
    do_eval: bool
    report_to: str
    remove_unused_columns: bool
    run_name: str