import torch
from peft import PeftModel


class InferenceHandler:
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