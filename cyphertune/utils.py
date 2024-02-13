from typing import Dict, Any, Optional, Tuple
import yaml

from .models import LoraConfig, TrainerConfig


class PromptHandler:
    
    @staticmethod
    def set_prompt(doc: Dict[str, Any]) -> str:
        """Constructs a cypher statement based on the given document.

        Args:
            doc (Dict[str, Any]): A dictionary containing schema, input, and output keys.

        Returns:
            str: A formatted cypher statement prompt.
        """
        prompt = f"""Given a sentence in natural language, construct a cypher statement in order to extract information from a knowledge graph.
            The graph will have the following schema:
            {doc["schema"]}

            ### sentence:
            {doc["input"]}
            
            ### cypher statement:
            {doc["output"]}
            """
        return prompt
    
class YamlFileManager:
    @staticmethod
    def parse_yaml_file(yaml_file_path: str) -> Optional[Tuple[LoraConfig, TrainerConfig]]:
        """Parse a YAML file and return a tuple of LoraConfig and TrainerConfig objects.

        Args:
            yaml_file_path (str): The path to the YAML file.

        Returns:
            Optional[Tuple[LoraConfig, TrainerConfig]]: A tuple containing LoraConfig and TrainerConfig objects, or None on error.
        """
        try:
            with open(yaml_file_path, "r") as yaml_file:
                yaml_content = yaml.safe_load(yaml_file)

            lora_config = LoraConfig(**yaml_content["LoraConfig"])
            trainer_config = TrainerConfig(**yaml_content["TrainerConfig"])

            return lora_config, trainer_config

        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error parsing YAML file: {e}")
            return None
