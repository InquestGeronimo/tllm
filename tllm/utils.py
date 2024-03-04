from typing import Dict, Any, Optional, Tuple
import yaml

from .models import TokenizerConfig, LoraConfig, TrainerConfig


class PromptHandler:
    @staticmethod
    def set_sql_prompt(doc: Dict[str, Any]) -> str:
        """
        Constructs a SQL query prompt based on the given document.

        Args:
            doc (Dict[str, Any]): A dictionary containing the input question, context, and optional output SQL query.

        Returns:
            str: A formatted text-to-SQL query prompt.
        """
        prompt = f"""Translate the following question into a SQL query, considering the specified database schema.
            The database has the following schema or context:
            {doc["schema"]}

            ### question:
            {doc["input"]}

            ### SQL query:
            {doc["output"]}
            """
        return prompt

    @staticmethod
    def set_cypher_prompt(doc: Dict[str, Any]) -> str:
        """Constructs a cypher statement based on the given document.

        Args:
            doc (Dict[str, Any]): A dictionary containing schema, input, and output keys.

        Returns:
            str: A formatted cypher statement prompt.
        """
        prompt = f"""Given a question in natural language, construct a cypher query in order to extract information from a knowledge graph.
            The graph will have the following schema:
            {doc["schema"]}

            ### question:
            {doc["input"]}
            
            ### cypher query:
            {doc["output"]}
            """
        return prompt

    @staticmethod
    def set_custom_prompt(doc: Dict[str, Any], context: str) -> str:
        """Constructs a custom output based on the given input.

        Args:
            doc (Dict[str, Any]): A dictionary containing schema, input, and output keys.
            context str: text describing LLM instruction to handle input to output generation.
        Returns:
            str: A formatted prompt.
        """
        prompt = f"""{context}

            ### question:
            {doc["input"]}
            
            ### cypher query:
            {doc["output"]}
            """
        return prompt


class YamlFileManager:
    @staticmethod
    def parse_yaml_file(
        yaml_file_path: str,
    ) -> Optional[Tuple[TokenizerConfig, LoraConfig, TrainerConfig]]:
        """Parse a YAML file and return a tuple of TokenizerConfig, LoraConfig and TrainerConfig objects.

        Args:
            yaml_file_path (str): The path to the YAML file.

        Returns:
            Optional[Tuple[TokenizerConfig, LoraConfig, TrainerConfig]]: A tuple containing TokenizerConfig, LoraConfig
            and TrainerConfig objects, or None on error.
        """
        try:
            with open(yaml_file_path, "r") as yaml_file:
                yaml_content = yaml.safe_load(yaml_file)

            token_config = TokenizerConfig(**yaml_content["TokenizerConfig"])
            lora_config = LoraConfig(**yaml_content["LoraConfig"])
            trainer_config = TrainerConfig(**yaml_content["TrainerConfig"])

            return token_config, lora_config, trainer_config

        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error parsing YAML file: {e}")
            return None


class ErrorHandler(Exception):
    """
    Custom error handler for specific ValueError instances.
    """

    context_error = "A 'context' parameter must be provided to the Trainer constructor for the 'input2output' task."

    @staticmethod
    def handle_task(task):
        return f"Unsupported task: {task}"
