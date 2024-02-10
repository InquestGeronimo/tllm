
class PromptHandler:
    
    @staticmethod
    def set_prompt(doc):
        prompt =f"""Given a sentence in natural language, construct a cypher statement in order to extract information from a knowledge graph.
            The graph will have the following schema:
            {doc["schema"]}

            ### cypher statement:
            {doc["output"]}

            ### sentence:
            {doc["input"]}
            """
        return prompt