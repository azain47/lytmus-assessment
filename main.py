from pathlib import Path
from datatypes import RelevanceAlignment, RelevanceSimilarity, RelevanceEvaluationReport
from prompts import format_relevance_alignment_system_prompt, format_relevance_alignment_system_prompt, format_relevance_user_prompt
from ai_provider import call_gemini, get_ai_client
import json
import os
from dotenv import load_dotenv
load_dotenv()

class Dataloader():
    def __init__(self, question_data_path: Path):
        if os.path.exists(question_data_path):
            with open(question_data_path,"r") as f:
                self.dataset = json.load(f)
        else:
            raise ValueError("Similar Questions Data path invalid.")
        return self.dataset

class RelevanceEvaluator():
    def __init__(self, similar_questions_data):
        self.dataset = similar_questions_data
    
    def evaluate(self):
        pass
        
def main():
    print("Hello from lytmus-assessment!")


if __name__ == "__main__":
    main()
