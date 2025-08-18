from pathlib import Path
from datatypes import RelevanceAlignment, RelevanceSimilarity, RelevanceEvaluationReport
from prompts import format_relevance_alignment_system_prompt, format_relevance_similarity_system_prompt, format_relevance_user_prompt
from ai_provider import call_gemini, get_ai_client
import json
import os
import numpy as np

from dotenv import load_dotenv
load_dotenv()

class Dataloader():
    def __init__(self, question_data_path: Path):
        if os.path.exists(question_data_path):
            with open(question_data_path,"r") as f:
                self.dataset = json.load(f)
        else:
            raise ValueError("Similar Questions Data path invalid.")
        
        print(f"========= Dataset loaded from {question_data_path}. =========")
        
    def get_dataset(self):
        return self.dataset
    
    def get_random_subset(self, size):
        return np.random.choice(self.dataset,int(size),replace=False).tolist()
           
class RelevanceEvaluator():
    def __init__(self, similar_questions_data):
        self.dataset = similar_questions_data
        
    async def evaluate(self):
        print("========= Starting Relevance Evaluation =========")
        result = []
        for data in self.dataset:
            subject = data['subject']
            question_id = data['question_id']
            main_question = data['question_text']
            similar_questions_array = data['similar_questions']
            
            # check similarity first 
            system_prompt = format_relevance_similarity_system_prompt(subject=subject)
            user_prompt = format_relevance_user_prompt(main_question, similar_questions_array, False)
            relevance_similarity: RelevanceSimilarity = await call_gemini(system_prompt, user_prompt, RelevanceSimilarity)
            
            # check alignment then
            system_prompt = format_relevance_alignment_system_prompt(subject=subject)
            user_prompt = format_relevance_user_prompt(main_question, similar_questions_array, True)
            relevance_alignment: RelevanceAlignment = await call_gemini(system_prompt, user_prompt, RelevanceAlignment)
            
            final_eval = RelevanceEvaluationReport(
                question_id=question_id,
                similarity=relevance_similarity,
                alignment=relevance_alignment
            )
            result.append(final_eval.model_dump(mode="json"))
            
        # can think of some better way of saving checkpoint
        with open("relevance_Eval.json","w") as f:
            json.dump(result, f, ensure_ascii=False)
            
        print("========= Relevance Evaluation Complete, check relevance_eval file for full report. =========")
        
class SolutionBuilder():
    pass
      
async def main():
    dataloader = Dataloader('similar_question_data.json')
    dataset = dataloader.get_random_subset(2)
    rel_eval = RelevanceEvaluator(dataset)
    await rel_eval.evaluate()

import asyncio 
if __name__ == "__main__":
    asyncio.run(main())
