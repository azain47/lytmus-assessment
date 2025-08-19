from pathlib import Path
from datatypes import RelevanceAlignment, RelevanceSimilarity, RelevanceEvaluationReport, GeneratedSolution
from prompts import format_relevance_alignment_system_prompt, format_relevance_similarity_system_prompt, format_relevance_user_prompt, format_solution_builder_prompt
from ai_provider import call_gemini, get_ai_client
import json
import os
import numpy as np
from tqdm import tqdm

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
        results_arr = []
        for data in tqdm(self.dataset, desc="Evaluating Relevance", unit="Question"):
            subject = data['subject']
            question_id = data['question_id']
            main_question = data['question_text']
            similar_questions_array = data['similar_questions']
            
            # check similarity first 
            system_prompt = format_relevance_similarity_system_prompt(subject=subject)
            user_prompt = format_relevance_user_prompt(main_question, similar_questions_array, False)
            relevance_similarity: RelevanceSimilarity = await call_gemini(user_prompt, system_prompt, RelevanceSimilarity, temperature=0.3)
            
            # check alignment then
            system_prompt = format_relevance_alignment_system_prompt(subject=subject)
            user_prompt = format_relevance_user_prompt(main_question, similar_questions_array, True)
            relevance_alignment: RelevanceAlignment = await call_gemini(user_prompt, system_prompt, RelevanceAlignment, temperature=0.3)
            
            final_eval = RelevanceEvaluationReport(
                question_id=question_id,
                similarity=relevance_similarity,
                alignment=relevance_alignment
            )
            results_arr.append(final_eval.model_dump(mode="json"))
            
        # can think of some better way of saving checkpoint
        with open("relevance_Eval.json","w") as f:
            json.dump(results_arr, f, indent=2, ensure_ascii=False)
            
        print("========= Relevance Evaluation Complete, check relevance_eval file for full report. =========")
        
class SolutionBuilder():
    def __init__(self, similar_question_data):
        self.dataset = similar_question_data
    
    async def build_solution(self):
        solutions_arr = []
        solutions_with_similar_arr = []
        print("========= Starting Solution Building =========")
        for data in tqdm(self.dataset, desc="Building Solutions", unit="Question"):
            subject = data['subject']
            question_id = data['question_id']
            main_question = data['question_text']
            similar_questions_array = data['similar_questions']
            
            user_prompt = format_solution_builder_prompt(subject, main_question, with_similar=False)
            user_prompt_with_similar = format_solution_builder_prompt(subject, main_question, similar_questions_array, with_similar=True)

            response = await call_gemini(user_prompt, temperature=0.2)
            solution = GeneratedSolution(
                question_id=question_id,
                generated_solution=response,
                was_solved_with_similar_questions=False
            )
            solutions_arr.append(solution.model_dump(mode="json"))
        
            response = await call_gemini(user_prompt_with_similar, temperature=0.2)
            solution_with_similar = GeneratedSolution(
                question_id=question_id,
                generated_solution=response,
                was_solved_with_similar_questions=True
            )
            solutions_with_similar_arr.append(solution_with_similar.model_dump(mode="json"))    
            
        # can think of some better way of saving checkpoint
        with open("generated_solutions_wo_similar.json","w") as f:
            json.dump(solutions_arr, f, indent=2, ensure_ascii=False)
        with open("generated_solutions_w_similar.json","w") as f:
            json.dump(solutions_with_similar_arr, f, indent=2, ensure_ascii=False)
            
        print("========= Solution Building Complete, check generated_solutions files for solutions. =========")
      
async def main():
    dataloader = Dataloader('similar_question_data.json')
    dataset = dataloader.get_random_subset(2)
    rel_eval = RelevanceEvaluator(dataset)
    # await rel_eval.evaluate()
    solution_builder = SolutionBuilder(dataset)
    await solution_builder.build_solution()

import asyncio 
if __name__ == "__main__":
    asyncio.run(main())
