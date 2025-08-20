import json
import os
from tqdm import tqdm
from core.prompts import format_solution_builder_prompt
from core.datatypes import Solution, GeneratedSolution
from helpers.ai_provider import call_gemini

class SolutionBuilder():
    def __init__(self, similar_question_data, reports_dir):
        self.dataset = similar_question_data
        self.reports_dir = reports_dir
        
    async def build_solution(self):
        print("========= Starting Solution Building =========")
        solutions_arr = []
        solutions_with_similar_arr = []
        
        for data in tqdm(self.dataset, desc="Building Solutions", unit="Question"):
            subject = data['subject']
            question_id = data['question_id']
            main_question = data['question_text']
            similar_questions_array = data['similar_questions']
            
            user_prompt = format_solution_builder_prompt(subject, main_question, with_similar=False)
            user_prompt_with_similar = format_solution_builder_prompt(subject, main_question, similar_questions_array, with_similar=True)

            response = await call_gemini(user_prompt, response_schema=Solution, temperature=0.1)
            solution = GeneratedSolution(
                **response.model_dump(),
                question_id=question_id,
                was_solved_with_similar_questions=False
            )
            solutions_arr.append(solution.model_dump(mode="json"))
        
            response = await call_gemini(user_prompt_with_similar, response_schema=Solution, temperature=0.1)
            solution_with_similar = GeneratedSolution(
                **response.model_dump(),
                question_id=question_id,
                was_solved_with_similar_questions=True
            )
            solutions_with_similar_arr.append(solution_with_similar.model_dump(mode="json"))    
            
        # can think of some better way of saving checkpoint
        with open(os.path.join(self.reports_dir,"generated_solutions_wo_similar.json"),"w") as f:
            json.dump(solutions_arr, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.reports_dir,"generated_solutions_w_similar.json"),"w") as f:
            json.dump(solutions_with_similar_arr, f, indent=2, ensure_ascii=False)
                
        print("========= Solution Building Complete, check generated_solutions files for solutions. =========")
        return solutions_arr, solutions_with_similar_arr
