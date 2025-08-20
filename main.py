from pathlib import Path
from datatypes import RelevanceAlignment, RelevanceSimilarity, RelevanceEvaluationReport, GeneratedSolution, MetricEvaluation, Solution
from prompts import format_relevance_alignment_system_prompt, format_relevance_similarity_system_prompt, format_relevance_user_prompt, format_solution_builder_prompt, format_solution_comparison_system_prompt_array, format_solution_comparison_user_prompt
from ai_provider import call_gemini, get_ai_client
from utils import convert_list_to_dict_with_key
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

            response = await call_gemini(user_prompt, response_schema=Solution, temperature=0.2)
            solution = GeneratedSolution(
                **response.model_dump(),
                question_id=question_id,
                was_solved_with_similar_questions=False
            )
            solutions_arr.append(solution.model_dump(mode="json"))
        
            response = await call_gemini(user_prompt_with_similar, response_schema=Solution, temperature=0.2)
            solution_with_similar = GeneratedSolution(
                **response.model_dump(),
                question_id=question_id,
                was_solved_with_similar_questions=True
            )
            solutions_with_similar_arr.append(solution_with_similar.model_dump(mode="json"))    
            
        # can think of some better way of saving checkpoint
        with open("generated_solutions_wo_similar.json","w") as f:
            json.dump(solutions_arr, f, indent=2, ensure_ascii=False)
        with open("generated_solutions_w_similar.json","w") as f:
            json.dump(solutions_with_similar_arr, f, indent=2, ensure_ascii=False)
                
        print("========= Solution Building Complete, check generated_solutions files for solutions. =========")
        return solutions_arr, solutions_with_similar_arr

class ComparativeAnalyzer():
    def __init__(self,similar_question_data, generated_solutions_w_similar, generated_solutions_wo_similar):
        self.dataset = convert_list_to_dict_with_key(similar_question_data, 'question_id')
        self.solutions_with_similar = convert_list_to_dict_with_key(generated_solutions_w_similar, 'question_id')
        self.solutions_without_similar = convert_list_to_dict_with_key(generated_solutions_wo_similar, 'question_id')
        self.solution_comparison_metrics = {
            "CORRECTNESS": "- Is the final answer correct?\n- Is the application of formulas, principles, and calculations accurate?",
            "COMPLETENESS": "- Are all necessary steps included to logically reach the conclusion?\n-Are there any 'magic steps' or unexplained logical jumps?",
            "CLARITY": "- Is the explanation for each step clear, concise, and easy for a student to follow?\n- Is the overall structure logical?"
        }
        
        for ques_id, data in self.dataset.items():
            data['solution_generated_with_similar'] = self.solutions_with_similar.get(ques_id)['generated_solution']
            data['solution_generated_without_similar'] = self.solutions_without_similar.get(ques_id)['generated_solution']
               
    async def analyze(self):
        print("========= Starting Comparative Analysis =========")
        analysis_arr = []
        
        for ques_id, data in tqdm(self.dataset.items(), desc="Analyzing Solutions", unit="Question"):
            subject = data['subject']
            main_question = data['question_text']
            solution_a = data['solution_generated_with_similar']
            solution_b = data['solution_generated_without_similar']
            
            system_prompts = format_solution_comparison_system_prompt_array(subject, self.solution_comparison_metrics)
            user_prompt = format_solution_comparison_user_prompt(main_question, solution_a, solution_b)
            
            analysis_report = {"question_id": ques_id} 
            
            progress = tqdm(system_prompts.items(), total=len(system_prompts), desc="Evaluating Metrics", unit="Metric")
            for metric, system_prompt in progress:
                progress.set_postfix_str(f"Evaluating {metric}")
                metric_eval : MetricEvaluation = await call_gemini(user_prompt, system_prompt, MetricEvaluation, temperature=0.1)
                analysis_report[metric.lower()] = metric_eval.model_dump(mode="json")
            
            analysis_arr.append(analysis_report)
            
        # can think of some better way of saving checkpoint
        with open("comparative_analysis_report.json","w") as f:
            json.dump(analysis_arr, f, indent=2, ensure_ascii=False)
        
        self.analysed_dataset = analysis_arr    
        print("========= Comparative Analysis Complete, check comparative_analysis_report file for full report. =========")
    
    async def generate_insights(self):
        for analysis in self.analysed_dataset:
            pass
    
async def main():
    dataloader = Dataloader('similar_question_data.json')
    dataset = dataloader.get_random_subset(1)
    
    rel_eval = RelevanceEvaluator(dataset)
    await rel_eval.evaluate()
    
    solution_builder = SolutionBuilder(dataset)
    solutions_with_similar, solutions_without_similar = await solution_builder.build_solution()
    
    comparative_analyzer = ComparativeAnalyzer(
        similar_question_data=dataset,
        generated_solutions_w_similar=solutions_with_similar,
        generated_solutions_wo_similar=solutions_without_similar
    )
    await comparative_analyzer.analyze()
    
import asyncio 
if __name__ == "__main__":
    asyncio.run(main())
