import json
import os
from tqdm import tqdm
from core.prompts import format_relevance_similarity_system_prompt, format_relevance_alignment_system_prompt, format_relevance_user_prompt
from core.datatypes import RelevanceSimilarity, RelevanceAlignment, RelevanceEvaluationReport
from helpers.ai_provider import call_gemini

class RelevanceEvaluator():
    def __init__(self, similar_questions_data, reports_dir):
        self.dataset = similar_questions_data
        self.reports_dir = reports_dir   
             
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
        with open(os.path.join(self.reports_dir, "relevance_eval_report.json"),"w") as f:
            json.dump(results_arr, f, indent=2, ensure_ascii=False)
            
        print("========= Relevance Evaluation Complete, check relevance_eval file for full report. =========")