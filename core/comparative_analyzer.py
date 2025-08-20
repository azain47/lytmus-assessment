import json
import os
from tqdm import tqdm
from core.prompts import format_solution_comparison_system_prompt_array, format_solution_comparison_user_prompt, format_solution_performance_analysis_prompt, format_insight_generation_prompt
from core.datatypes import MetricEvaluation, SolutionPerformanceAnalysis, InsightReport
from helpers.ai_provider import call_gemini
from helpers.utils import convert_list_to_dict_with_key
from core.datatypes import WinnerSolution

class ComparativeAnalyzer():
    def __init__(self,similar_question_data, generated_solutions_w_similar, generated_solutions_wo_similar, reports_dir):
        # convert arrays to dict with question_id as key for easier retrieval
        self.dataset = convert_list_to_dict_with_key(similar_question_data, 'question_id')
        self.solutions_with_similar = convert_list_to_dict_with_key(generated_solutions_w_similar, 'question_id')
        self.solutions_without_similar = convert_list_to_dict_with_key(generated_solutions_wo_similar, 'question_id')
        self.reports_dir = reports_dir
        
        # define metrics, this is dynamic, can be extended or modified.
        self.solution_comparison_metrics = {
            "CORRECTNESS": "- Is the final answer correct?\n- Is the application of formulas, principles, and calculations accurate?",
            "COMPLETENESS": "- Are all necessary steps included to logically reach the conclusion?\n-Are there any 'magic steps' or unexplained logical jumps?",
            "CLARITY": "- Is the explanation for each step clear, concise, and easy for a student to follow?\n- Is the overall structure logical?"
        }
        
        # basically create a unified dataset with both solutions 
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
            
            # was initially thinking of flipping sol a and sol b to lessen bias
            user_prompt = format_solution_comparison_user_prompt(main_question, solution_a, solution_b)
            
            analysis_report = {"question_id": ques_id}    
            # just a hack to show the exact metric being evaluated in tqdm
            progress = tqdm(system_prompts.items(), total=len(system_prompts), desc="Evaluating Metrics", unit="Metric")
            for metric, system_prompt in progress:
                progress.set_postfix_str(f"Evaluating {metric}")
                metric_eval : MetricEvaluation = await call_gemini(user_prompt, system_prompt, MetricEvaluation, temperature=0.1)
                analysis_report[metric.lower()] = metric_eval.model_dump(mode="json")
            
            analysis_arr.append(analysis_report)
            
        # can think of some better way of saving checkpoint
        with open(os.path.join(self.reports_dir,"comparative_analysis_report.json"),"w") as f:
            json.dump(analysis_arr, f, indent=2, ensure_ascii=False)
        
        self.analysed_dataset = analysis_arr    
        print("========= Comparative Analysis Complete, check comparative_analysis_report file for full report. =========")
    
    async def generate_insights(self):
        print("========= Starting Insight Generation =========")
        processed_analysis = []
        self.analysed_dataset = json.load(open(os.path.join(self.reports_dir,"comparative_analysis_report.json"), "r"))
        for analysis in self.analysed_dataset:
            ques_id = analysis['question_id']
            
            scores = {}
            for metric,_ in self.solution_comparison_metrics.items():
                metric_normalized = metric.lower()
                winner = analysis[metric_normalized]['winner']
                margin_winner = analysis[metric_normalized]['margin_of_winning']
                
                metric_score = 0.0
                # Positive score for solution A, i.e. Sol generated with similar questions.
                if winner == WinnerSolution.SOLUTION_A.value:
                    metric_score = margin_winner
                # Negative score for solution B, i.e. Sol generated without similar questions.
                elif winner == WinnerSolution.SOLUTION_B.value:
                    metric_score = -margin_winner
                
                scores[metric_normalized] = metric_score
            # average of all metric scores 
            average_score = sum(scores.values()) / len(scores)
            
            processed_analysis.append({
                "question_id": ques_id,
                "average_score": average_score,
                "full_analysis": analysis,
                "original_question_data": self.dataset.get(ques_id)
            })
        
        with open(os.path.join(self.reports_dir,"full_analysis_report.json"), "w") as f:
                json.dump(processed_analysis, f, indent=2, ensure_ascii=False)
        print("========= Full Analysis Report Generated. Check full_analysis_report.json for details. =========")   
            
        # Determine when Solution A (Generated WITH similar questions) won and lost.
        threshold = 0.2
        strong_wins = [w for w in processed_analysis if w['average_score'] > threshold]
        strong_losses = [l for l in processed_analysis if l['average_score'] < -threshold]
        
        # couldnt pass the threshold, so no strong wins or losses
        if strong_losses == [] and strong_wins == []:
            print("No strong wins or losses found. Here is the full analysis report:")
            for analysis in processed_analysis:
                print(f"\nQuestion ID: {analysis['question_id']}")
                print(f"Average Score: {analysis['average_score']}")
                print(f"Full Analysis: {json.dumps(analysis['full_analysis'], indent=2)}")
            print("========= Insight Generation Complete. No strong wins or losses found. =========")
            return
        
        win_analysis_arr = []
        loss_analysis_arr = []
        
        for wins in tqdm(strong_wins[:min(len(strong_wins), 4)], desc="Analyzing wins", unit="WINS"):
            original_data = wins['original_question_data']
            user_prompt = format_solution_performance_analysis_prompt(original_data['subject'], "WIN", wins['average_score'], original_data['question_text'], original_data['similar_questions'], wins['full_analysis'])
            
            response: SolutionPerformanceAnalysis = await call_gemini(user_prompt, response_schema=SolutionPerformanceAnalysis, temperature=0.1)
            win_analysis = response.model_dump(mode="json")
            win_analysis['question_id'] = original_data['question_id']
            win_analysis_arr.append(win_analysis)
            
        for loss in tqdm(strong_losses[:min(len(strong_losses), 4)], desc="Analyzing losses", unit="LOSSES"):
            original_data = loss['original_question_data']
            user_prompt = format_solution_performance_analysis_prompt(original_data['subject'], "LOSS", loss['average_score'], original_data['question_text'], original_data['similar_questions'], loss['full_analysis'])
            
            response: SolutionPerformanceAnalysis = await call_gemini(user_prompt, response_schema=SolutionPerformanceAnalysis, temperature=0.1)
            loss_analysis = response.model_dump(mode="json")
            loss_analysis['question_id'] = original_data['question_id']
            loss_analysis_arr.append(loss_analysis)
            
        insight_user_prompt = format_insight_generation_prompt(win_analysis_arr, loss_analysis_arr)
        
        final_report: InsightReport = await call_gemini(
            user_message=insight_user_prompt,
            response_schema=InsightReport
        )
             
        for idx,insight in enumerate(final_report.insights):
            print(f"\nRECOMMENDATION {idx + 1}: {insight.recommendation}")
            print(f"Reasoning: {insight.reasoning}\n")

        print("========= Insight Generation Complete. Check the console for insights. =========")
        
        with open(os.path.join(self.reports_dir,"insight_report.json"), "w") as f:
            json.dump(final_report.model_dump(mode="json"), f, indent=2, ensure_ascii=False)