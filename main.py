import os
from pathlib import Path
from core.relevance_evaluator import RelevanceEvaluator
from core.solution_builder import SolutionBuilder
from core.comparative_analyzer import ComparativeAnalyzer
from helpers.dataloader import Dataloader

from dotenv import load_dotenv
load_dotenv()
      
async def main():
    reports_dir = Path("./reports")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        
    dataloader = Dataloader('similar_question_data.json')
    dataset = dataloader.get_random_subset(2)
    
    rel_eval = RelevanceEvaluator(similar_questions_data=dataset, reports_dir=reports_dir)
    await rel_eval.evaluate()
    
    solution_builder = SolutionBuilder(similar_question_data=dataset, reports_dir=reports_dir)
    solutions_with_similar, solutions_without_similar = await solution_builder.build_solution()
    
    comparative_analyzer = ComparativeAnalyzer(
        similar_question_data=dataset,
        generated_solutions_w_similar=solutions_with_similar,
        generated_solutions_wo_similar=solutions_without_similar,
        reports_dir=reports_dir
    )
    await comparative_analyzer.analyze()
    await comparative_analyzer.generate_insights()
    
import asyncio 
if __name__ == "__main__":
    asyncio.run(main())
