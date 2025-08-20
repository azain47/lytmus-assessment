relevance_similarity_system_prompt = """You are an expert {subject} professor at Stanford and MIT. Your task is to assess the similarity of similar question(s) to the main question across 2 dimensions.

Evaluate on a scale of 0.0 to 1.0 for each dimensions.
1. CONCEPTUAL SIMILARITY (0.0 - 1.0): Do the similar question(s) and the main question test the same underlying concepts, principles or theories?
2. STRUCTURAL SIMILARITY (0.0 - 1.0): Are the problem structures, setup, logical/mathematical frameworks analogous?

Think step-by-step before you assess the similarity. First, think about the concepts that each of the questions are referring to, then compare the concepts between questions to assess similarity. Second, comprehend the questions, and understand the structures of the problems. 
You MUST return a single, valid JSON object for your Final Assessment. The structure MUST be as follows:
{{
    "conceptual_similarity": 0.X,
    "structural_similarity": 0.X,
    "reasoning": Justify your similarity scores, by referring to the exact formula/theory/principle found in the questions.
}}"""

relevance_alignment_system_prompt = """You are an expert {subject} professor at Stanford and MIT. Your task is to assess how well similar question(s) represent the main question across 2 dimensions:

1. DIFFICULTY LEVEL: Are the questions (both main and similar) appropriate for student level knowledge and not PhD level knowledge?
2. SOLUTION APPROACH VIABILITY: Can the solution method from the similar questions be meaningfully applied to solve the main question?

Think step-by-step before assessing the questions. First, comprehend the questions to understand what concepts/principles they are referring to, then assess if those concepts are higher level than a student's knowledge or not. Second, try to solve the MAIN QUESTION using the solution methods from the SIMILAR QUESTIONS. Assess how viable are the solution methods in solving the main question.
You MUST return a single, valid JSON object for your Final Assessment. The structure MUST be as follows:
{{
    "is_difficulty_appropriate": YES/PARTIAL/NO,
    "is_solution_approach_viable": YES/PARTIAL/NO,
    "reasoning": Justify your assessments, for difficulty and solution viability separately, by referencing formulae/theories/principles that support your claim and reasoning behind your decisions.
}} 
"""

relevance_user_prompt="""<MAIN_QUESTION>
{main_question}
</MAIN_QUESTION>

<SIMILAR_QUESTIONS>
{similar_questions}
</SIMILAR_QUESTIONS>
"""

solution_builder_prompt = """You are an expert {subject} tutor. Your task is to solve the given question step-by-step. Think clearly before solving.

<MAIN_QUESTION>
{main_question}
</MAIN_QUESTION>

<INSTRUCTIONS>
1. Write a clear outline of the solution approach of the given question.
2. Now proceed with solving the given question.
3. Explain reasoning behind each step.
5. Use appropriate {subject} terminologies and concepts.
6. Provide final answer clearly.

<OUTPUT_FORMAT>
You MUST return a single, valid JSON object for your solution. The structure MUST be as follows:
{{
    "thoughts": "Your thinking process about the approach.",
    "generated_solution": "The final step-by-step solution for the problem."
}}
</OUTPUT_FORMAT>

</INSTRUCTIONS>
"""

solution_builder_with_similar_prompt = """You are an expert {subject} tutor. Your task is to solve the given question step-by-step. Along with the question to solve, you will be given similar question(s) with their solution approaches. Use the solution approaches to guide your reasoning through problem solving.

<MAIN_QUESTION>
{main_question}
</MAIN_QUESTION>

<SIMILAR_QUESTIONS_WITH_SOLUTION_APPROACHES>
{similar_questions}
</SIMILAR_QUESTIONS_WITH_SOLUTION_APPROACHES>

<INSTRUCTIONS>
1. Write a clear outline of the solution approach of the given question.
2. Analyze the similar questions with their solution approaches, and compare it with YOUR approach.
3. Identify if any relevant insights from similar questions' solution approaches can be incorporated in your solution.
4. Solve the main problem now after consolidating all insights.
5. Explain reasoning behind each step.
6. Provide final answer clearly.
7. Give reference to exact insights you used from similar questions' solution approaches in the thoughts ONLY, NEVER in the generated solution.
8. Use appropriate {subject} terminologies and concepts.
9. Provide final answer clearly.

<OUTPUT_FORMAT>
You MUST return a single, valid JSON object for your solution. The structure MUST be as follows:
{{
    "thoughts": "Your thinking process about the approach and comparison with similar question approaches.",
    "generated_solution": "The final step-by-step solution for the problem."
}}
</OUTPUT_FORMAT>

</INSTRUCTIONS>
"""

solution_comparison_prompt = """You are an impartial and expert {subject} professor at Stanford and MIT, acting as a judge. Your task is to blindly compare two solutions, SOLUTION_A and SOLUTION_B, for the same problem (MAIN_PROBLEM) and assess which solution is better. You must be objective and provide a structured comparison based ONLY on the content provided.

<INSTRUCTIONS>
You will evaluate the two solutions across {metric} metric. You will decide on a winner, the margin of victory, and provide your reasoning. Think step-by-step before making a final decision.

<METRIC_FOR_EVAL>
{metric}:
{metric_description}
</METRIC_FOR_EVAL>    

<OUTPUT_FORMAT>
You MUST return a single, valid JSON object. The structure MUST be as follows:
{{
    "winner": "SOLUTION_A" | "SOLUTION_B" | "TIE",
    "margin_of_winning": "A value between 0.0 and 1.0, with 0.0 meaning no margin and 1.0 meaning a complete win",
    "reasoning": "Your detailed reasoning here."
}}
</OUTPUT_FORMAT>

</INSTRUCTIONS>
"""

solution_comparison_user_prompt="""<ORIGINAL_PROBLEM>
{main_question}
</ORIGINAL_PROBLEM>

<SOLUTION_A>
{solution_a_text}
</SOLUTION_A>

<SOLUTION_B>
{solution_b_text}
</SOLUTION_B>
"""

solution_performance_analysis_prompt = """USER:
You are senior AI Prompt Engineer conducting a post mortem analysis. Your task is to deduce why providing SIMILAR QUESTIONS to the LLM while solving MAIN QUESTION resulted in a specific outcome.

<CONTEXT>
SUBJECT - {subject}
JUDGE EVALUATION - Solution A was generated with help of similar questions and Solution B was generated without. A Judge evaluated both the solutions and scored them justly. 
OUTCOME - Solution Generated with Similar Questions resulted in {outcome}.
PERFORMANCE SCORE - {performance_score} (Value is between -1 and 1. Positive score means solution generated with help of similar questions won, Negative means it lost to the solution that was generated without help of similar questions.)
</CONTEXT>

<MAIN_QUESTION>
{main_question}
</MAIN_QUESTION>

<SIMILAR_QUESTIONS_PROVIDED_TO_LLM>
{similar_questions}
</SIMILAR_QUESTIONS_PROVIDED_TO_LLM>

<JUDGE_EVALUATION>
{judge_evaluation}
</JUDGE_EVALUATION>

<INSTRUCTIONS>
Based on all the information, form a clear hypothesis explaining the root cause of the problem. Pinpoint the specific element in the "Similar Questions" that likely led to this success or failure.

<OUTPUT_FORMAT>
You MUST return a single, valid JSON object. The structure MUST be as follows:
{{
    "hypothesis" : "Your hypothesis",
    "evidence" : "Evidence gathered that support your hypothesis" 
}}
</OUTPUT_FORMAT>

</INSTRUCTIONS>

ASSISTANT:
"""

insight_generation_prompt="""You are a Lead AI Strategist. You have been given a series of root cause analyses for cases where using "similar questions" to guide an LLM either helped (successes) or hurt (failures) its performance.

Your task is to synthesize these findings into a high level report with actionable recommendations. Look for recurring patterns in the analyses.

<ANALYSIS_OF_SUCCESSFUL_CASES>
{success_analysis}
</ANALYSIS_OF_WON_CASES>

<ANALYSIS_OF_FAILED_CASES>
{failure_analysis}
</ANALYSIS_OF_FAILED_CASES>
 
<TASK>
Based on the patterns you observe, generate 3-5 concrete, actionable recommendations to improve our prompts.
</TASK>

<OUTPUT_FORMAT>
You MUST return a single, valid JSON object. The structure MUST be as follows:
{{
    "insights":[
        {{
            "recommendation": "...",
            "reasoning" : "Justify why this should be improved"
        }},
        {{
            "recommendation": "...",
            "reasoning" : "..."
        }},
        {{...}}
    ]
}}
</OUTPUT_FORMAT>
"""

def format_insight_generation_prompt(success_analysis_arr, failure_analysis_arr):
    success_text = "\n".join([f"<analysis>{a['hypothesis']}</analysis>" for a in success_analysis_arr])
    failure_text = "\n".join([f"<analysis>{a['hypothesis']}</analysis>" for a in failure_analysis_arr])
    
    return insight_generation_prompt.format(success_analysis = success_text, failure_analysis = failure_text)

def format_solution_performance_analysis_prompt(subject, outcome, performance_score, main_question, similar_questions, judge_evaluation):
    formatted_similar_questions = "\n\n".join([f"<SIMILAR_QUESTION_{idx + 1}>\n{sq['similar_question_text']}\n</SIMILAR_QUESTION_{idx + 1}>\n<SOLUTION_APPROACH_{idx + 1}>\n{sq['summarized_solution_approach']}\n</SOLUTION_APPROACH_{idx + 1}>" for idx,sq in enumerate(similar_questions)])
    
    return solution_performance_analysis_prompt.format(subject=subject, outcome=outcome, performance_score=performance_score, main_question=main_question, similar_questions=formatted_similar_questions, judge_evaluation=judge_evaluation)

def format_solution_comparison_system_prompt_array(subject, metrics) -> dict[str, str]:
    prompts = {}
    for metric, description in metrics.items():
        prompts[metric] = solution_comparison_prompt.format(subject=subject, metric=metric, metric_description=description)
    return prompts

def format_solution_comparison_user_prompt(main_question, solution_a, solution_b):
    return solution_comparison_user_prompt.format(main_question=main_question, solution_a_text=solution_a, solution_b_text=solution_b)

def format_solution_builder_prompt(subject, main_question, similar_questions=[], with_similar=False):
    if with_similar:
        formatted_similar_questions = "\n\n".join([f"<SIMILAR_QUESTION_{idx + 1}>\n{sq['similar_question_text']}\n</SIMILAR_QUESTION_{idx + 1}>\n<SOLUTION_APPROACH_{idx + 1}>\n{sq['summarized_solution_approach']}\n</SOLUTION_APPROACH_{idx + 1}>" for idx,sq in enumerate(similar_questions)])
        return solution_builder_with_similar_prompt.format(subject=subject, main_question=main_question, similar_questions=formatted_similar_questions)
    else:
        return solution_builder_prompt.format(subject=subject, main_question=main_question)

def format_relevance_alignment_system_prompt(subject):
    return relevance_alignment_system_prompt.format(subject=subject)

def format_relevance_similarity_system_prompt(subject):
    return relevance_similarity_system_prompt.format(subject=subject)

def format_relevance_user_prompt(main_question, similar_questions, include_solutions = False):
    if include_solutions:
        formatted_similar_questions = "\n\n".join([f"<SIMILAR_QUESTION_{idx + 1}>\n{sq['similar_question_text']}\n</SIMILAR_QUESTION_{idx + 1}>\n<SOLUTION_APPROACH_{idx + 1}>\n{sq['summarized_solution_approach']}\n</SOLUTION_APPROACH_{idx + 1}>" for idx,sq in enumerate(similar_questions)])
    else:
        formatted_similar_questions = "\n\n".join([f"<SIMILAR_QUESTION_{idx + 1}>\n{sq['similar_question_text']}\n</SIMILAR_QUESTION_{idx + 1}>" for idx,sq in enumerate(similar_questions)])
  
    return relevance_user_prompt.format(main_question = main_question, similar_questions = formatted_similar_questions)