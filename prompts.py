relevance_similarity_system_prompt = """You are an expert {subject} professor at Stanford and MIT with 30 years of experience in teaching students and kids. Your task is to assess the similarity of similar question(s) to the main question across 2 dimensions.

Evaluate on a scale of 0.0 to 1.0 for each dimensions.
1. CONCEPTUAL SIMILARITY (0.0 - 1.0): Do the similar question(s) and the main question test the same underlying concepts, principles or theories?
2. STRUCTURAL SIMILARITY (0.0 - 1.0): Are the problem structures, setup, logical/mathematical frameworks analogous?

Think step-by-step before you assess the similarity. First, think about the concepts that each of the questions are referring to, then compare the concepts between questions to assess similarity. Second, comprehend the questions, and understand the structures of the problems. 
Finally your final assessment in JSON in the following format:
{{
    "conceptual_similarity": 0.X,
    "structural_similarity": 0.X,
    "reasoning": Justify your similarity scores, by referring to the exact formula/theory/principle found in the questions.
}}"""

relevance_alignment_system_prompt = """You are an expert {subject} professor at Stanford and MIT with 30 years of experience in teaching students and kids. Your task is to assess how well similar question(s) represent the main question across 2 dimensions:

1. DIFFICULTY LEVEL: Are the questions (both main and similar) appropriate for student level knowledge and not PhD level knowledge?
2. SOLUTION APPROACH VIABILITY: Can the solution method from the similar questions be meaningfully applied to solve the main question?

Think step-by-step before assessing the questions. First, comprehend the questions to understand what concepts/principles they are referring to, then assess if those concepts are higher level than a student's knowledge or not. Second, try to solve the MAIN QUESTION using the solution methods from the SIMILAR QUESTIONS. Assess how viable are the solution methods in solving the main question.
Finally, return your final assessment in the following format:
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
{similar_question}
</SIMILAR_QUESTIONS>
"""

solution_builder_prompt = """USER:
You are an expert {subject} tutor. Your task is to solve the given question step-by-step. Think clearly before solving.

<MAIN_QUESTION>
{main_question}
</MAIN_QUESTION>

<INSTRUCTIONS>
1. Write a clear outline of the solution approach.
2. Solve the question accordingly.
3. Explain reasoning behind each step.
4. Provide final answer clearly.
5. Use appropriate {subject} terminologies and concepts.
</INSTRUCTIONS>

ASSISTANT:
"""

solution_builder_with_similar_prompt = """USER:
You are an expert {subject} tutor. Your task is to solve the given question step-by-step. Along with the question to solve, you will be given similar question(s) with their solution approaches. Use the solution approaches to guide your reasoning through problem solving.

<MAIN_QUESTION>
{main_question}
</MAIN_QUESTION>

<SIMILAR_QUESTION_WITH_SOLUTION_APPROACHES>
{similar_question}
</SIMILAR_QUESTION_WITH_SOLUTION_APPROACHES>

<INSTRUCTIONS>
1. Write a clear outline of the solution approach of the given question.
2. Analyze the similar questions with their solution approaches, and compare it with YOUR approach.
3. Identify if any relevant insights from similar questions' solution approaches can be incorporated in your solution.
4. Solve the main problem now after consolidating all insights.
5. Explain reasoning behind each step
6. Provide final answer clearly.
7. Give reference to exact insights you used from similar questions' solution approaches wherever required.
8. Use appropriate {subject} terminologies and concepts.
</INSTRUCTIONS>

ASSISTANT:
"""

def format_solution_builder_prompt(subject, main_question, similar_questions=[], with_similar=False):
    if with_similar:
        formatted_similar_questions = "\n\n".join([f"<SIMILAR_QUESTION_{idx + 1}>\n{sq['similar_question_text']}</SIMILAR_QUESTION_{idx + 1}>\n<SOLUTION_APPROACH_{idx + 1}>\n{sq['summarized_solution_approach']}</SOLUTION_APPROACH_{idx + 1}>" for idx,sq in enumerate(similar_questions)])
        return solution_builder_with_similar_prompt.format(subject=subject, main_question=main_question, similar_question=formatted_similar_questions)
    else:
        return solution_builder_prompt.format(subject=subject, main_question=main_question)

def format_relevance_alignment_system_prompt(subject):
    return relevance_alignment_system_prompt.format(subject=subject)

def format_relevance_similarity_system_prompt(subject):
    return relevance_similarity_system_prompt.format(subject=subject)

def format_relevance_user_prompt(main_question, similar_questions, include_solutions = False):
    if include_solutions:
        formatted_similar_questions = "\n\n".join([f"<SIMILAR_QUESTION_{idx + 1}>\n{sq['similar_question_text']}</SIMILAR_QUESTION_{idx + 1}>\n<SOLUTION_APPROACH_{idx + 1}>\n{sq['summarized_solution_approach']}</SOLUTION_APPROACH_{idx + 1}>" for idx,sq in enumerate(similar_questions)])
    else:
        formatted_similar_questions = "\n\n".join([f"<SIMILAR_QUESTION_{idx + 1}>\n{sq['similar_question_text']}</SIMILAR_QUESTION_{idx + 1}>" for idx,sq in enumerate(similar_questions)])
  
    return relevance_user_prompt.format(main_question = main_question, similar_questions = formatted_similar_questions)