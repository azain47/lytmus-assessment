from pydantic import BaseModel, Field
from enum import Enum

class Subjects(Enum):
    PHYSICS = "PHYSICS"
    MATHS = "MATHS"
    CHEMISTRY = "CHEMISTRY"

class AppropriateAlignment(Enum):
    YES = "YES"
    PARTIAL = "PARTIAL"
    NO = "NO"

class WinnerSolution(Enum):
    SOLUTION_A = "SOLUTION_A"
    SOLUTION_B = "SOLUTION_B"
    TIE = "TIE"

class RelevanceSimilarity(BaseModel):
    conceptual_similarity: float = Field(ge=0.0, le=1.0)
    structural_similarity: float = Field(ge=0.0, le=1.0) 
    reasoning: str

class RelevanceAlignment(BaseModel):
    is_difficulty_appropriate: AppropriateAlignment = Field(description="YES if difficulty is appropriate, PARTIAL if some higher level information is needed, NO if not appropriate.")
    is_solution_approach_viable: AppropriateAlignment = Field(description="YES if solution methods can be referenced to solve the main question, PARTIAL if some key external information is needed, NO if solution is irrelevant.")
    reasoning: str    
       
class RelevanceEvaluationReport(BaseModel):
    question_id: str
    similarity: RelevanceSimilarity
    alignment:  RelevanceAlignment

class GeneratedSolution(BaseModel):
    question_id: str
    generated_solution: str
    was_solved_with_similar_questions: bool
    
class MetricEvaluation(BaseModel):
    winner: WinnerSolution = Field(description="Which solution was superior for this specific metric?")
    margin_of_winning: float = Field(description="A score from 0.0 (a tie) to 1.0 (complete win for the winner solution) between the solutions A and B.", ge=0.0, le=1.0)
    reasoning: str = Field(description="A concise justification.")
    