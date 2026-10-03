import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. SWITCHED MODEL: We are using 3.5-flash to bypass the exhausted 3.8-flash quota.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ==========================================
# Unified Schema (Everything in 1 Request)
# ==========================================
class ComprehensiveEvaluation(BaseModel):
    # Profile Data
    candidate_name: str = Field(description="Name of the candidate if available, else 'Unknown'")
    education_level: str = Field(description="Highest degree or field of study")
    years_of_experience: float = Field(description="Estimated total years of professional or internship experience")
    key_projects: List[str] = Field(description="Key project titles or major technical contributions")
    
    # Scoring & Gaps
    overall_match_score: int = Field(description="Overall ATS fit score from 0 to 100")
    technical_skills_score: int = Field(description="Score from 0 to 10 for technical skills alignment")
    experience_fit_score: int = Field(description="Score from 0 to 10 for experience depth and relevance")
    rationale: str = Field(description="Brief justification for the scores")
    matched_skills: List[str] = Field(description="Skills found in both resume and JD")
    missing_skills: List[str] = Field(description="Key skills required in JD that candidate lacks")
    
    # Generation
    coaching_rewrites: str = Field(description="3 high-impact resume bullet point rewrites formatted in Markdown")
    cover_letter: str = Field(description="A professional 3-4 paragraph cover letter formatted in Markdown")

# ==========================================
# Single Pipeline Orchestrator
# ==========================================
def run_full_evaluation_pipeline(resume_text: str, job_description: str) -> dict:
    """Executes Extraction, Scoring, Coaching, and Cover Letter in ONE single API call."""
    
    parser = JsonOutputParser(pydantic_object=ComprehensiveEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an elite ATS system, technical recruiter, and career coach. 
        Analyze the candidate's resume against the Job Description. 
        Provide a strictly formatted JSON response containing the extracted profile, ATS scoring, actionable resume rewrites, and a tailored cover letter.
        \n{format_instructions}"""),
        ("human", "Resume Content:\n{resume_text}\n\nTarget Job Description:\n{job_description}")
    ]).partial(format_instructions=parser.get_format_instructions())

    # Create the single execution chain
    chain = prompt | llm | parser

    # MAKE EXACTLY 1 API CALL
    raw_result = chain.invoke({
        "resume_text": resume_text,
        "job_description": job_description
    })

    # Map the unified JSON back into the separate dictionaries that app.py expects
    return {
        "profile": {
            "candidate_name": raw_result.get("candidate_name", "Unknown"),
            "education_level": raw_result.get("education_level", "Unknown"),
            "years_of_experience": raw_result.get("years_of_experience", 0),
            "key_projects": raw_result.get("key_projects", [])
        },
        "scores": {
            "overall_match_score": raw_result.get("overall_match_score", 0),
            "technical_skills_score": raw_result.get("technical_skills_score", 0),
            "experience_fit_score": raw_result.get("experience_fit_score", 0),
            "rationale": raw_result.get("rationale", ""),
            "matched_skills": raw_result.get("matched_skills", []),
            "missing_skills": raw_result.get("missing_skills", [])
        },
        "coaching": raw_result.get("coaching_rewrites", "No coaching provided."),
        "cover_letter": raw_result.get("cover_letter", "No cover letter generated.")
    }