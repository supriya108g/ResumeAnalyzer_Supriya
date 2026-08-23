import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ResumeAnalyzer
from ResumeAnalyzer import (
    ResumeRequest,
    analyze_candidate_profile,
    calculate_resume_score,
    extract_candidate_name,
    extract_education_summary,
    extract_keywords,
    ensure_improved_resume_score,
    finalize_generated_resume,
    is_matching_resume_request,
    mask_pii,
    process_resume_workflow,
    run_crewai_workflow,
    score_generated_resume,
)


@pytest.fixture(autouse=True)
def isolate_external_ai_services(monkeypatch):
    """Keep tests independent of local secrets, OpenAI, and CrewAI network calls."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(ResumeAnalyzer, "call_llm", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        ResumeAnalyzer,
        "run_crewai_workflow",
        lambda *_args, **_kwargs: {"crew_status": "Skipped in tests"},
    )


# OpenAI configuration and fail-fast behavior

def test_run_crewai_workflow_skips_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_crewai_workflow(
        "Experienced DevOps engineer with Docker, AWS, CI/CD",
        "Junior DevOps Support Engineer with Azure Cloud Support",
        "Fallback context",
    )

    assert result["crew_status"] in {"Skipped", "Not available"}
    assert "OPENAI_API_KEY" in str(result.get("notes", "")) or "Not available" in str(result.get("notes", ""))


def test_process_resume_requires_openai_key_without_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = process_resume_workflow(
        ResumeRequest(
            candidate_name="Test Candidate",
            resume_text="Sales professional with account management and communication experience.",
            job_description="Seeking a sales professional with account management experience.",
            user_id="test-user",
        )
    )

    assert result["status"] == "Configuration Required"
    assert result["reason"] == "missing_openai_api_key"
    assert "OPENAI_API_KEY" in result["message"]
    assert result["generated_resume"] is None


# End-to-end workflow decisions and score consistency

def test_process_resume_workflow_returns_two_scores():
    request = ResumeRequest(
        candidate_name="Test Candidate",
        resume_text="DevOps engineer with AWS, Docker, Kubernetes, CI/CD, Python, Linux, Terraform, monitoring, and deployment experience.",
        job_description="We need a DevOps engineer with AWS, Docker, Kubernetes, CI/CD and infrastructure automation experience.",
        user_id="test-user",
    )

    result = process_resume_workflow(request)

    assert "resume_score_out_of_10" in result
    assert "improved_resume_score_out_of_10" in result
    assert result["improved_resume_score_out_of_10"] == score_generated_resume(
        result["generated_resume"], request.job_description
    )


def test_uploaded_score_stays_below_seven_when_improvement_is_at_least_seven():
    resume = "I have basic Excel skills and some reporting work but no major technical background."
    jd = "Need a data analyst with SQL, Python, Power BI, and ETL experience."

    request = ResumeRequest(
        candidate_name="Test Candidate",
        resume_text=resume,
        job_description=jd,
        user_id="test-user",
    )

    result = process_resume_workflow(request)

    assert result["uploaded_resume_score_out_of_10"] < 7.0
    assert result["improved_resume_score_out_of_10"] == score_generated_resume(
        result["generated_resume"], request.job_description
    )


def test_final_resume_preserves_candidate_name_and_completed_education():
    generated = {
        "professional_summary": "DevOps engineer",
        "education_section": "Bachelor of Science in Computer Science (In Progress)",
    }
    original_resume = """
    ARIKERA LIKITHA
    EDUCATION
    Bachelor of Technology in Computer Science, ABC University
    Graduated 2022
    AWARDS
    Employee recognition award
    """

    result = finalize_generated_resume(generated, "ARIKERA LIKITHA", original_resume)

    # Uploaded facts must replace conflicting education invented by a model.
    assert result["candidate_name"] == "ARIKERA LIKITHA"
    assert "Bachelor of Technology" in result["education_section"]
    assert "Graduated 2022" in result["education_section"]
    assert "In Progress" not in result["education_section"]


def test_existing_job_reuses_saved_resume_without_agent_outputs(monkeypatch):
    job_description = "Need a DevOps engineer with AWS, Docker, Kubernetes, and CI/CD experience."
    saved_resume = {
        "professional_summary": "DevOps engineer with AWS delivery experience",
        "skills_section": "AWS, Docker, Kubernetes, CI/CD",
        "experience_bullets": ["Delivered cloud deployment experience."],
        "project_descriptions": [],
        "education_section": "Bachelor of Technology",
    }
    monkeypatch.setattr(
        ResumeAnalyzer,
        "find_matching_logged_resume",
        lambda _job_description, _resume_text: {
            "request_id": "saved-request",
            "generated_resume": saved_resume,
            "crewai": {"crew_status": "Executed"},
        },
    )

    result = process_resume_workflow(
        ResumeRequest(
            candidate_name="Candidate Name",
            resume_text="Entry-level candidate with basic Python knowledge.",
            job_description=job_description,
            user_id="test-user",
        )
    )

    assert result["used_existing_log"] is True
    assert result["decision"] == "Reuse existing improved resume"


# Domain classification and skill extraction

def test_healthcare_resume_uses_healthcare_domain_and_skills():
    resume = "Registered nurse experienced in patient care, clinical assessment, EHR, and medication administration."

    profile = analyze_candidate_profile(resume)

    assert profile["primary_domain"] == "Healthcare"
    assert extract_keywords(resume) == [
        "patient care",
        "clinical assessment",
        "ehr",
        "medication administration",
    ]


def test_finance_resume_scores_against_finance_job_skills():
    resume = "Accountant skilled in financial reporting, general ledger, reconciliation, and GAAP."
    job_description = "Seeking an accountant with GAAP, financial reporting, reconciliation, and auditing experience."

    assert calculate_resume_score(resume, job_description) > 0
    assert analyze_candidate_profile(job_description)["primary_domain"] == "Finance and Accounting"


# Candidate identity, education boundaries, and PII protection

def test_candidate_name_skips_skill_title_from_pdf_extraction():
    resume = """
    Java
    TestNG, Cucumber, Selenium, Apache POI
    Selenium WebDriver
    Scripting and Markup
    XML, JavaScript
    CI/CD
    GitLab, Jenkins
    Database
    Oracle, MySQL
    Bug Reporting Tool
    Manual API testing using Postman
    Designed an automation framework using Java
    732-532-8891
    candidate@example.com
    PROFESSIONAL EXPERIENCE
    SUPRATIM DASGUPTA
    QA Automation Engineer
    PROFESSIONAL SUMMARY
    Experienced software testing professional
    """

    assert extract_candidate_name(resume) == "SUPRATIM DASGUPTA"


def test_candidate_name_is_blank_when_resume_has_no_name():
    resume = """
    FirstName LastName
    Street Address
    City State Zip
    PROFESSIONAL SUMMARY
    Motivated sales professional with communication and data analysis skills.
    SKILLS
    Salesforce, Salesloft, Gong
    """

    assert extract_candidate_name(resume) == ""


def test_final_resume_marks_missing_education_as_not_provided():
    generated = {
        "candidate_name": "FirstName LastName",
        "education_section": "Bachelor's Degree from Example University",
    }
    original_resume = "Sales professional with communication and account management experience."

    result = finalize_generated_resume(generated, "", original_resume)

    assert result["candidate_name"] == ""
    assert result["education_section"] == "Education not provided in uploaded resume."


def test_education_stops_before_references_and_masks_pii():
    resume = """
    EDUCATION
    Masters in Business Administration, ICFAI University 2013 | Bachelors of Electronics, JNTU 2004
    | REFERENCES | Supratim Dasgupta | supratim.dasgupta@example.com | +91 9380766109 |
    WORK EXPERIENCE
    Automation Test Engineer, DBS, 2014-2015
    """

    education = extract_education_summary(resume)

    assert "Masters in Business Administration" in education
    assert "REFERENCES" not in education
    assert "Automation Test Engineer" not in education
    assert "supratim.dasgupta@example.com" not in education
    assert "9380766109" not in education


def test_education_preserves_complete_multiline_section():
    resume = """
    Acme Resources, Garden Grove, CA                         1989 - 1992
    Project Assistant
    EDUCATION
    UNIVERSITY of CALIFORNIA LOS ANGELES, Los Angeles, California          1988
    Bachelor of Arts, History
    Honors and Activities: Recipient of Full Scholarship, Varsity Baseball Captain, All-American
    COMPUTER EXPERIENCE
    Microsoft Windows 98, MS Word, Excel, and PowerPoint
    """

    education = extract_education_summary(resume)

    assert "UNIVERSITY of CALIFORNIA LOS ANGELES" in education
    assert "1988" in education
    assert "Bachelor of Arts, History" in education
    assert "Honors and Activities" in education
    assert "COMPUTER EXPERIENCE" not in education
    assert "Microsoft Windows" not in education


def test_education_recovers_degree_text_before_pdf_heading():
    resume = """
    Got third position in SQL Hackathon
    Masters in Business Administration,
    IT & Systems, ICFAI University 2013
    Bachelors of Electronics and
    Instrumentation Engineering
    JNTU,2004
    www.linkedin.com/in/candidate
    Software Test Engineer, HSA/DBS - March 2014-Dec 2015
    EDUCATION
    PUBLICATIONS
    Software Test Engineer, Infosys - Jan 2022 to Present
    """

    education = extract_education_summary(resume)

    assert "Masters in Business Administration" in education
    assert "ICFAI University 2013" in education
    assert "Bachelors of Electronics" in education
    assert "JNTU,2004" in education
    assert "linkedin.com" not in education
    assert "PUBLICATIONS" not in education
    assert "Software Test Engineer" not in education


def test_mask_pii_masks_linkedin_and_international_phone():
    text = "Email me@example.com, call +91 9380766109, or visit www.linkedin.com/in/supriya-123."

    masked = mask_pii(text)

    assert "me@example.com" not in masked
    assert "9380766109" not in masked
    assert "linkedin.com/in/supriya-123" not in masked


# ATS improvement and workflow side-effect isolation

def test_ats_guard_preserves_uploaded_matches_and_improves_score():
    resume = "QA engineer with Selenium, API testing, and CI/CD experience."
    job_description = "QA engineer with Selenium, API testing, CI/CD, and performance testing experience."
    uploaded_score = calculate_resume_score(resume, job_description)
    generated_resume = {
        "professional_summary": "QA engineer focused on software quality.",
        "skills_section": ["Selenium"],
        "experience_bullets": ["Executed automated tests."],
        "project_descriptions": ["Delivered a QA automation project."],
        "education_section": "Bachelor's degree",
    }

    improved_resume = ensure_improved_resume_score(
        generated_resume, resume, job_description, uploaded_score
    )

    assert "api testing" in improved_resume["skills_section"].lower()
    assert "ci/cd" in improved_resume["skills_section"].lower()
    assert score_generated_resume(improved_resume, job_description) > uploaded_score


def test_workflow_improved_score_is_greater_than_uploaded_score(monkeypatch):
    # Keep this score test deterministic and avoid log/vector-store writes.
    monkeypatch.setattr(ResumeAnalyzer, "find_matching_logged_resume", lambda *_args: None)
    monkeypatch.setattr(ResumeAnalyzer, "run_crewai_workflow", lambda *_args: {"crew_status": "Skipped"})
    monkeypatch.setattr(ResumeAnalyzer, "store_audit_log", lambda *_args: None)
    monkeypatch.setattr(ResumeAnalyzer, "add_resume_log_to_vector_store", lambda *_args: None)
    request = ResumeRequest(
        candidate_name="Alex Morgan",
        resume_text="QA engineer with Selenium and API testing experience.",
        job_description="QA engineer with Selenium, API testing, CI/CD, and performance testing experience.",
        user_id="test-user",
    )

    result = process_resume_workflow(request)

    assert result["improved_resume_score_out_of_10"] > result["uploaded_resume_score_out_of_10"]
    assert result["improved_resume_score_out_of_10"] == score_generated_resume(
        result["generated_resume"], request.job_description
    )


# Saved-result matching must bind both the job and the candidate resume

def test_saved_result_is_not_reused_for_a_different_resume():
    job_description = "Sales representative responsible for B2B sales, lead generation, CRM, and closing."
    qa_resume = "QA engineer with Selenium, API testing, Java, TestNG, and CI/CD automation experience."
    sales_resume = "Retail sales associate with prospecting, customer service, negotiation, and revenue growth experience."

    assert not is_matching_resume_request(
        job_description, sales_resume, job_description, qa_resume
    )


def test_saved_result_can_be_reused_for_same_resume_and_job():
    job_description = "Sales representative responsible for B2B sales, lead generation, CRM, and closing."
    resume = "Retail sales associate with prospecting, customer service, negotiation, and revenue growth experience."

    assert is_matching_resume_request(
        job_description, resume, job_description, resume
    )


# Regression coverage for resume layouts observed in uploaded documents

def test_retail_resume_extracts_single_name_and_employment_years():
    resume = """
    Foot Locker Sales Associate Resume
    Isaiah
    Current Residential Address:
    Career Objective:
    their needs
    Related Professional Experience
    Foot Locker Inc, Sales Associate
    June 2010 till date
    Other Experience
    Store Assistant, Fashion 32, CA
    August 2008 to May 2010
    """

    assert extract_candidate_name(resume) == "Isaiah"
    assert ResumeAnalyzer.extract_years_experience(resume) >= 15
    assert analyze_candidate_profile(resume)["candidate_level"] == "Senior"


def test_retail_education_excludes_achievements_and_skills():
    resume = """
    Education:
    ● Achieved 20% more than the actual target set
    ● Gained appreciation for initiating plans for developing sales prospects
    ● Achieved Diploma in High School, June 2013, Rising Stars Business Academy
    Youthbuild School, Moreno Valley, CA
    ● Major - English
    Skill Sets:
    ● Comprehensive knowledge of retail sales and merchandising
    """

    education = extract_education_summary(resume)

    assert "Diploma in High School" in education
    assert "Youthbuild School" in education
    assert "Major - English" in education
    assert "20% more" not in education
    assert "sales prospects" not in education
    assert "retail sales" not in education
