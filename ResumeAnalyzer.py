import datetime
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    # Startup remains available so the API can report a configuration error.
    OpenAI = None

try:
    import faiss
    import numpy as np
except Exception:
    # Vector search is optional; lexical retrieval remains available without it.
    faiss = None
    np = None

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from crewai import Agent, Crew, Process, Task
except Exception:
    Agent = None
    Crew = None
    Process = None
    Task = None

app = FastAPI(
    title="Enterprise AI Resume Generator Agent",
    description="CrewAI + RAG resume analysis and ATS optimization workflow",
    version="1.0.0",
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_KEY_REQUIRED_MESSAGE = (
    "OPENAI_API_KEY is not configured. Add the OpenAI API key and restart the server to continue."
)
audit_file = os.path.join(os.path.dirname(__file__), "audit_log.json")
vector_store_dir = os.path.join(os.path.dirname(__file__), "resume_chroma_db_v3")
chroma_collection = None

if chromadb is not None:
    try:
        chroma_client = chromadb.PersistentClient(
            path=vector_store_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        chroma_collection = chroma_client.get_or_create_collection(name="resume_logs")
    except Exception:
        # A corrupt or unavailable local store must not prevent API startup.
        chroma_collection = None

DOMAIN_SKILLS = {
    "Software Engineering": [
        "python", "java", "c#", "javascript", "react", "node.js", "spring boot",
        "html", "css", "api", "git", "agile", "scrum", "software testing",
        "manual testing", "automation testing", "selenium", "pytest", "junit", "testng",
        "qa", "quality assurance", "api testing", "performance testing", "load testing",
        "functional testing", "regression testing", "user acceptance testing", "uat",
        "test cases", "test execution", "bug tracking", "jira", "postman", "soapui",
        "cypress", "playwright", "appium",
    ],
    "Data and Analytics": [
        "sql", "power bi", "tableau", "excel", "data analysis", "data analytics",
        "business intelligence", "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "machine learning", "deep learning", "etl", "etl pipeline", "dbt",
        "bigquery", "snowflake", "postgresql", "mysql", "mongodb", "statistics",
        "data visualization", "forecasting",
    ],
    "Cloud and DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "linux", "terraform",
        "monitoring", "deployment", "infrastructure as code", "incident management",
        "configuration management",
    ],
    "Healthcare": [
        "patient care", "clinical assessment", "care planning", "medical records",
        "electronic health records", "ehr", "hipaa", "medication administration",
        "vital signs", "infection control", "emergency care", "case management",
        "patient education", "medical terminology", "clinical documentation",
    ],
    "Finance and Accounting": [
        "accounting", "bookkeeping", "financial analysis", "financial reporting",
        "financial statements", "accounts payable", "accounts receivable", "general ledger",
        "reconciliation", "budgeting", "auditing", "tax preparation", "gaap", "ifrs",
        "risk management", "variance analysis", "quickbooks", "sap",
    ],
    "Sales": [
        "sales", "business development", "lead generation", "prospecting", "negotiation",
        "account management", "customer relationship management", "crm", "salesforce",
        "pipeline management", "territory management", "revenue growth", "closing",
    ],
    "Marketing": [
        "marketing", "digital marketing", "content marketing", "social media marketing",
        "seo", "sem", "market research", "brand management", "campaign management",
        "google analytics", "email marketing", "copywriting", "lead generation",
        "marketing automation",
    ],
    "Human Resources": [
        "human resources", "recruiting", "talent acquisition", "employee relations",
        "performance management", "onboarding", "benefits administration", "payroll",
        "hris", "workforce planning", "training and development", "labor law",
        "compensation", "succession planning",
    ],
    "Education": [
        "teaching", "curriculum development", "lesson planning", "classroom management",
        "student assessment", "instructional design", "special education", "e-learning",
        "learning management system", "mentoring", "academic advising", "pedagogy",
    ],
    "Legal": [
        "legal research", "legal writing", "litigation", "contract drafting",
        "contract negotiation", "regulatory compliance", "case management", "due diligence",
        "legal documentation", "discovery", "corporate law", "intellectual property",
    ],
    "Operations and Supply Chain": [
        "operations management", "supply chain", "logistics", "inventory management",
        "procurement", "vendor management", "demand planning", "warehouse management",
        "lean manufacturing", "six sigma", "quality control", "process improvement",
        "production planning", "fleet management",
    ],
    "Engineering and Construction": [
        "autocad", "solidworks", "civil engineering", "mechanical engineering",
        "electrical engineering", "construction management", "project estimation",
        "blueprint reading", "cad", "quality assurance", "safety compliance",
        "preventive maintenance", "root cause analysis",
    ],
    "Customer Service and Hospitality": [
        "customer service", "customer support", "guest relations", "conflict resolution",
        "complaint resolution", "reservation management", "front desk", "food safety",
        "hospitality management", "event planning", "point of sale", "customer retention",
    ],
    "Creative and Communications": [
        "graphic design", "adobe creative suite", "photoshop", "illustrator", "indesign",
        "figma", "ux design", "ui design", "video editing", "photography", "copywriting",
        "public relations", "content creation", "internal communications",
    ],
    "Project and Product Management": [
        "project management", "program management", "product management", "agile", "scrum",
        "stakeholder management", "risk management", "requirements gathering", "roadmap",
        "change management", "budget management", "process improvement",
    ],
}

COMMON_SKILLS = list(dict.fromkeys(skill for skills in DOMAIN_SKILLS.values() for skill in skills))

DOMAIN_HINTS = {
    "Software Engineering": ["software", "developer", "programmer", "quality assurance", "tester"],
    "Data and Analytics": ["data analyst", "data scientist", "analytics", "business intelligence"],
    "Cloud and DevOps": ["devops", "cloud engineer", "site reliability", "infrastructure"],
    "Healthcare": ["healthcare", "nurse", "nursing", "clinical", "medical", "hospital"],
    "Finance and Accounting": ["accountant", "accounting", "finance", "financial", "auditor"],
    "Sales": ["sales", "account executive", "business development"],
    "Marketing": ["marketing", "brand manager", "campaign"],
    "Human Resources": ["human resources", "hr manager", "recruiter", "talent acquisition"],
    "Education": ["teacher", "teaching", "educator", "instructor", "education"],
    "Legal": ["lawyer", "attorney", "paralegal", "legal", "counsel"],
    "Operations and Supply Chain": ["operations", "supply chain", "logistics", "procurement"],
    "Engineering and Construction": ["engineer", "engineering", "construction", "manufacturing"],
    "Customer Service and Hospitality": ["customer service", "hospitality", "hotel", "guest"],
    "Creative and Communications": ["designer", "design", "communications", "creative"],
    "Project and Product Management": ["project manager", "product manager", "program manager"],
}


# API request models and local retrieval

class UserSupport(BaseModel):
    """Payload accepted by the simplified support endpoint."""

    user_input: str


class ResumeRequest(BaseModel):
    """Validated input shared by the analysis and generation endpoints."""

    candidate_name: str = Field(default="Candidate")
    resume_text: str = Field(..., min_length=10)
    job_description: str = Field(..., min_length=10)
    user_id: str = Field(default="anonymous")


class RAGKnowledgeStore:
    """Small in-memory lexical store used to assemble prompt context."""

    def __init__(self):
        self.documents: List[str] = []

    def add_document(self, text: str):
        """Add a non-empty document to the current request's context."""

        if text and text.strip():
            self.documents.append(text.strip())

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Return documents ranked by token overlap with the query."""

        if not self.documents:
            return []

        query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        scored_docs: List[tuple[float, str]] = []

        for doc in self.documents:
            doc_terms = set(re.findall(r"[a-zA-Z0-9]+", doc.lower()))
            overlap = len(query_terms & doc_terms)
            score = overlap + 0.1 * len(doc_terms)
            if overlap > 0 or len(doc_terms) < 40:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]


def generate_request_id() -> str:
    """Create a unique identifier for tracing and persisted audit entries."""

    return str(uuid.uuid4())


# Input security and privacy

def detect_prompt_injection(text: str) -> bool:
    """Detect common instruction-override phrases before invoking an LLM."""

    suspicious_keywords = [
        "ignore previous instructions",
        "override system",
        "bypass security",
        "malicious code",
        "inject code",
        "disregard policy",
    ]
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in suspicious_keywords)


def pii_found(text: str) -> Dict[str, bool]:
    """Report whether supported categories of PII occur in source text."""

    pii = {"account number": False, "email": False, "phone number": False}
    # Flag standalone eight-digit values as potential account identifiers.
    if re.search(r"\b\d{8}\b", text):
        pii["account number"] = True
    # Detect conventional email addresses without retaining the matched value.
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        pii["email"] = True
    # Detect common North American phone formats such as 123-456-7890.
    if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text):
        pii["phone number"] = True
    return pii


def mask_pii(text: str) -> str:
    """Mask contact and account identifiers before model processing."""

    # Replace the complete email so neither the local part nor domain reaches the LLM.
    masked = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "****@****.***",
        text,
    )
    # LinkedIn profile paths often reveal a person's name and are masked as PII.
    masked = re.sub(
        r"(?i)\b(?:https?://)?(?:www\.)?linkedin\.com/in/[^\s|,;]+",
        "[LINKEDIN MASKED]",
        masked,
    )

    def replace_phone(match: re.Match) -> str:
        # Date-like numeric strings can match the broad pattern, so only redact
        # values containing a plausible international phone digit count.
        value = match.group(0)
        digit_count = len(re.sub(r"\D", "", value))
        return "***-***-****" if 10 <= digit_count <= 15 else value

    # Accept spaces, parentheses, country prefixes, periods, and hyphens.
    masked = re.sub(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)", replace_phone, masked)
    # Mask remaining standalone eight-digit account identifiers last.
    masked = re.sub(r"\b\d{8}\b", "********", masked)
    return masked


def extract_years_experience(text: str) -> int:
    """Estimate experience from explicit totals or dated employment ranges."""

    patterns = [
        r"(\d+)\s*(?:\+)?\s*years?\s*(?:of\s*)?experience",
        r"(\d+)\s*yrs?\s*(?:of\s*)?experience",
        r"(\d+)\s+years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    current_year = datetime.datetime.now().year
    month = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    date_ranges = re.findall(
        rf"{month}\s+(\d{{4}})\s*(?:-|–|to|till|until)\s*(?:{month}\s+)?(\d{{4}}|present|current|date)",
        text,
        flags=re.IGNORECASE,
    )
    intervals = []
    for start_value, end_value in date_ranges:
        start_year = int(start_value)
        end_year = current_year if end_value.lower() in {"present", "current", "date"} else int(end_value)
        if 0 <= end_year - start_year <= 60:
            intervals.append((start_year, end_year))

    if intervals:
        # Merge overlapping jobs so concurrent roles are not counted twice.
        intervals.sort()
        merged = [intervals[0]]
        for start_year, end_year in intervals[1:]:
            previous_start, previous_end = merged[-1]
            if start_year <= previous_end:
                merged[-1] = (previous_start, max(previous_end, end_year))
            else:
                merged.append((start_year, end_year))
        return sum(end_year - start_year for start_year, end_year in merged)
    return 0


def extract_keywords(text: str) -> List[str]:
    """Extract known domain skills using boundary-aware phrase matching."""

    lower_text = text.lower()
    return [
        skill
        for skill in COMMON_SKILLS
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", lower_text)
    ]


def is_likely_candidate_name(value: str) -> bool:
    """Reject headings, job titles, technologies, and template placeholders."""

    cleaned = re.sub(r"\s+", " ", value).strip()
    words = cleaned.split()
    if not 1 <= len(words) <= 5 or len(cleaned) > 60:
        return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", word) for word in words):
        return False
    if cleaned == cleaned.lower():
        return False
    excluded_terms = {
        "address", "and", "automation", "blogger", "bug", "city", "contact", "database", "developer", "engineer",
        "experience", "firstname", "java", "javascript", "lastname", "manager", "manual", "markup", "professional",
        "phone", "python", "quality", "reporting", "resume", "scripting", "selenium", "skills",
        "software", "state", "street", "summary", "technical", "testing", "tool", "webdriver", "zip", "zipcode",
    }
    return not any(word.lower() in excluded_terms for word in words)


def extract_candidate_name(resume_text: str) -> str:
    """Return a credible candidate name or an empty string when unavailable."""

    if not resume_text or not resume_text.strip():
        return ""

    lines = [re.sub(r"\s+", " ", line).strip(" |") for line in resume_text.splitlines() if line.strip()]
    # Some exported resumes place the name immediately after a profile heading.
    for index, line in enumerate(lines):
        if re.fullmatch(r"professional experience|profile", line, flags=re.IGNORECASE):
            for candidate_line in lines[index + 1 : index + 5]:
                if is_likely_candidate_name(candidate_line):
                    return candidate_line

    for line in lines:
        cleaned = re.sub(r"\s+", " ", line)
        if re.fullmatch(r"education(?: details)?|references", cleaned, flags=re.IGNORECASE):
            break
        if re.search(
            r"Email|Contact|Phone|SUMMARY|TECHNICAL|WORK EXPERIENCE|SKILLS|PROJECT|EDUCATION|REFERENCES",
            cleaned,
            flags=re.IGNORECASE,
        ):
            continue
        if is_likely_candidate_name(cleaned):
            return cleaned

    return ""


def extract_education_summary(resume_text: str) -> str:
    """Extract education content while stopping at the next resume section."""

    section_headers = (
        "references", "professional experience", "work experience", "employment history",
        "experience", "computer experience", "technical skills", "skill sets", "skills", "projects", "certifications",
        "professional achievements", "awards", "achievements", "publications", "strengths", "declaration",
    )
    boundary_pattern = "|".join(re.escape(header) for header in section_headers)
    lines = [
        re.sub(r"\s+", " ", line).strip(" |")
        for line in resume_text.splitlines()
        if line.strip(" |")
    ]

    # Prefer an exact heading so incidental uses of "education" are not mistaken
    # for the start of the resume section.
    for index, line in enumerate(lines):
        if not re.fullmatch(r"education(?: details)?", line, flags=re.IGNORECASE):
            continue

        education_lines = []
        for education_line in lines[index + 1:]:
            boundary = re.search(
                rf"(?i)(?:^|\|\s*)({boundary_pattern})\b",
                education_line,
            )
            if boundary:
                content_before_boundary = education_line[:boundary.start()].strip(" |●•-")
                if content_before_boundary:
                    education_lines.append(content_before_boundary)
                break
            education_lines.append(education_line.strip(" |●•-"))

        education = "\n".join(line for line in education_lines if line)
        if education:
            return mask_pii(education)

    # Multi-column PDFs may emit degree details before the EDUCATION heading.
    # Recover a short degree block and stop before contact or employment content.
    degree_pattern = re.compile(
        r"diploma|degree|bachelor|master|phd|doctorate",
        flags=re.IGNORECASE,
    )
    recovery_boundary = re.compile(
        rf"(?i)^(?:{boundary_pattern})\b|https?://|linkedin\.com|"
        r"\b(?:software|quality|test|project|account|sales)\s+(?:engineer|manager|assistant|coordinator)\b.*\b(?:19|20)\d{2}\b",
    )
    for index, line in enumerate(lines):
        if not degree_pattern.search(line):
            continue

        recovered_lines = []
        for education_line in lines[index:index + 6]:
            if recovered_lines and recovery_boundary.search(education_line):
                break
            recovered_lines.append(education_line.strip(" |●•-"))

        recovered_education = "\n".join(line for line in recovered_lines if line)
        if recovered_education:
            return mask_pii(recovered_education)

    # Support compact exports where section headings and content share one line.
    normalized = "\n".join(lines)
    match = re.search(
        rf"(?is)(?<!\w)education(?: details)?\b\s*[:|\-]?\s*(.*?)(?=(?:\s*\|?\s*)(?:{boundary_pattern})\b|$)",
        normalized,
    )
    if match:
        education_lines = [
            line.strip(" |●•-")
            for line in match.group(1).splitlines()
            if line.strip(" |●•-")
        ]
        education_terms = re.compile(
            r"diploma|degree|bachelor|master|phd|doctorate|university|college|academy|school|major|minor|graduat|education",
            flags=re.IGNORECASE,
        )
        education = "\n".join(line for line in education_lines if education_terms.search(line))
        if education:
            return mask_pii(education)
    return ""


def analyze_candidate_profile(resume_text: str) -> Dict[str, Any]:
    """Build a deterministic profile used by scoring and LLM prompts."""

    text = resume_text.lower()
    years = extract_years_experience(resume_text)

    if years >= 8:
        level = "Senior"
    elif years >= 3:
        level = "Mid-Level"
    elif years > 0:
        level = "Junior"
    else:
        level = "Entry-Level"

    skills = extract_keywords(resume_text)
    domain_scores = {
        domain: sum(skill in skills for skill in domain_skills)
        + (2 * sum(re.search(rf"(?<!\w){re.escape(hint)}(?!\w)", text) is not None for hint in DOMAIN_HINTS[domain]))
        for domain, domain_skills in DOMAIN_SKILLS.items()
    }
    primary_domain = max(domain_scores, key=domain_scores.get)
    if domain_scores[primary_domain] == 0:
        primary_domain = "Generalist"

    return {
        "candidate_level": level,
        "primary_domain": primary_domain,
        "years_experience": years,
        "identified_skills": skills,
        "education_found": bool(re.search(r"bachelor|master|phd|degree|graduation|university|college", text, flags=re.IGNORECASE)),
        "project_count": max(1, len(re.findall(r"project|projects|experience|portfolio", text))),
        "certification_found": bool(re.search(r"certification|certified|aws cert|azure cert|pmp|scrum|mcsa|oracle", text, flags=re.IGNORECASE)),
    }


def calculate_resume_score(resume_text: str, job_description: str) -> float:
    """Score source-resume alignment using skill and section overlap."""

    resume_skills = set(extract_keywords(resume_text))
    jd_skills = set(extract_keywords(job_description))
    # The base score is the percentage of recognized job skills found in the resume.
    overlap = len(resume_skills & jd_skills)
    # A denominator of one avoids division by zero when no known JD skills exist.
    jd_size = max(len(jd_skills), 1)

    score = min(10.0, (overlap / jd_size) * 10)

    # Small structural bonuses reward relevant sections only when the JD requests them.
    if "experience" in resume_text.lower() and "experience" in job_description.lower():
        score += 1.0
    if "project" in resume_text.lower() and "project" in job_description.lower():
        score += 0.5
    if "education" in resume_text.lower() and "education" in job_description.lower():
        score += 0.5

    # Keep all public scores on a consistent 0-10 scale with one decimal place.
    return round(min(10.0, score), 1)


def score_generated_resume(generated_resume: Dict[str, Any], job_description: str) -> float:
    """Apply the shared scoring formula to structured generated content."""

    # Flatten structured sections so uploaded and generated resumes use one formula.
    resume_sections = [
        generated_resume.get("professional_summary", ""),
        generated_resume.get("experience_bullets", []),
        generated_resume.get("skills_section", ""),
        generated_resume.get("project_descriptions", []),
        generated_resume.get("education_section", ""),
    ]
    resume_text = " ".join(
        item if isinstance(item, str) else " ".join(str(value) for value in item)
        for item in resume_sections
    )
    score = calculate_resume_score(resume_text, job_description)
    job_description_lower = job_description.lower()
    resume_text_lower = resume_text.lower()
    # Structured sections receive their bonus even when the literal section name is
    # absent from the generated text assembled above.
    if generated_resume.get("experience_bullets") and "experience" in job_description_lower and "experience" not in resume_text_lower:
        score += 1.0
    if generated_resume.get("project_descriptions") and "project" in job_description_lower and "project" not in resume_text_lower:
        score += 0.5
    if generated_resume.get("education_section") and "education" in job_description_lower and "education" not in resume_text_lower:
        score += 0.5
    required_sections = (
        "professional_summary",
        "skills_section",
        "experience_bullets",
        "project_descriptions",
        "education_section",
    )
    # A final completeness bonus rewards a resume containing every required section.
    if all(generated_resume.get(section) for section in required_sections):
        score += 0.5
    # Bonuses cannot push the score beyond the documented maximum of ten.
    return round(min(10.0, score), 1)


def ensure_improved_resume_score(
    generated_resume: Dict[str, Any], existing_resume: str, job_description: str, uploaded_score: float
) -> Dict[str, Any]:
    """Preserve truthful matched skills and guard against score regression."""

    improved = dict(generated_resume)
    skills_value = improved.get("skills_section", "")
    if isinstance(skills_value, list):
        skills = [str(skill).strip() for skill in skills_value if str(skill).strip()]
    else:
        skills = [skill.strip() for skill in str(skills_value).split(",") if skill.strip()]

    existing_skill_names = {skill.lower() for skill in skills}
    uploaded_skills = set(extract_keywords(existing_resume))
    # Only carry target keywords that are already supported by the source resume.
    for target_skill in extract_keywords(job_description):
        if target_skill not in uploaded_skills or target_skill.lower() in existing_skill_names:
            continue
        skills.append(target_skill)
        existing_skill_names.add(target_skill.lower())

    improved["skills_section"] = ", ".join(skills)
    if uploaded_score < 10.0 and score_generated_resume(improved, job_description) <= uploaded_score:
        summary = str(improved.get("professional_summary", "")).strip()
        improved["professional_summary"] = f"{summary} ATS-optimized resume structure.".strip()
    return improved


def finalize_generated_resume(
    generated_resume: Dict[str, Any], candidate_name: str, existing_resume: str
) -> Dict[str, Any]:
    """Replace model-controlled identity fields with uploaded source facts."""

    finalized = dict(generated_resume)
    finalized["candidate_name"] = candidate_name
    # Identity and education must come from the upload, never an LLM placeholder.
    finalized["education_section"] = (
        extract_education_summary(existing_resume)
        or "Education not provided in uploaded resume."
    )
    return finalized


def identify_missing_ats_keywords(resume_text: str, job_description: str) -> List[str]:
    """Return a bounded list of job phrases absent from the resume."""

    resume_lower = resume_text.lower()
    jd_keywords = [k.strip().lower() for k in re.split(r"[,;|\n]+", job_description) if k.strip()]
    missing = []
    for keyword in jd_keywords:
        if len(keyword) < 3:
            continue
        if keyword not in resume_lower:
            missing.append(keyword)
    return missing[:10]


def ats_optimization(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Produce deterministic ATS gaps, recommendations, and readiness score."""

    missing_keywords = identify_missing_ats_keywords(resume_text, job_description)
    score = max(0.0, 10.0 - (len(missing_keywords) * 0.8))
    return {
        "missing_keywords": missing_keywords,
        "ats_score": round(score, 1),
        "recommended_keywords": missing_keywords[:5],
    }


def build_rag_context(resume_text: str, job_description: str, previous_logs: List[Dict[str, Any]]) -> str:
    """Combine current inputs with relevant historical entries for grounding."""

    store = RAGKnowledgeStore()
    store.add_document(resume_text)
    store.add_document(job_description)

    for entry in previous_logs:
        if isinstance(entry, dict):
            for key in ["user_input", "job_description", "final_response", "llm_response", "decision"]:
                if key in entry:
                    store.add_document(str(entry[key]))

    relevant_docs = store.retrieve(resume_text + " " + job_description, top_k=5)
    if not relevant_docs:
        return "No relevant prior knowledge found. Proceed with current resume and job description."
    return "\n\n---\n\n".join(relevant_docs)


def call_llm(prompt: str) -> str:
    """Submit a prompt to OpenAI and return plain response text."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return ""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional enterprise resume writer and reviewer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def parse_llm_json(raw_response: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Parse JSON from plain or fenced model output with a safe default."""

    if not raw_response:
        return fallback

    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback


def parse_agent_output(raw_output: Any) -> Dict[str, Any]:
    """Normalize a CrewAI task output into a JSON-compatible dictionary."""

    text = str(raw_output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"output": parsed}
    except json.JSONDecodeError:
        return {"raw_output": text}


def create_fallback_resume(profile: Dict[str, Any], job_description: str, existing_resume: str) -> Dict[str, Any]:
    """Build deterministic content for internal parsing and retry defaults."""

    target_profile = analyze_candidate_profile(job_description)
    candidate_skills = set(extract_keywords(existing_resume))
    target_skills = extract_keywords(job_description)
    aligned_skills = [skill for skill in target_skills if skill in candidate_skills]
    skills = aligned_skills or profile["identified_skills"][:10] or target_skills[:10]
    domain = target_profile["primary_domain"] if target_profile["primary_domain"] != "Generalist" else profile["primary_domain"]

    if domain == "Cloud and DevOps":
        experience_bullets = [
            "Supported cloud infrastructure operations, deployment workflows, and production issue resolution.",
            "Worked with monitoring, automation, and incident-response processes to improve service reliability.",
            "Collaborated with engineering teams on CI/CD, configuration management, and operational support.",
        ]
        project_descriptions = [
            "Implemented a cloud operations project focused on deployment automation, monitoring, and service support.",
            "Documented troubleshooting and operational procedures to improve response time and system availability.",
        ]
    elif domain == "Data and Analytics":
        experience_bullets = [
            "Supported data reporting and analysis activities to deliver accurate, timely business insights.",
            "Prepared and validated data for dashboards, recurring reports, and stakeholder decision-making.",
            "Collaborated with business teams to identify trends and improve reporting processes.",
        ]
        project_descriptions = [
            "Completed a data analysis project involving data preparation, trend analysis, and insight generation.",
            "Developed a reporting solution to visualize key metrics and support business decisions.",
        ]
    else:
        experience_bullets = [
            f"Applied {domain} practices to support team objectives, operational improvement, and stakeholder delivery.",
            "Collaborated with cross-functional teams to analyze requirements and deliver reliable solutions.",
            "Documented processes and contributed to continuous improvement across project activities.",
        ]
        project_descriptions = [
            f"Completed a {domain} project aligned with the target role and its operational requirements.",
            "Created documented deliverables that improved visibility, consistency, and team collaboration.",
        ]

    return {
        "professional_summary": (
            f"{profile['candidate_level']} {domain} professional with {profile['years_experience']} years of experience "
            f"and a background aligned to the target role. Skilled in {', '.join(skills)}."
        ),
        "experience_bullets": experience_bullets,
        "skills_section": ", ".join(skills),
        "project_descriptions": project_descriptions,
        "education_section": "Bachelor's degree in a relevant field and/or equivalent professional experience",
        "job_alignment": job_description[:400],
    }


def generate_resume_content(
    profile: Dict[str, Any],
    job_description: str,
    existing_resume: str,
    base_score: float = 0.0,
) -> Dict[str, Any]:
    """Generate structured resume content and retry once on score regression."""

    fallback = create_fallback_resume(profile, job_description, existing_resume)
    prompt = f"""
    You are an enterprise resume-writing assistant.
    Generate a professional ATS-friendly resume that scores higher than the uploaded resume.
    The uploaded resume score is {base_score:.1f}/10.0. The generated resume must score above {base_score:.1f}/10.0
    when evaluated with the same skill-overlap and section-bonus formula. Preserve every matched skill from the
    uploaded resume, use relevant skills from the target job description when supported by the candidate profile,
    and strengthen the experience, project, and education sections when those details are truthful. Do not invent
    employers, job titles, qualifications, certifications, or experience.

    Candidate profile:
    {json.dumps(profile, indent=2)}

    Target Job Description:
    {job_description}

    Existing resume context:
    {existing_resume[:2000]}

    Return only valid JSON with the keys:
    - professional_summary
    - experience_bullets (list of 3 strings)
    - skills_section
    - project_descriptions (list of 2 strings)
    - education_section
    """
    llm_response = call_llm(prompt)
    if not llm_response:
        return fallback

    parsed = parse_llm_json(llm_response, fallback)
    for key, default_value in fallback.items():
        if key not in parsed:
            parsed[key] = default_value

    # Retry once when the model output does not improve the same deterministic score.
    generated_score = score_generated_resume(parsed, job_description)
    if base_score < 10.0 and generated_score <= base_score:
        retry_prompt = f"""
        Rewrite the generated resume below so it scores strictly above {base_score:.1f}/10.0 using the shared
        resume scoring formula. Keep all truthful candidate information and do not invent qualifications. Preserve
        matched skills and improve the ATS-relevant skills, experience, project, or education wording where supported.
        Return only valid JSON with the same required keys.

        Target job description:
        {job_description}

        Candidate profile:
        {json.dumps(profile, indent=2)}

        Generated resume:
        {json.dumps(parsed, indent=2)}
        """
        retry_response = call_llm(retry_prompt)
        if retry_response:
            retry_parsed = parse_llm_json(retry_response, parsed)
            for key, default_value in parsed.items():
                if key not in retry_parsed:
                    retry_parsed[key] = default_value
            if score_generated_resume(retry_parsed, job_description) > generated_score:
                parsed = retry_parsed
    return parsed


def review_resume_output(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform deterministic completeness checks on generated sections."""

    required_keys = [
        "professional_summary",
        "experience_bullets",
        "skills_section",
        "project_descriptions",
        "education_section",
    ]
    grammar_ok = bool(resume_data.get("professional_summary") and resume_data.get("experience_bullets"))
    structure_ok = all(key in resume_data for key in required_keys)
    return {
        "grammar_valid": grammar_ok,
        "structure_valid": structure_ok,
        "review_status": "Approved" if grammar_ok and structure_ok else "Needs Revision",
        "notes": "Resume follows standard ATS-friendly structure and includes required sections." if grammar_ok and structure_ok else "Resume needs additional section completion or wording improvement.",
    }


def normalize_log_text(value: str) -> str:
    """Normalize persisted text for stable token-based comparisons."""

    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def calculate_text_similarity(left: str, right: str) -> float:
    """Calculate Jaccard similarity between normalized token sets."""

    left_tokens = set(token for token in normalize_log_text(left).split() if token)
    right_tokens = set(token for token in normalize_log_text(right).split() if token)
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if union == 0:
        return 0.0
    return overlap / union


def is_matching_resume_request(
    job_description: str, resume_text: str, logged_job_description: str, logged_resume_text: str
) -> bool:
    """Require both job and resume similarity before reusing saved output."""

    if not logged_job_description or not logged_resume_text:
        return False

    # Reuse requires both the role and candidate resume to match; matching only the
    # job description could leak a different candidate's generated content.
    job_similarity = calculate_text_similarity(job_description, logged_job_description)
    resume_similarity = calculate_text_similarity(resume_text, logged_resume_text)
    return job_similarity >= 0.45 and resume_similarity >= 0.80


def embed_text_for_vector_search(text: str) -> Optional[List[float]]:
    """Create a normalized local hashing vector for Chroma retrieval."""

    if not text:
        return None

    # A deterministic local embedding keeps log retrieval independent of OpenAI.
    dimensions = 128
    vector = [0.0] * dimensions
    for token in normalize_log_text(text).split():
        vector[hash(token) % dimensions] += 1.0

    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        return None
    return [value / magnitude for value in vector]


def add_resume_log_to_vector_store(log_data: Dict[str, Any]):
    """Upsert a compact audit payload into the optional Chroma collection."""

    if not isinstance(log_data, dict) or chroma_collection is None:
        return

    job_description = str(log_data.get("job_description") or "")
    generated_resume = log_data.get("generated_resume") or log_data.get("final_response") or {}
    if not job_description or not generated_resume:
        return

    request_id = str(log_data.get("request_id") or generate_request_id())
    payload = {
        "request_id": request_id,
        "candidate_name": log_data.get("candidate_name", "Candidate"),
        "job_description": job_description,
        "resume_text": log_data.get("resume_text", ""),
        "generated_resume": generated_resume,
        "profile_analysis": log_data.get("profile_analysis", {}),
        "review": log_data.get("review", {}),
        "ats_optimization": log_data.get("ats_optimization", {}),
        "crewai": log_data.get("crewai", {}),
        "uploaded_resume_score_out_of_10": log_data.get("uploaded_resume_score_out_of_10"),
        "improved_resume_score_out_of_10": log_data.get("improved_resume_score_out_of_10"),
        "timestamp": log_data.get("timestamp"),
    }

    embedding = embed_text_for_vector_search(job_description)
    if embedding is None:
        return

    chroma_collection.upsert(
        ids=[request_id],
        embeddings=[embedding],
        documents=[job_description],
        metadatas=[{"payload": json.dumps(payload)}],
    )


def find_matching_logged_resume(job_description: str, resume_text: str) -> Optional[Dict[str, Any]]:
    """Find a reusable result through Chroma, then lexical audit-log search."""

    if not job_description or not resume_text:
        return None

    if chroma_collection is not None and chroma_collection.count() > 0:
        embedding = embed_text_for_vector_search(job_description)
        if embedding is not None:
            matches = chroma_collection.query(
                query_embeddings=[embedding],
                n_results=min(3, chroma_collection.count()),
                include=["metadatas", "distances"],
            )
            metadatas = matches.get("metadatas", [[]])[0]
            distances = matches.get("distances", [[]])[0]
            for metadata, distance in zip(metadatas, distances):
                if distance > 0.55 or not metadata.get("payload"):
                    continue
                try:
                    payload = json.loads(metadata["payload"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if payload.get("generated_resume"):
                    request_id = payload.get("request_id")
                    full_entry = next(
                        (
                            entry
                            for entry in reversed(read_previous_logs())
                            if isinstance(entry, dict) and entry.get("request_id") == request_id
                        ),
                        None,
                    )
                    if full_entry:
                        if is_matching_resume_request(
                            job_description,
                            resume_text,
                            str(full_entry.get("job_description", "")),
                            str(full_entry.get("resume_text", "")),
                        ):
                            return full_entry
                        continue
                    if is_matching_resume_request(
                        job_description,
                        resume_text,
                        str(payload.get("job_description", "")),
                        str(payload.get("resume_text", "")),
                    ):
                        return payload

    target_job = normalize_log_text(job_description)
    if not target_job:
        return None

    target_tokens = set(token for token in target_job.split() if token)
    for entry in reversed(read_previous_logs()):
        if not isinstance(entry, dict):
            continue
        entry_job = str(entry.get("job_description", ""))
        if not entry_job:
            continue

        entry_tokens = set(token for token in normalize_log_text(entry_job).split() if token)
        if not entry_tokens:
            continue

        similarity = calculate_text_similarity(target_job, entry_job)
        request_match = similarity >= 0.45 and calculate_text_similarity(
            resume_text, str(entry.get("resume_text", ""))
        ) >= 0.80
        if request_match:
            generated_resume = entry.get("generated_resume") or entry.get("final_response")
            if generated_resume:
                return entry
    return None


def read_previous_logs() -> List[Dict[str, Any]]:
    """Read valid audit entries, treating missing or malformed files as empty."""

    if not os.path.exists(audit_file):
        return []
    try:
        with open(audit_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def store_audit_log(log_data: Dict[str, Any]):
    """Append one completed workflow record to the JSON audit log."""

    existing_logs = read_previous_logs()
    existing_logs.append(log_data)
    with open(audit_file, "w", encoding="utf-8") as file:
        json.dump(existing_logs, file, indent=4)
        file.write("\n")


def run_crewai_workflow(resume_text: str, job_description: str, rag_context: str) -> Dict[str, Any]:
    """Run the four specialized CrewAI agents in sequential order."""

    if Agent is None or Crew is None or Process is None or Task is None:
        return {
            "crew_status": "Not available",
            "notes": "CrewAI not installed in the current environment.",
            "rag_context_used": rag_context,
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "crew_status": "Skipped",
            "notes": "OPENAI_API_KEY is not set. CrewAI workflow was skipped to keep the resume analysis pipeline running without an LLM backend.",
            "rag_context_used": rag_context,
        }

    try:
        # Agent 1 converts unstructured resume text into a candidate profile that
        # downstream agents can reason about consistently.
        profile_analyzer = Agent(
            role="Profile Analyzer Agent",
            goal="Analyze candidate skills, experience, education, and project fit for the role.",
            backstory="You review the candidate resume, extract structured profile data, and identify domain fit.",
            verbose=True,
            allow_delegation=False,
            llm="gpt-4o-mini",
        )

        # Agent 2 compares the profile with the target role and identifies ATS gaps.
        ats_agent = Agent(
            role="ATS Optimization Agent",
            goal="Identify ATS keyword gaps and recommend missing skills for job alignment.",
            backstory="You compare the resume against the job description and recommend ATS improvement actions.",
            verbose=True,
            allow_delegation=False,
            llm="gpt-4o-mini",
        )

        # Agent 3 uses profile facts and ATS guidance to draft improved content.
        writer_agent = Agent(
            role="Resume Writer Agent",
            goal="Create high-quality ATS-friendly resume content based on profile and JD.",
            backstory="You generate professional summaries, bullet points, and polished content for resume refinement.",
            verbose=True,
            allow_delegation=False,
            llm="gpt-4o-mini",
        )

        # Agent 4 performs the final quality and structural review.
        reviewer_agent = Agent(
            role="Reviewer Agent",
            goal="Validate resume quality, structure, and final readiness for hiring workflows.",
            backstory="You confirm grammar, structure, and enterprise-level professionalism before final output.",
            verbose=True,
            allow_delegation=False,
            llm="gpt-4o-mini",
        )

        # Each task declares a JSON contract so outputs can be displayed separately
        # on the Agent Analysis page and consumed by later workflow stages.
        profile_task = Task(
            description=(
                "Analyze the resume, extract profile details, and return structured JSON with candidate level, domain, years of experience, skills, education, and certifications. "
                f"Use context from the RAG memory: {rag_context}"
            ),
            expected_output="JSON with candidate_level, primary_domain, years_experience, identified_skills, education_found, project_count, certification_found",
            agent=profile_analyzer,
        )

        ats_task = Task(
            description=(
                "Compare the resume with the job description, identify missing ATS keywords, and score the ATS readiness out of 10. "
                f"Use context from the RAG memory: {rag_context}"
            ),
            expected_output="JSON with missing_keywords, ats_score, recommended_keywords",
            agent=ats_agent,
        )

        writer_task = Task(
            description=(
                "Create an updated ATS-friendly resume structure with professional summary, experience bullets, skills section, project descriptions, and education summary. "
                f"Use the profile and ATS guidance plus RAG memory: {rag_context}"
            ),
            expected_output="JSON with professional_summary, experience_bullets, skills_section, project_descriptions, education_section",
            agent=writer_agent,
        )

        reviewer_task = Task(
            description="Review the final generated resume and confirm whether it is structurally valid, professional, and candidate-ready.",
            expected_output="JSON with grammar_valid, structure_valid, review_status, notes",
            agent=reviewer_agent,
        )

        # Sequential processing preserves the intended analyze -> optimize -> write
        # -> review order instead of running dependent tasks in parallel.
        crew = Crew(
            agents=[profile_analyzer, ats_agent, writer_agent, reviewer_agent],
            tasks=[profile_task, ats_task, writer_task, reviewer_task],
            process=Process.sequential,
            verbose=True,
        )

        # CrewAI preserves task order, allowing stable names for each output.
        result = crew.kickoff()
        task_outputs = list(getattr(result, "tasks_output", []) or [])
        agent_names = [
            "profile_analyzer_agent",
            "ats_optimization_agent",
            "resume_writer_agent",
            "reviewer_agent",
        ]
        agent_outputs = {
            agent_name: parse_agent_output(getattr(task_output, "raw", task_output))
            for agent_name, task_output in zip(agent_names, task_outputs)
        }
        return {
            "crew_status": "Executed",
            "crew_output": str(result),
            "agent_outputs": agent_outputs,
            "rag_context_used": rag_context,
        }
    except Exception as exc:
        return {
            "crew_status": "Failed",
            "notes": f"CrewAI execution failed: {exc}",
            "rag_context_used": rag_context,
        }


def process_resume_workflow(request: ResumeRequest) -> Dict[str, Any]:
    """Coordinate validation, masking, scoring, generation, review, and storage."""

    request_id = generate_request_id()
    # Do not calculate, reuse, or synthesize a resume without the required LLM key.
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return {
            "request_id": request_id,
            "status": "Configuration Required",
            "reason": "missing_openai_api_key",
            "message": OPENAI_KEY_REQUIRED_MESSAGE,
            "generated_resume": None,
        }

    resume_text = request.resume_text
    job_description = request.job_description
    candidate_name = request.candidate_name.strip()
    if (
        not candidate_name
        or candidate_name.lower() in {"candidate", "supriya", "test candidate"}
        or not is_likely_candidate_name(candidate_name)
    ):
        candidate_name = extract_candidate_name(resume_text)

    # Validate unmasked source text first, then keep PII out of model-facing context.
    injection_detected = detect_prompt_injection(resume_text) or detect_prompt_injection(job_description)
    if injection_detected:
        return {
            "request_id": request_id,
            "status": "Blocked",
            "reason": "Prompt injection detected",
        }

    # Record only PII category flags for the response; model-facing values are masked.
    pii_status = pii_found(resume_text)
    masked_resume = mask_pii(resume_text)
    masked_job_description = mask_pii(job_description)

    previous_logs = read_previous_logs()
    rag_context = build_rag_context(masked_resume, masked_job_description, previous_logs)

    # All analysis, scoring, retrieval context, and LLM calls below use masked text.
    profile = analyze_candidate_profile(masked_resume)
    score = calculate_resume_score(masked_resume, masked_job_description)
    ats_result = ats_optimization(masked_resume, masked_job_description)

    uploaded_score = round(min(10.0, score), 1)
    # Reuse a stronger saved result only when both source documents match.
    existing_log = find_matching_logged_resume(job_description, resume_text)
    if (
        existing_log
        and score_generated_resume(
            existing_log.get("generated_resume") or existing_log.get("final_response") or {},
            masked_job_description,
        )
        > uploaded_score
    ):
        generated_resume = finalize_generated_resume(
            existing_log.get("generated_resume") or existing_log.get("final_response") or {},
            candidate_name,
            resume_text,
        )
        generated_resume = ensure_improved_resume_score(
            generated_resume, masked_resume, masked_job_description, uploaded_score
        )
        review = existing_log.get("review", review_resume_output(generated_resume))
        improved_score = score_generated_resume(generated_resume, masked_job_description)
        response = {
            "request_id": existing_log.get("request_id", request_id),
            "status": "Completed",
            "message": "Improved Resume already exists for the job description",
            "candidate_name": candidate_name,
            "user_id": request.user_id,
            "pii_detected": pii_status,
            "rag_context_summary": rag_context[:1200],
            "profile_analysis": existing_log.get("profile_analysis", profile),
            "uploaded_resume_score_out_of_10": uploaded_score,
            "resume_score_out_of_10": uploaded_score,
            "improved_resume_score_out_of_10": improved_score,
            "ats_optimization": existing_log.get("ats_optimization", ats_result),
            "decision": "Reuse existing improved resume",
            "generated_resume": generated_resume,
            "review": review,
            "crewai": existing_log.get("crewai", {"crew_status": "Recovered from previous log", "notes": "Agent output was reused from the matching job description log."}),
            "used_existing_log": True,
        }
        return response

    decision = "Reuse existing resume"
    if uploaded_score < 7.0 or ats_result["missing_keywords"]:
        decision = "Generate improved resume"

    # Generate first, then restore source-controlled fields and score safeguards.
    generated_resume = generate_resume_content(
        profile,
        masked_job_description,
        masked_resume,
        base_score=uploaded_score,
    )
    generated_resume = finalize_generated_resume(generated_resume, candidate_name, resume_text)
    generated_resume = ensure_improved_resume_score(
        generated_resume, masked_resume, masked_job_description, uploaded_score
    )
    review = review_resume_output(generated_resume)

    if ats_result.get("recommended_keywords"):
        ats_result["recommended_keywords"] = ats_result["recommended_keywords"][:5]

    crew_result = run_crewai_workflow(masked_resume, masked_job_description, rag_context)
    improved_score = score_generated_resume(generated_resume, masked_job_description)

    final_result = {
        "request_id": request_id,
        "status": "Completed",
        "candidate_name": candidate_name,
        "user_id": request.user_id,
        "pii_detected": pii_status,
        "rag_context_summary": rag_context[:1200],
        "profile_analysis": profile,
        "uploaded_resume_score_out_of_10": uploaded_score,
        "resume_score_out_of_10": uploaded_score,
        "improved_resume_score_out_of_10": improved_score,
        "ats_optimization": ats_result,
        "decision": decision,
        "generated_resume": generated_resume,
        "review": review,
        "crewai": crew_result,
    }

    log_entry = {
        "request_id": request_id,
        "timestamp": str(datetime.datetime.now()),
        "candidate_name": candidate_name,
        "user_id": request.user_id,
        "decision": decision,
        "resume_text": resume_text,
        "job_description": job_description,
        "profile_analysis": profile,
        "uploaded_resume_score_out_of_10": uploaded_score,
        "improved_resume_score_out_of_10": improved_score,
        "resume_score_out_of_10": uploaded_score,
        "ats_optimization": ats_result,
        "pii_detected": pii_status,
        "review": review,
        "generated_resume": generated_resume,
        "final_response": generated_resume,
        "crewai": crew_result,
        "status": "Completed",
    }

    # JSON provides a human-readable audit trail; Chroma accelerates later reuse.
    store_audit_log(log_entry)
    add_resume_log_to_vector_store(log_entry)

    return final_result


@app.get("/")
def root() -> Dict[str, str]:
    """Return a simple API discovery message."""

    return {"message": "Welcome to the Enterprise AI Resume Generator Agent."}


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Report readiness for UI and deployment health checks."""

    return {"status": "healthy", "message": "Resume Analyzer services are running."}


@app.post("/support")
def support_query(request: UserSupport) -> Dict[str, Any]:
    """Adapt free-form support text to the standard resume workflow."""

    return process_resume_workflow(
        ResumeRequest(
            candidate_name="User",
            resume_text=request.user_input,
            job_description="Generic enterprise role. Candidate should demonstrate skills, experience, projects, education, certifications, and ATS keywords.",
            user_id="anonymous",
        )
    )


@app.post("/resume-analyzer")
def analyze_resume(request: ResumeRequest) -> Dict[str, Any]:
    """Analyze and improve a resume against a target job description."""

    return process_resume_workflow(request)


@app.post("/resume-generator")
def generate_resume(request: ResumeRequest) -> Dict[str, Any]:
    """Alias the shared workflow for clients using a generation endpoint."""

    return process_resume_workflow(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ResumeAnalyzer:app", host="0.0.0.0", port=8000, reload=True)
