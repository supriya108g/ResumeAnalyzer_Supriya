# Enterprise AI Resume Generator

## Implementation Plan

**Version:** 1.0  
**Project type:** Python web application  
**Frontend:** Streamlit  
**Backend:** FastAPI  
**AI services:** OpenAI and CrewAI  
**Data stores:** JSON audit log and ChromaDB

## 1. Objective

The objective is to implement an AI-assisted resume analyzer that compares an uploaded resume with a job description, calculates ATS-oriented scores, generates an improved resume, exposes specialized CrewAI agent analysis, and allows the result to be downloaded as DOCX or PDF.

The implementation must preserve candidate facts, protect supported personally identifiable information (PII), prevent inappropriate reuse of another resume's output, and stop with a clear message when the OpenAI API key is not configured.

## 2. Scope

### In Scope

- Upload resumes and job descriptions in TXT, PDF, and DOCX formats.
- Extract text, including DOCX tables, headers, footers, and text boxes.
- Detect candidate name, education, skills, and experience.
- Detect prompt-injection phrases.
- Detect and mask supported PII before AI processing.
- Calculate uploaded and improved resume scores.
- Generate ATS-friendly resume content through OpenAI.
- Run four sequential CrewAI agents.
- Build RAG context from current inputs and previous analyses.
- Store audit data in JSON and searchable records in ChromaDB.
- Reuse previous output only when both resume and job description match.
- Display results and individual agent outputs in Streamlit.
- Export the improved resume as DOCX and PDF.
- Provide automated unit and workflow tests.

### Out of Scope

- Production authentication and user management.
- Cloud deployment and managed databases.
- Commercial ATS integration.
- OCR for image-only resumes.
- Automated verification of every factual claim made by an LLM.
- Multi-tenant data isolation.

## 3. Implementation Approach

The project will use an incremental, test-driven approach:

1. Establish the FastAPI and Streamlit application foundations.
2. Implement deterministic parsing, security, and scoring first.
3. Add OpenAI generation behind a required environment variable.
4. Add CrewAI orchestration and RAG retrieval.
5. Add persistence and safe result reuse.
6. Build result rendering, navigation, and downloads.
7. Add regression tests for real resume layouts.
8. Validate and package the final assignment.

Deterministic controls will surround AI output. Candidate name and education will be restored from uploaded source text after generation so model placeholders do not become final resume facts.

## 4. Work Breakdown

### Phase 1: Project Foundation

**Tasks**

- Create the Python project structure.
- Define dependencies in `requirements.txt`.
- Configure environment loading with `python-dotenv`.
- Create the FastAPI application.
- Create the Streamlit application.
- Add a FastAPI health endpoint.
- Define local ports: Streamlit `8501`, FastAPI `8001`.

**Deliverables**

- `ResumeAnalyzer.py`
- `streamlit_app.py`
- `requirements.txt`
- Working `/health` endpoint

**Acceptance Criteria**

- FastAPI starts successfully on port `8001`.
- Streamlit starts successfully on port `8501`.
- Streamlit can reach the backend health service.

### Phase 2: File Upload and Text Extraction

**Tasks**

- Add Streamlit upload controls for resume and job description.
- Support TXT files with UTF-8 replacement handling.
- Support PDF files with `pypdf`.
- Support DOCX files through WordprocessingML extraction.
- Include DOCX paragraphs, tables, text boxes, headers, and footers.
- Reject unsupported file types with a clear message.
- Validate that both extracted documents contain content.

**Deliverables**

- Reusable file-reading functions.
- Two-column upload form.
- Input validation messages.

**Acceptance Criteria**

- Text can be extracted from all three supported formats.
- Education and other content inside DOCX text boxes or tables are included.
- Analysis is not submitted when either document is empty.

### Phase 3: Input Validation, Security, and Privacy

**Tasks**

- Define Pydantic request models.
- Require at least 10 characters for resume and job-description text.
- Detect common prompt-injection phrases.
- Detect supported PII categories.
- Mask email addresses.
- Mask LinkedIn profile URLs.
- Mask domestic and international phone numbers.
- Mask potential eight-digit account numbers.
- Use masked text for scoring, retrieval context, OpenAI, and CrewAI.
- Return only PII category flags in the API response.

**Deliverables**

- Prompt-injection guard.
- `pii_found` function.
- `mask_pii` function.
- Blocked-request response contract.

**Acceptance Criteria**

- Suspicious input is blocked before AI processing.
- Supported PII values do not appear in model-facing text.
- PII masking tests pass for email, phone, LinkedIn, and account patterns.

### Phase 4: Resume Parsing and Candidate Profiling

**Tasks**

- Build a domain-specific skill dictionary.
- Extract recognized multiword and single-word skills.
- Estimate years of experience from explicit statements and employment ranges.
- Merge overlapping employment ranges to avoid double counting.
- Determine candidate level: Entry-Level, Junior, Mid-Level, or Senior.
- Determine the primary professional domain.
- Extract a candidate name while rejecting headings, job titles, technologies, and template placeholders.
- Extract education until the next recognized section heading.
- Support multiline and compact single-line education formats.

**Deliverables**

- Candidate profile structure.
- Name extraction function.
- Education extraction function.
- Experience calculation function.
- Domain and skill extraction functions.

**Acceptance Criteria**

- Missing names remain blank.
- Values such as `FirstName LastName` and `Street Address` are not accepted as names.
- Education is copied from the uploaded resume.
- Education extraction stops before unrelated sections.
- Candidate domains are correctly identified for covered examples.

### Phase 5: ATS and Resume Scoring

**Tasks**

- Extract resume and job-description skill sets.
- Calculate the base skill-overlap score.
- Add bounded experience, project, and education bonuses.
- Cap all scores at `10.0`.
- Round scores to one decimal place.
- Flatten generated resume sections for equivalent scoring.
- Add a generated-resume completeness bonus.
- Identify missing ATS phrases.
- Limit recommended keywords to a concise list.
- Preserve only job keywords supported by the uploaded resume.

**Deliverables**

- `calculate_resume_score`
- `score_generated_resume`
- `ats_optimization`
- Score-improvement guard

**Acceptance Criteria**

- Uploaded and generated resumes use the same core scoring method.
- Scores remain between `0.0` and `10.0`.
- The displayed improved score equals the score calculated from the final generated resume.
- Unsupported skills are not added solely to increase the score.

### Phase 6: OpenAI Resume Generation

**Tasks**

- Load `OPENAI_API_KEY` from the environment.
- Stop processing when the key is missing or blank.
- Return `Configuration Required` with an actionable message.
- Build a structured resume-generation prompt.
- Require JSON output with defined resume sections.
- Parse plain and fenced JSON responses.
- Retry once when generated content fails to improve the score.
- Restore candidate name and education from uploaded source data.
- Never return a fallback resume when the API key is absent.

**Deliverables**

- OpenAI client integration.
- Structured generation prompt.
- JSON parser.
- Missing-key response handling.

**Acceptance Criteria**

- No generated resume is returned without `OPENAI_API_KEY`.
- The UI displays the key-configuration message.
- Generated content follows the expected section structure.
- Candidate identity and education remain source-controlled.

### Phase 7: CrewAI Multi-Agent Workflow

**Tasks**

- Create the Profile Analyzer Agent.
- Create the ATS Optimization Agent.
- Create the Resume Writer Agent.
- Create the Reviewer Agent.
- Define one JSON-oriented task per agent.
- Run tasks with `Process.sequential`.
- Normalize each task output into a dictionary.
- Return individual agent outputs to the UI.
- Handle missing CrewAI dependencies and runtime failures gracefully.

**Agent Responsibilities**

| Agent | Responsibility |
|---|---|
| Profile Analyzer | Extract candidate level, domain, skills, education, and experience |
| ATS Optimization | Identify gaps, recommendations, and ATS readiness |
| Resume Writer | Draft ATS-friendly resume sections |
| Reviewer | Evaluate grammar, structure, and candidate readiness |

**Deliverables**

- CrewAI agents and tasks.
- Sequential crew configuration.
- Agent output response structure.
- Agent Analysis UI view.

**Acceptance Criteria**

- Four agents run in the intended order.
- Each agent output can be viewed separately.
- Agent failure does not crash the FastAPI service.

### Phase 8: RAG, Audit, and ChromaDB

**Tasks**

- Build an in-memory lexical knowledge store.
- Add current resume and job description to request context.
- Add relevant historical records to context.
- Create a deterministic local hashing vector.
- Configure ChromaDB persistent storage at `resume_chroma_db_v3`.
- Store compact searchable analysis payloads.
- Append complete records to `audit_log.json`.
- Fall back to lexical audit-log search when ChromaDB is unavailable.
- Require both job and resume similarity before reuse.
- Reuse only a saved result that scores higher than the current upload.

**Deliverables**

- RAG context builder.
- ChromaDB collection.
- JSON audit storage.
- Saved-result matching logic.

**Acceptance Criteria**

- ChromaDB is recreated automatically when the backend starts.
- The application remains usable when ChromaDB is unavailable.
- A result is not reused for a different candidate resume.
- Matching resume and job inputs may reuse a stronger saved result.

### Phase 9: Streamlit Results and Navigation

**Tasks**

- Display uploaded and improved scores.
- Display decision and workflow status.
- Render generated summary, skills, experience, projects, and education.
- Hide invalid candidate names.
- Store the current result in Streamlit session state.
- Track selected files with an input fingerprint.
- Clear stale results when files change.
- Preserve results when navigating to Agent Analysis and back.
- Save the latest response to `latest_analysis.json`.

**Deliverables**

- Result Summary view.
- Generated Resume view.
- Agent Analysis view.
- Session-state navigation.

**Acceptance Criteria**

- Returning from Agent Analysis does not erase the current result.
- Selecting different files clears the previous displayed result.
- Missing-key responses show an error rather than a success message.

### Phase 10: Resume Export

**Tasks**

- Build a plain-text representation of the generated resume.
- Generate a DOCX file with headings and bullet lists.
- Generate a PDF file with page overflow handling.
- Use the candidate name in filenames only when valid.
- Use `improved_resume` as the fallback filename.

**Deliverables**

- DOCX download.
- PDF download.

**Acceptance Criteria**

- Both downloads are generated from the final displayed resume.
- Exported education matches the uploaded source resume.
- Blank or invalid names do not appear in download filenames.

### Phase 11: Testing and Quality Assurance

**Tasks**

- Test missing OpenAI key behavior.
- Test score consistency and improvement.
- Test source-controlled name and education.
- Test placeholder-name rejection.
- Test PII masking.
- Test healthcare and finance domain classification.
- Test multiline and inline education sections.
- Test saved-result reuse and candidate isolation.
- Test real-world retail resume layouts.
- Compile all Python files.
- Verify FastAPI health and Streamlit availability.

**Deliverables**

- `test_resume_analyzer.py`
- Passing pytest report.
- Syntax compilation validation.

**Acceptance Criteria**

- All 21 automated tests pass.
- Project Python files compile successfully.
- No unresolved errors exist in project-owned UI or test files.

### Phase 12: Documentation and Submission Preparation

**Tasks**

- Add code comments and function docstrings.
- Prepare architecture and design documentation.
- Prepare this implementation plan.
- Blank the API key in `.env`.
- Remove generated ChromaDB folders.
- Remove virtual environments, caches, temporary files, and candidate data from the submission.
- Verify dependency and startup instructions.

**Deliverables**

- `ARCHITECTURE_AND_DESIGN.md`
- `IMPLEMENTATION_PLAN.md`
- Clean assignment package

**Acceptance Criteria**

- No API key is included in submitted files.
- No real candidate PII is included in submitted audit/cache files.
- The package contains source, requirements, tests, and documentation.
- A grader can install, configure, run, and test the project using documented commands.

## 5. Suggested Schedule

| Day | Planned Work | Main Deliverable |
|---:|---|---|
| 1 | Project setup and FastAPI/Streamlit foundations | Running frontend and backend |
| 2 | File extraction and validation | TXT, PDF, DOCX ingestion |
| 3 | Security, PII masking, and parsing | Safe deterministic input pipeline |
| 4 | Candidate profiling and scoring | Profile and ATS score output |
| 5 | OpenAI generation | Structured improved resume |
| 6 | CrewAI workflow | Four sequential agent outputs |
| 7 | RAG, audit, and ChromaDB | Historical context and result reuse |
| 8 | Streamlit results, state, and downloads | Complete user workflow |
| 9 | Tests and real-layout regressions | Passing automated suite |
| 10 | Documentation and packaging | Submission-ready project |

This schedule is illustrative and can be adjusted to assignment deadlines.

## 6. Dependencies

| Dependency | Purpose |
|---|---|
| FastAPI | REST API service |
| Pydantic | Request validation |
| Uvicorn | ASGI server |
| Streamlit | Web user interface |
| Requests | Streamlit-to-FastAPI communication |
| OpenAI | Resume generation |
| CrewAI | Multi-agent orchestration |
| ChromaDB | Persistent similarity retrieval |
| pypdf | PDF text extraction |
| python-docx | DOCX export |
| ReportLab | PDF export |
| python-dotenv | Environment loading |
| pytest | Automated testing |

`faiss-cpu` and `numpy` are optional in the current implementation. The active Chroma retrieval path uses deterministic local hashing vectors.

## 7. Environment and Startup Plan

### Environment Configuration

Create or update `.env`:

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

The user must provide a valid key before analysis can proceed. The key must remain blank in the submitted assignment.

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Start FastAPI

```powershell
python -m uvicorn ResumeAnalyzer:app --host 127.0.0.1 --port 8001
```

### Start Streamlit

```powershell
python -m streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

### Run Tests

```powershell
python -m pytest test_resume_analyzer.py -q
```

## 8. Validation Plan

### Functional Validation

- Upload each supported file format.
- Confirm extracted name and education match the source.
- Analyze a resume against a relevant job description.
- Confirm both scores are displayed.
- Confirm generated sections are complete.
- Open Agent Analysis and return without losing the result.
- Download both DOCX and PDF versions.

### Security Validation

- Submit a prompt-injection phrase and confirm blocking.
- Submit email, phone, LinkedIn, and account patterns and confirm masking.
- Remove the OpenAI key and confirm no fallback resume is generated.
- Confirm a saved result is not reused for a different resume.

### Persistence Validation

- Start without a Chroma directory and confirm `resume_chroma_db_v3` is created.
- Complete an analysis and confirm audit and vector records are written.
- Repeat matching inputs and confirm valid reuse.
- Disable ChromaDB and confirm lexical retrieval remains available.

### Submission Validation

- Confirm `.env` contains no key value.
- Confirm generated databases and caches are removed.
- Confirm no real candidate data remains in JSON files.
- Install dependencies in a clean virtual environment.
- Run the test suite and startup commands from the documentation.

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Missing or invalid OpenAI key | Analysis cannot continue | Fail fast with an actionable UI message |
| LLM returns invalid JSON | Generated content cannot be parsed | Strip code fences, use controlled parsing, and retry once |
| LLM invents identity or education | Inaccurate resume | Restore these fields from uploaded source text |
| DOCX content exists in text boxes | Important text is missed | Read WordprocessingML directly |
| Previous result belongs to another resume | Incorrect candidate output | Require both resume and job similarity |
| ChromaDB unavailable | Vector retrieval fails | Fall back to lexical audit-log search |
| Audit log contains PII | Privacy exposure | Exclude real logs from submission; use encrypted/redacted storage in production |
| Prompt injection bypasses keywords | Model manipulation | Treat keyword detection as baseline and add stronger classifiers for production |
| Heuristic score differs from commercial ATS | Misleading expectations | Label it as an internal alignment score |
| Generated local files inflate submission | Larger or sensitive package | Remove Chroma, cache, audit, and temporary artifacts |

## 10. Definition of Done

The implementation is complete when:

- FastAPI and Streamlit start using the documented commands.
- Resume and job-description uploads work for TXT, PDF, and DOCX.
- Missing OpenAI configuration prevents generation and shows a clear message.
- Prompt-injection and PII controls run before AI processing.
- Uploaded and improved scores are calculated and displayed.
- Candidate name and education are preserved from source documents.
- Four CrewAI agent outputs are available in Agent Analysis.
- Matching historical results can be safely reused.
- DOCX and PDF downloads are available.
- Navigation preserves the current result.
- All 21 tests pass.
- The API key is blank and generated/sensitive artifacts are excluded from submission.
- Architecture, design, implementation, startup, and testing documentation are included.
