# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from resume_processor import ResumeProcessor
from model import rank_candidate, batch_rank
import uvicorn
import json

app = FastAPI(title="Resume Screener API")

processor = ResumeProcessor()

@app.post("/parse_resume")
async def parse_resume(file: UploadFile = File(...)):
    content = await file.read()
    parsed = processor.parse_resume(content)
    return JSONResponse(parsed)

@app.post("/score_candidate")
async def score_candidate(file: UploadFile = File(...), job_profile_text: str = Form(...), job_skills: Optional[str] = Form(None)):
    # job_skills: optional comma-separated string
    content = await file.read()
    parsed = processor.parse_resume(content)
    js = [s.strip() for s in job_skills.split(',')] if job_skills else []
    result = rank_candidate(parsed['text'], parsed['skills'], job_profile_text, js)
    resp = {
        "parsed": parsed,
        "ranking": result
    }
    return JSONResponse(resp)

@app.post("/batch_score")
async def batch_score(files: List[UploadFile] = File(...), job_profile_text: str = Form(...), job_skills: Optional[str] = Form(None)):
    job_skills_list = [s.strip() for s in job_skills.split(',')] if job_skills else []
    candidates = []
    for i, f in enumerate(files):
        content = await f.read()
        parsed = processor.parse_resume(content)
        candidates.append({"id": f.filename or i, "text": parsed['text'], "skills": parsed['skills'], "parsed": parsed})
    ranking = batch_rank(candidates, job_profile_text, job_skills_list)
    # attach parsed info
    id_to_parsed = {c['id']: c['parsed'] for c in candidates}
    final = []
    for r in ranking:
        pid = r['id']
        final.append({
            "id": pid,
            "parsed": id_to_parsed.get(pid, {}),
            "ranking": r
        })
    return JSONResponse({"results": final})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
