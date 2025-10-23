# streamlit_dashboard.py
import streamlit as st
import requests
import pandas as pd
from io import BytesIO

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Resume Screener Dashboard", layout="wide")

st.title("Intelligent Resume Screener — HR Dashboard")

st.sidebar.header("Job Profile")
job_title = st.sidebar.text_input("Job Title", "NLP Engineer")
job_description = st.sidebar.text_area("Job Description (paste text)", height=200,
                                       value="We are looking for an NLP Engineer with experience in spaCy, Transformers, PyTorch, and deploying REST APIs.")
job_skills = st.sidebar.text_input("Job Skills (comma separated)", "nlp, transformers, python, pytorch, fastapi")

st.sidebar.markdown("---")
st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload multiple resume PDFs", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Run Screening"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one resume PDF.")
    else:
        with st.spinner("Scoring candidates..."):
            files = []
            for f in uploaded_files:
                files.append(('files', (f.name, f.getvalue(), 'application/pdf')))
            data = {
                'job_profile_text': job_description,
                'job_skills': job_skills
            }
            resp = requests.post(API_URL + "/batch_score", files=files, data=data)
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} {resp.text}")
            else:
                results = resp.json()['results']
                # build table
                rows = []
                for r in results:
                    pid = r['id']
                    parsed = r['parsed']
                    ranking = r['ranking']
                    rows.append({
                        "id": pid,
                        "score": ranking['composite_score'],
                        "semantic": ranking['semantic_score'],
                        "keyword": ranking['keyword_score'],
                        "top_skills": ", ".join(parsed.get('skills', [])[:6]),
                        "experience_years": parsed.get('approx_experience_years', "")
                    })
                df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
                st.subheader("Shortlisted Candidates")
                st.dataframe(df)

                st.markdown("### Detailed Candidate View")
                sel = st.selectbox("Select candidate", df['id'].tolist())
                detail = next((r for r in results if r['id'] == sel), None)
                if detail:
                    parsed = detail['parsed']
                    ranking = detail['ranking']
                    st.write("**Composite score:**", ranking['composite_score'])
                    st.write("**Semantic similarity:**", ranking['semantic_score'])
                    st.write("**Keyword overlap:**", ranking['keyword_score'])
                    st.write("**Extracted skills:**", parsed.get('skills', []))
                    st.write("**Contacts:**", parsed.get('entities', {}).get('EMAIL', []), parsed.get('entities', {}).get('PHONE', []))
                    st.write("**Education heuristics:**", parsed.get('entities', {}).get('EDUCATION', []))
                    st.write("**Resume text (first 1000 chars):**")
                    st.code(parsed.get('text', '')[:1000])
