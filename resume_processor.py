# resume_processor.py
from typing import Dict, List
import re
from io import BytesIO
from PyPDF2 import PdfReader
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy model globally to reuse
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    # User must run: python -m spacy download en_core_web_sm
    raise RuntimeError("spaCy model not found. Run: python -m spacy download en_core_web_sm") from e

# A starter skill list — extend this with corporate skill lexicon for better recall
DEFAULT_SKILLS = [
    "python", "java", "c++", "c", "javascript", "react", "angular", "node.js", "django",
    "flask", "fastapi", "sql", "postgresql", "mongodb", "nosql", "aws", "azure", "gcp",
    "docker", "kubernetes", "ci/cd", "git", "nlp", "transformers", "spaCy", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "data analysis", "deep learning", "computer vision",
    "rest api", "microservices", "unix", "linux", "html", "css"
]

class ResumeProcessor:
    def __init__(self, extra_skills: List[str] = None):
        self.skill_terms = list(DEFAULT_SKILLS)
        if extra_skills:
            self.skill_terms.extend(extra_skills)
        # build phrase matcher for skills
        self.phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(s) for s in set(self.skill_terms)]
        self.phrase_matcher.add("SKILL", patterns)

    def pdf_to_text(self, file_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                # fallback: skip page if extraction fails
                continue
        text = "\n".join(text_parts)
        # clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = nlp(text)
        entities = {"PERSON": [], "ORG": [], "GPE": [], "EDUCATION": [], "EMAIL": [], "PHONE": []}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # simple heuristics for education (degree keywords)
        ed_patterns = ["bachelor", "master", "b.sc", "b.e", "m.sc", "m.e", "mba", "phd", "btech", "mtech"]
        educations = []
        text_lower = text.lower()
        for p in ed_patterns:
            if p in text_lower:
                educations.append(p)
        entities["EDUCATION"] = list(set(educations))

        # emails & phones via regex
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        phones = re.findall(r'(\+?\d{1,3}[-.\s]?)?(\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})', text)
        phones_flat = []
        for p in phones:
            phones_flat.append(''.join(p))
        entities["EMAIL"] = list(set(emails))
        entities["PHONE"] = list(set(phones_flat))
        return entities

    def extract_skills(self, text: str) -> List[str]:
        doc = nlp(text)
        matches = self.phrase_matcher(doc)
        found = set()
        for _, start, end in matches:
            span = doc[start:end].text
            found.add(span.lower())
        # also check common patterns like "Experience in X" or comma-separated lists after "Skills"
        # naive parse of "Skills:" lines
        skills_from_sections = self._extract_skills_from_sections(text)
        found.update([s.lower() for s in skills_from_sections])
        return sorted(found)

    def _extract_skills_from_sections(self, text: str) -> List[str]:
        found = []
        # split into lines and find lines with 'skill' or 'technical'
        lines = re.split(r'[\r\n]+', text)
        for line in lines:
            lower = line.lower()
            if 'skill' in lower or 'technical' in lower or 'technologies' in lower:
                # take commas or pipes separated items
                parts = re.split(r'[,:|\-–]', line)
                for p in parts[1:]:
                    for token in p.split(','):
                        token = token.strip()
                        if len(token) > 1 and len(token.split()) <= 4:
                            found.append(token)
            # quick list detection: many commas and known skill words
            if ',' in line and len(line) < 200:
                if any(s in lower for s in ['python', 'java', 'sql', 'aws', 'docker', 'react']):
                    tokens = [t.strip() for t in line.split(',')]
                    found.extend(tokens)
        # sanitize
        cleaned = []
        for s in found:
            s = re.sub(r'[^A-Za-z0-9\.\+\#\s\/\-]', '', s).strip()
            if s:
                cleaned.append(s)
        return cleaned

    def parse_resume(self, file_bytes: bytes) -> Dict:
        text = self.pdf_to_text(file_bytes)
        entities = self.extract_entities(text)
        skills = self.extract_skills(text)
        # simple experience extraction: look for "X years" pattern
        exp = re.findall(r'(\d+)\+?\s+(?:years|yrs)\s+of', text.lower())
        total_experience = None
        if exp:
            try:
                total_experience = max(int(x) for x in exp)
            except Exception:
                total_experience = None

        return {
            "text": text,
            "entities": entities,
            "skills": skills,
            "approx_experience_years": total_experience
        }
