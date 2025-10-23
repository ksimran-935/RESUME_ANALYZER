# model.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict

# Load model once
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_text(texts: List[str]):
    # returns numpy array of embeddings
    return embed_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def semantic_similarity_score(emb1, emb2):
    # emb1 and emb2 are sentence-transformers tensors
    score = util.cos_sim(emb1, emb2).item()
    # cos_sim returns in [-1,1] — convert to [0,1]
    return (score + 1) / 2

def keyword_overlap_score(candidate_skills: List[str], job_skills: List[str]):
    if not job_skills:
        return 0.0
    cand = set([s.lower() for s in candidate_skills])
    job = set([s.lower() for s in job_skills])
    overlap = cand.intersection(job)
    return len(overlap) / max(1, len(job))

def rank_candidate(resume_text: str,
                   candidate_skills: List[str],
                   job_profile_text: str,
                   job_skills: List[str],
                   weights: Dict[str, float] = None):
    """
    Returns a composite score and sub-scores.
    weights: {"semantic":0.6, "keyword":0.4}
    """
    if weights is None:
        weights = {"semantic": 0.6, "keyword": 0.4}
    emb_cand = embed_text([resume_text])[0]
    emb_job = embed_text([job_profile_text])[0]
    sem = semantic_similarity_score(emb_cand, emb_job)
    key = keyword_overlap_score(candidate_skills, job_skills)
    score = weights["semantic"] * sem + weights["keyword"] * key
    return {
        "composite_score": float(score),
        "semantic_score": float(sem),
        "keyword_score": float(key)
    }

def batch_rank(candidates: List[Dict], job_profile_text: str, job_skills: List[str], weights=None):
    # candidates: list of {"id":..., "text":..., "skills": [...]}
    # For efficiency, compute embeddings in batch
    texts = [c['text'] for c in candidates]
    emb_cands = embed_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    emb_job = embed_model.encode([job_profile_text], convert_to_tensor=True, show_progress_bar=False)[0]
    results = []
    for i, c in enumerate(candidates):
        sem = util.cos_sim(emb_cands[i], emb_job).item()
        sem = (sem + 1) / 2
        key = keyword_overlap_score(c.get('skills', []), job_skills)
        w = weights if weights else {"semantic": 0.6, "keyword": 0.4}
        score = w["semantic"] * sem + w["keyword"] * key
        results.append({
            "id": c.get("id", i),
            "composite_score": float(score),
            "semantic_score": float(sem),
            "keyword_score": float(key)
        })
    # sort descending by composite_score
    results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
    return results
