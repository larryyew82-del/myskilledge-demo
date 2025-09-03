# MySkillEdge – Analytics + Recommendations (Streamlit or CLI)
# -----------------------------------------------------------------
# This file can run in two modes:
# 1) Streamlit UI (if `streamlit` is available) – ideal for Streamlit Cloud.
# 2) CLI demo (no Streamlit installed) – prints JSON to stdout so it runs
#    in restricted/sandbox environments that don’t allow adding packages.
#
# Fixes included in this version:
# - Optional Streamlit import (prevents ModuleNotFoundError in sandbox).
# - Proper `global` placement for vectorizer rebuild.
# - **NEW**: Convert np.matrix → NumPy ndarray in learner profile vectors to
#   avoid: "TypeError: np.matrix is not supported" in sklearn cosine_similarity.
# - Added extra tests around vector shape and cosine call.

from __future__ import annotations
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Optional Streamlit import with safe fallback
# ---------------------------
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:  # pragma: no cover
    STREAMLIT_AVAILABLE = False

    class _DummyStreamlit:
        """Minimal shim for decorators so the logic layer works without UI."""
        def cache_data(self, *args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator

        def cache_resource(self, *args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator

    st = _DummyStreamlit()  # type: ignore

# =============================================================
# Configuration
# =============================================================
DEFAULT_ALPHA = 0.5  # hybrid mixing factor α
UNFINISHED_BOOST_RANGE = (0.4, 0.8)
LOW_RATING_THRESHOLD = 3.5
LOW_COMPLETION_THRESHOLD = 0.40

if STREAMLIT_AVAILABLE:  # UI-only config
    st.set_page_config(page_title="MySkillEdge Demo (Analytics + Recommendations)", layout="wide")

# =============================================================
# Synthetic Mock Data (idempotent)
# =============================================================
@st.cache_data(show_spinner=False)
def seed_data() -> Dict[str, pd.DataFrame]:
    # Users
    users = pd.DataFrame([
        {"user_id": 1, "name": "Amir Hakim", "role": "learner", "company_id": 1},
        {"user_id": 2, "name": "Aisha Tan",  "role": "learner", "company_id": 1},
        {"user_id": 3, "name": "Dr. Tan",    "role": "trainer", "company_id": None},
        {"user_id": 4, "name": "Rahman HR",  "role": "employer","company_id": 1},
        {"user_id": 5, "name": "Noraini",    "role": "admin",   "company_id": None},
    ])

    # Companies
    companies = pd.DataFrame([
        {"company_id": 1, "name": "Penang E&E Sdn Bhd"},
    ])

    # Skills universe
    skills = [
        "python","sql","ml","excel","logistics","privacy","health","dashboards",
        "statistics","pandas","nlp","cv","cloud","etl","javascript","powerbi"
    ]

    # Courses (with ratings & historical completion_rate for quality prior)
    courses = pd.DataFrame([
        {"course_id": 101, "title": "AI for E&E",                "description": "Intro to AI in Electronics",   "skills": "python, statistics, pandas",         "tags": "ai, electronics",     "rating": 4.6, "completion_rate": 0.62},
        {"course_id": 102, "title": "Applied Machine Learning",  "description": "Regression & classification",   "skills": "ml, python, pandas",                "tags": "ml, model",           "rating": 4.4, "completion_rate": 0.58},
        {"course_id": 103, "title": "Logistics Optimization",    "description": "Routing & WMS",                "skills": "excel, logistics",                   "tags": "ops, supply",         "rating": 4.0, "completion_rate": 0.55},
        {"course_id": 104, "title": "Healthcare Data",           "description": "Privacy and dashboards",       "skills": "health, privacy, dashboards",        "tags": "health, bi",          "rating": 3.8, "completion_rate": 0.44},
        {"course_id": 105, "title": "Practical SQL",             "description": "Joins, windows, perf",         "skills": "sql, etl",                           "tags": "data, warehouse",     "rating": 4.7, "completion_rate": 0.69},
        {"course_id": 106, "title": "Cloud Data Engineering",    "description": "Pipelines, ETL",               "skills": "cloud, etl, python",                "tags": "gcp, pipelines",      "rating": 4.2, "completion_rate": 0.48},
        {"course_id": 107, "title": "Data Dashboards",           "description": "PowerBI and JS basics",        "skills": "dashboards, powerbi, javascript",   "tags": "bi, viz",             "rating": 3.3, "completion_rate": 0.35},  # deliberately low
    ])

    # Course skill mapping table format (explode skills)
    course_skills = courses.assign(skill_list=courses.skills.str.split(",")).explode("skill_list")
    course_skills["skill_list"] = course_skills["skill_list"].str.strip()
    course_skills = course_skills[["course_id","skill_list"]].rename(columns={"skill_list":"skill"})

    # Enrollments (progress in 0..1, quiz_score optional)
    enrollments = pd.DataFrame([
        {"user_id": 1, "course_id": 101, "progress": 0.60, "quiz_score": 75},
        {"user_id": 1, "course_id": 102, "progress": 0.20, "quiz_score": None},
        {"user_id": 1, "course_id": 103, "progress": 1.00, "quiz_score": 82},
        {"user_id": 2, "course_id": 103, "progress": 1.00, "quiz_score": 88},
        {"user_id": 2, "course_id": 104, "progress": 0.45, "quiz_score": 65},
    ])

    # Events (views & quizzes) to compute engagement
    np.random.seed(42)
    events_rows = []
    for u in users.user_id:
        for cid in courses.course_id:
            if np.random.rand() < 0.25:  # some interactions
                for _ in range(np.random.randint(1,4)):
                    dt = datetime.utcnow() - timedelta(days=np.random.randint(0, 45))
                    duration = float(np.random.uniform(60, 600))
                    is_quiz = np.random.rand() < 0.3
                    score = float(np.random.uniform(50, 100)) if is_quiz else None
                    events_rows.append({
                        "user_id": u,
                        "course_id": cid,
                        "event_type": "quiz" if is_quiz else "view",
                        "duration_seconds": duration,
                        "score": score,
                        "created_at": dt,
                    })
    events = pd.DataFrame(events_rows)

    # Roles & required skills (weights allowed)
    roles = pd.DataFrame([
        {"role_id": 201, "name": "Data Analyst",     "required_skills": {"python":1.0, "sql":1.0, "dashboards":0.7}},
        {"role_id": 202, "name": "ML Engineer",      "required_skills": {"python":1.0, "ml":1.0, "cloud":0.6}},
        {"role_id": 203, "name": "Ops (Logistics)", "required_skills": {"excel":1.0, "logistics":1.0}},
        {"role_id": 204, "name": "Health Analyst",  "required_skills": {"health":1.0, "privacy":0.8}},
    ])

    return {
        "users": users,
        "companies": companies,
        "skills": pd.DataFrame({"skill": skills}),
        "courses": courses,
        "course_skills": course_skills,
        "enrollments": enrollments,
        "events": events,
        "roles": roles,
    }

DATA = seed_data()

# =============================================================
# TF-IDF vectorizer (content-based)
# =============================================================
@st.cache_resource(show_spinner=False)
def build_vectorizer(courses: pd.DataFrame, course_skills: pd.DataFrame) -> Tuple[TfidfVectorizer, np.ndarray, List[int]]:
    """Build TF-IDF vectorizer + course matrix + ordered course_ids."""
    skill_map = course_skills.groupby("course_id")["skill"].apply(lambda s: " ".join(s)).to_dict()
    texts, ids = [], []
    for _, row in courses.iterrows():
        blob = " ".join([
            str(row.get("title","")),
            str(row.get("description","")),
            str(row.get("tags","")),
            skill_map.get(row.course_id, "")
        ])
        texts.append(blob)
        ids.append(int(row.course_id))
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X, ids

# Initial vectorizer
vectorizer, course_matrix, course_ids = build_vectorizer(DATA["courses"], DATA["course_skills"])

# Proper global update helper (prevents global-declaration SyntaxError)
def rebuild_vectorizer(courses: pd.DataFrame, course_skills: pd.DataFrame) -> None:
    global vectorizer, course_matrix, course_ids
    vectorizer, course_matrix, course_ids = build_vectorizer(courses, course_skills)

# =============================================================
# Analytics helpers
# =============================================================

def learner_analytics(user_id: int) -> Dict[str, object]:
    enrolls = DATA["enrollments"][DATA["enrollments"].user_id == user_id]
    events = DATA["events"][DATA["events"].user_id == user_id]

    now = datetime.utcnow().date()
    last_30 = now - timedelta(days=30)
    active_days_last_30 = 0
    time_on_platform = 0.0
    avg_quiz = None

    if not events.empty:
        events = events.copy()
        events["day"] = events["created_at"].dt.date
        recent = events[events["day"] >= last_30]
        active_days_last_30 = int(recent["day"].nunique())
        time_on_platform = float(events["duration_seconds"].sum())
        quiz_scores = events.query("event_type == 'quiz'")[["score"]].dropna()
        avg_quiz = float(quiz_scores["score"].mean()) if not quiz_scores.empty else None

    courses_in_progress = int((enrolls["progress"] < 1.0).sum()) if not enrolls.empty else 0
    completion_rate = float((enrolls["progress"] >= 1.0).mean()) if not enrolls.empty else 0.0

    return {
        "user_id": user_id,
        "active_days_last_30": active_days_last_30,
        "time_on_platform_seconds": round(time_on_platform, 1),
        "courses_in_progress": courses_in_progress,
        "completion_rate": round(completion_rate, 3),
        "avg_quiz_score": None if avg_quiz is None else round(avg_quiz, 2),
    }


def company_analytics(company_id: int) -> Dict[str, object]:
    users = DATA["users"][DATA["users"].company_id == company_id]
    user_ids = users.user_id.tolist()
    events = DATA["events"][DATA["events"].user_id.isin(user_ids)]
    enrolls = DATA["enrollments"][DATA["enrollments"].user_id.isin(user_ids)]

    now = datetime.utcnow().date()
    last_1 = now - timedelta(days=1)
    last_30 = now - timedelta(days=30)

    dau = mau = 0
    if not events.empty:
        events = events.copy()
        events["day"] = events["created_at"].dt.date
        dau = int(events[events["day"] >= last_1]["user_id"].nunique())
        mau = int(events[events["day"] >= last_30]["user_id"].nunique())
    dau_mau = round((dau / mau) if mau else 0.0, 3)

    avg_completion = 0.0
    if not enrolls.empty:
        avg_completion = float((enrolls["progress"] >= 1.0).sum() / max(len(enrolls),1))

    top_skills: List[List[object]] = []
    heatmap: Dict[str, float] = {}
    if not events.empty:
        cids = events["course_id"].unique().tolist()
        cskills = DATA["course_skills"][DATA["course_skills"].course_id.isin(cids)]
        counts = cskills["skill"].value_counts().head(10)
        top_skills = list(map(lambda kv: [kv[0], int(kv[1])], counts.items()))

        completed = enrolls[enrolls["progress"] >= 1.0]
        skill_to_users: Dict[str, Set[int]] = {}
        for _, row in completed.iterrows():
            sk = DATA["course_skills"][DATA["course_skills"].course_id == row.course_id]["skill"].tolist()
            for sname in sk:
                skill_to_users.setdefault(sname, set()).add(int(row.user_id))
        N = max(len(user_ids), 1)
        for sname, users_set in skill_to_users.items():
            heatmap[sname] = round(len(users_set) / N, 3)

    return {
        "company_id": company_id,
        "dau": dau,
        "mau": mau,
        "dau_mau": dau_mau,
        "avg_completion_rate": round(avg_completion, 3),
        "top_skills_trained": top_skills,
        "skill_gap_heatmap": heatmap,
    }

# =============================================================
# Rule-based scoring
# =============================================================

def quality_prior(course_row: pd.Series) -> float:
    rating = float(course_row.get("rating", 4.0) or 4.0)
    comp = float(course_row.get("completion_rate", 0.6) or 0.6)
    if rating < LOW_RATING_THRESHOLD or comp < LOW_COMPLETION_THRESHOLD:
        q = max(0.0, min(1.0, (rating - 2.0)/3.0)) * 0.7 + max(0.0, min(1.0, comp)) * 0.3
    else:
        q = min(1.0, (rating/5.0)*0.7 + comp*0.3)
    return float(q)


def learner_verified_skills(user_id: int) -> Set[str]:
    out: Set[str] = set()
    completed = DATA["enrollments"][
        (DATA["enrollments"].user_id == user_id) & (DATA["enrollments"]["progress"] >= 1.0)
    ]
    for cid in completed["course_id"].tolist():
        out |= set(DATA["course_skills"][DATA["course_skills"].course_id == cid]["skill"].tolist())
    return out


def learner_missing_skills(user_id: int, target_role_id: int) -> Set[str]:
    role = DATA["roles"][DATA["roles"].role_id == target_role_id].iloc[0]
    required = set(role.required_skills.keys())
    have = learner_verified_skills(user_id)
    return required - have


def rule_score_for_user(user_id: int, course_id: int, target_role_id: Optional[int]) -> float:
    enrolls = DATA["enrollments"][DATA["enrollments"].user_id == user_id]
    course_row = DATA["courses"][DATA["courses"].course_id == course_id].iloc[0]

    if not enrolls.empty and course_id in enrolls[enrolls["progress"] >= 1.0]["course_id"].tolist():
        return 0.0

    score = 0.2

    row = enrolls[enrolls["course_id"] == course_id]
    if not row.empty:
        p = float(row.iloc[0]["progress"])
        if UNFINISHED_BOOST_RANGE[0] <= p <= UNFINISHED_BOOST_RANGE[1]:
            score += 0.4

    score += 0.3 * quality_prior(course_row)

    if target_role_id:
        role = DATA["roles"][DATA["roles"].role_id == target_role_id].iloc[0]
        role_skills = set(role.required_skills.keys())
        course_sk = set(DATA["course_skills"][DATA["course_skills"].course_id == course_id]["skill"].tolist())
        overlap = len(course_sk & role_skills)
        if overlap > 0:
            score += 0.3 + 0.1 * min(overlap, 3)

        miss = learner_missing_skills(user_id, target_role_id)
        if miss and len(course_sk & miss) > 0:
            score += 0.2

    return float(max(0.0, min(1.0, score)))

# =============================================================
# ML Scoring (TF-IDF)
# =============================================================

def _to_row_ndarray(x) -> np.ndarray:
    """Ensure output is a dense NumPy array with shape (1, n)."""
    # np.asarray converts np.matrix to ndarray; CSR/CSC means/np.matrix also handled
    arr = np.asarray(x).astype(float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    # Some libs return shape (1, n) already; ensure row-major
    return arr.reshape(1, -1)


def learner_profile_vector(user_id: int) -> np.ndarray:
    """Return a **(1, n)** dense ndarray representing the learner profile.

    We convert any np.matrix / sparse mean results to ndarray via `_to_row_ndarray`
    to satisfy sklearn's cosine_similarity requirements.
    """
    enrolls = DATA["enrollments"][DATA["enrollments"].user_id == user_id]
    liked = enrolls[enrolls["progress"] >= 0.5]["course_id"].tolist()
    if not liked:
        return _to_row_ndarray(course_matrix.mean(axis=0))  # cold start
    idxs = [course_ids.index(cid) for cid in liked if cid in course_ids]
    if not idxs:
        return _to_row_ndarray(course_matrix.mean(axis=0))
    return _to_row_ndarray(course_matrix[idxs].mean(axis=0))


def ml_scores_for_user(user_id: int) -> Dict[int, float]:
    prof = learner_profile_vector(user_id)  # guaranteed (1, n) ndarray
    sims = cosine_similarity(prof, course_matrix).flatten()
    mn, mx = sims.min(), sims.max()
    denom = (mx - mn) or 1.0
    sims = (sims - mn) / denom
    return {cid: float(sims[i]) for i, cid in enumerate(course_ids)}

# =============================================================
# Hybrid recommendations & Skill gap report
# =============================================================

def recommend_courses(user_id: int, k: int = 5, alpha: float = DEFAULT_ALPHA, target_role_id: Optional[int] = None) -> List[Dict[str, object]]:
    ml_scores = ml_scores_for_user(user_id)
    recs = []
    for cid in course_ids:
        r = rule_score_for_user(user_id, cid, target_role_id)
        m = ml_scores.get(cid, 0.0)
        score = float(alpha) * r + (1.0 - float(alpha)) * m
        cr = DATA["courses"][DATA["courses"].course_id == cid].iloc[0]
        recs.append({
            "course_id": cid,
            "title": cr.title,
            "rule": round(r,3),
            "ml": round(m,3),
            "score": round(score,3)
        })
    completed_ids = set(DATA["enrollments"][
        (DATA["enrollments"].user_id==user_id) & (DATA["enrollments"]["progress"]>=1.0)
    ]["course_id"].tolist())
    recs.sort(key=lambda x: (x["course_id"] in completed_ids, -x["score"]))
    return recs[:k]


def suggest_courses_for_skill(skill: str, completed_ids: Set[int]) -> List[Dict[str, object]]:
    rows = DATA["course_skills"][DATA["course_skills"].skill == skill]["course_id"].tolist()
    ranked = []
    for cid in rows:
        if cid in completed_ids:
            continue
        cr = DATA["courses"][DATA["courses"].course_id == cid].iloc[0]
        ranked.append({"course_id": cid, "quality": round(quality_prior(cr), 3), "title": cr.title})
    ranked.sort(key=lambda x: x["quality"], reverse=True)
    return ranked[:3]


def skill_gap_report(user_id: int, target_role_id: int) -> Dict[str, object]:
    role = DATA["roles"][DATA["roles"].role_id == target_role_id].iloc[0]
    missing = learner_missing_skills(user_id, target_role_id)
    completed_ids = set(DATA["enrollments"][DATA["enrollments"].user_id == user_id]["course_id"].tolist())
    suggestions = {s: suggest_courses_for_skill(s, completed_ids) for s in missing}
    sorted_missing = sorted(list(missing), key=lambda s: -role.required_skills.get(s, 0))
    return {"missing_skills": sorted_missing, "suggested_courses": suggestions}

# =============================================================
# Self-tests (minimal unit checks)
# =============================================================

def run_self_tests() -> List[str]:
    msgs: List[str] = []
    # Test 1: TF-IDF integrity
    try:
        assert len(course_ids) == DATA["courses"].shape[0]
        msgs.append("✔ TF-IDF course_ids length matches courses")
    except AssertionError:
        msgs.append("✘ TF-IDF course_ids length mismatch")

    # Test 2: rule demotes completed (existing behavior)
    try:
        r = rule_score_for_user(1, 103, None)  # user 1 completed 103
        assert r == 0.0
        msgs.append("✔ Rule score is 0 for completed course")
    except AssertionError:
        msgs.append("✘ Rule score for completed course not zero")

    # Test 3: ML scores dict available
    try:
        scores = ml_scores_for_user(1)
        assert isinstance(scores, dict) and len(scores) == len(course_ids)
        msgs.append("✔ ML scores dict present for all courses")
    except AssertionError:
        msgs.append("✘ ML scores dict missing/size mismatch")

    # New Test 4: quality prior demotes low-rated course 107 vs 105
    try:
        q107 = quality_prior(DATA["courses"][DATA["courses"].course_id == 107].iloc[0])
        q105 = quality_prior(DATA["courses"][DATA["courses"].course_id == 105].iloc[0])
        assert q107 < q105
        msgs.append("✔ Quality prior lower for low-rated course (107 < 105)")
    except AssertionError:
        msgs.append("✘ Quality prior did not demote low-rated course")

    # New Test 5: vectorizer rebuild updates size
    try:
        before = len(course_ids)
        new_id = int(DATA["courses"]["course_id"].max()) + 1
        DATA["courses"] = pd.concat([
            DATA["courses"],
            pd.DataFrame([{ "course_id": new_id, "title": "Tmp Course", "description": "tmp", "tags": "tmp", "skills": "python", "rating": 4.0, "completion_rate": 0.5 }])
        ], ignore_index=True)
        DATA["course_skills"] = pd.concat([DATA["course_skills"], pd.DataFrame([{ "course_id": new_id, "skill": "python" }])], ignore_index=True)
        rebuild_vectorizer(DATA["courses"], DATA["course_skills"])
        after = len(course_ids)
        assert after == before + 1
        msgs.append("✔ Rebuild vectorizer reflects newly added course")
    except AssertionError:
        msgs.append("✘ Vectorizer rebuild did not reflect new course")
    except Exception as e:
        msgs.append(f"✘ Vectorizer rebuild raised exception: {e}")
    finally:
        # remove temp row to keep demo stable
        DATA["courses"] = DATA["courses"][DATA["courses"].course_id != new_id].reset_index(drop=True)
        DATA["course_skills"] = DATA["course_skills"][DATA["course_skills"].course_id != new_id].reset_index(drop=True)
        rebuild_vectorizer(DATA["courses"], DATA["course_skills"])  # restore

    # New Test 6: learner_profile_vector returns (1, n) ndarray
    try:
        prof = learner_profile_vector(1)
        assert isinstance(prof, np.ndarray) and prof.ndim == 2 and prof.shape[0] == 1
        msgs.append("✔ Learner profile is a (1, n) ndarray (not np.matrix)")
    except AssertionError:
        msgs.append("✘ Learner profile is not a (1, n) ndarray")

    # New Test 7: cosine_similarity executes without type errors
    try:
        _ = ml_scores_for_user(1)
        msgs.append("✔ cosine_similarity ran without type errors")
    except Exception as e:
        msgs.append(f"✘ cosine_similarity raised: {e}")

    return msgs

# =============================================================
# UI (Streamlit) or CLI demo
# =============================================================

def _ui_streamlit():  # UI path (only when Streamlit is installed)
    portal = st.sidebar.radio("Portal", ["Learner", "Trainer", "Employer", "Admin"], index=0)
    alpha = st.sidebar.slider("Hybrid α (rule vs ML)", 0.0, 1.0, DEFAULT_ALPHA, 0.05, help="Final score = α*rule + (1-α)*ML")

    st.sidebar.markdown("---")
    role_names = DATA["roles"]["name"].tolist()
    role_label_to_id = {row["name"]: row["role_id"] for _, row in DATA["roles"].iterrows()}
    selected_role_label = st.sidebar.selectbox("Target role for alignment (optional)", ["(none)"] + role_names, index=0)
    selected_role_id = None if selected_role_label == "(none)" else role_label_to_id[selected_role_label]

    if portal == "Learner":
        st.header("🎓 Learner Dashboard")
        learner_name = st.selectbox("Select Learner", DATA["users"][DATA["users"]["role"]=="learner"]["name"].tolist(), index=0)
        uid = int(DATA["users"][DATA["users"]["name"]==learner_name]["user_id"].iloc[0])

        st.subheader("Analytics")
        la = learner_analytics(uid)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Active days (30d)", la["active_days_last_30"]) 
        c2.metric("Time on platform (s)", la["time_on_platform_seconds"]) 
        c3.metric("In-progress", la["courses_in_progress"]) 
        c4.metric("Completion rate", f"{la['completion_rate']*100:.1f}%") 
        c5.metric("Avg quiz", "-" if la["avg_quiz_score"] is None else la["avg_quiz_score"])

        st.subheader("Recommendations (hybrid)")
        k = st.number_input("Top K", min_value=1, max_value=20, value=5)
        recs = recommend_courses(uid, k=int(k), alpha=alpha, target_role_id=selected_role_id)
        st.dataframe(recs, use_container_width=True)

        st.subheader("Skill Gap vs Target Role")
        if selected_role_id is None:
            st.info("Select a target role in the sidebar to compute skill gaps.")
        else:
            gap = skill_gap_report(uid, selected_role_id)
            colA, colB = st.columns(2)
            with colA:
                st.write("**Missing skills (sorted):**", gap["missing_skills"]) 
            with colB:
                st.write("**Suggested courses per skill:**")
                st.json(gap["suggested_courses"])

    elif portal == "Trainer":
        st.header("👩‍🏫 Trainer Dashboard")
        st.write("Create or review courses (demo only). New courses affect ML vectorizer live.")
        with st.expander("Create Course"):
            t = st.text_input("Title")
            d = st.text_area("Description")
            tg = st.text_input("Tags (comma-separated)")
            sk = st.text_input("Skills (comma-separated)")
            rating = st.slider("Historic rating", 1.0, 5.0, 4.2, 0.1)
            comp = st.slider("Historic completion rate", 0.0, 1.0, 0.55, 0.01)
            if st.button("Add Course"):
                new_id = int(DATA["courses"]["course_id"].max()) + 1
                DATA["courses"] = pd.concat([
                    DATA["courses"],
                    pd.DataFrame([{ "course_id": new_id, "title": t, "description": d, "tags": tg, "skills": sk, "rating": float(rating), "completion_rate": float(comp)}])
                ], ignore_index=True)
                if sk.strip():
                    rows = [{"course_id": new_id, "skill": s.strip()} for s in sk.split(",")]
                    DATA["course_skills"] = pd.concat([DATA["course_skills"], pd.DataFrame(rows)], ignore_index=True)
                st.success(f"Course {new_id} added.")
                rebuild_vectorizer(DATA["courses"], DATA["course_skills"])  
                st.toast("Vectorizer retrained.")

        st.subheader("All Courses")
        st.dataframe(DATA["courses"][ ["course_id","title","rating","completion_rate","skills"] ], use_container_width=True)

    elif portal == "Employer":
        st.header("🏢 Employer Dashboard")
        company_name = DATA["companies"].iloc[0]["name"]
        st.caption(f"Company: {company_name}")
        ca = company_analytics(1)
        c1, c2, c3 = st.columns(3)
        c1.metric("DAU", ca["dau"]) 
        c2.metric("MAU", ca["mau"]) 
        c3.metric("DAU/MAU", ca["dau_mau"]) 
        st.metric("Avg completion rate", f"{ca['avg_completion_rate']*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top skills trained")
            st.json(ca["top_skills_trained"])  # [[skill, count], ...]
        with col2:
            st.subheader("Skill proficiency heatmap")
            st.json(ca["skill_gap_heatmap"])   # {skill: % proficient}

    elif portal == "Admin":
        st.header("⚙️ Admin Dashboard")
        st.write("System datasets (read-only). Use Trainer to add courses.")
        msgs = run_self_tests()
        st.expander("Self-tests").write("\n".join(msgs))
        tabs = st.tabs(["Users","Courses","Enrollments","Events","Roles","Course-Skills"])
        with tabs[0]:
            st.dataframe(DATA["users"], use_container_width=True)
        with tabs[1]:
            st.dataframe(DATA["courses"], use_container_width=True)
        with tabs[2]:
            st.dataframe(DATA["enrollments"], use_container_width=True)
        with tabs[3]:
            st.dataframe(DATA["events"], use_container_width=True)
        with tabs[4]:
            st.dataframe(DATA["roles"][ ["role_id","name","required_skills"] ], use_container_width=True)
        with tabs[5]:
            st.dataframe(DATA["course_skills"], use_container_width=True)

    st.caption("Demo only: hybrid recommender = α*rule + (1-α)*ML. Ratings/completions act as quality priors; low-quality items are demoted. No PII is logged.")


def _cli_demo():  # CLI path when Streamlit is missing
    print("{\"mode\": \"cli\", \"message\": \"Streamlit not found; running CLI demo.\"}")
    # Run self-tests
    tests = run_self_tests()
    print(json.dumps({"self_tests": tests}, indent=2))

    # Show sample analytics and recommendations
    learner_id = 1
    company_id = 1
    role_id = 202  # ML Engineer

    la = learner_analytics(learner_id)
    ca = company_analytics(company_id)
    recs = recommend_courses(learner_id, k=5, alpha=DEFAULT_ALPHA, target_role_id=role_id)
    gap = skill_gap_report(learner_id, role_id)

    out = {
        "learner_analytics": la,
        "company_analytics": ca,
        "recommendations": recs,
        "skill_gap": gap,
    }
    print(json.dumps(out, indent=2, default=str))


# Entrypoint selection
if STREAMLIT_AVAILABLE:
    _ui_streamlit()
else:
    _cli_demo()
