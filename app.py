# app.py
# MySkillEdge Demo – single-file Streamlit app
# Shows Student / Employer / University / Agency flows with mock data
# Generates:
# - PTPK Evidence Pack (JSON)
# - HRDF Claim Pack (JSON)
# - Co-branded Credential (JSON with SHA-256 hash)
#
# How to run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Deploy on Streamlit Cloud:
#   Push this file + requirements.txt to a public GitHub repo, then deploy with app.py as entrypoint.

import os
import json
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

# -------------------------
# Basic configuration
# -------------------------
st.set_page_config(page_title="MySkillEdge Demo", layout="wide")
DATA_DIR = "data"
EXPORT_DIR = "exports"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# -------------------------
# Bootstrap mock datasets (auto-creates CSVs if missing)
# -------------------------
MOCKS = {
    "users.csv": """user_id,name,role,org_id,cohort
S001,Amir Hakim,student,,2025A
S002,Aisha Tan,student,,2025A
S003,Arun Kumar,student,,2025B
E101,Rahman HR,employer_admin,ORG_EE,
U201,Dr Tan,university_admin,ORG_MMU,
A301,Noraini,agency_admin,ORG_HRDF,
""",
    "organizations.csv": """org_id,org_name,type,levy_balance_rm
ORG_EE,Penang E&E Sdn Bhd,employer,120000
ORG_MMU,Multimedia University,university,0
ORG_HRDF,HRDF,agency,0
""",
    "modules.csv": """module_code,module_name,industry,claimable,claim_agency,hours,outcomes,fee_rm
AI101,AI for E&E (Certificate),E&E,yes,PTPK,180,"Python basics; data wrangling; E&E datasets",12000
AI201,Applied Machine Learning,E&E,yes,HRDF,90,"Regression; classification; model eval",5000
LG101,Logistics Optimization (Cert),Logistics,yes,PTPK,180,"Routing; WMS; Excel",12000
HC101,Healthcare Data (Cert),Healthcare,yes,PTPK,180,"Privacy basics; dashboards",12000
AD301,Advanced Diploma Capstone (AI in Industry),Cross,yes,HRDF,300,"Employer capstone project",15000
""",
    "enrollments.csv": """enroll_id,user_id,module_code,status,grade,attendance_hours
ENR1,S001,AI101,active,,0
ENR2,S002,LG101,completed,A,185
ENR3,S003,AI101,active,,0
ENR4,S001,AI201,planned,,0
""",
    "attendance.csv": """event_id,enroll_id,date,hours,mode
ATT1,ENR1,2025-08-01,3,live
ATT2,ENR1,2025-08-02,3,live
ATT3,ENR1,2025-08-03,2,recording
ATT4,ENR2,2025-07-15,4,live
ATT5,ENR2,2025-07-16,4,live
""",
    "jobs.csv": """job_id,org_id,title,skills,location,salary_min,salary_max,status
J001,ORG_EE,Junior Data Analyst (E&E),"python, sql, pandas, electronics",Penang,3500,4500,open
J002,ORG_EE,Production Data Tech,"excel, powerbi, manufacturing",Penang,2800,3800,open
""",
    "applications.csv": """app_id,job_id,user_id,stage,score
APP1,J001,S001,screen,0.72
APP2,J002,S002,offer,0.86
""",
    "claims.csv": """claim_id,claim_type,org_id,user_id,module_code,amount_rm,status
CLM1,PTPK,,S001,AI101,12000,in_progress
CLM2,HRDF,ORG_EE,S002,AI201,5000,approved
""",
    "credit_map.csv": """module_code,university_org_id,degree_program,credits_awarded
AI101,ORG_MMU,BSc Data Science,12
AI201,ORG_MMU,BSc Data Science,6
LG101,ORG_MMU,BBA Supply Chain,12
AD301,ORG_MMU,Master of Applied AI,9
"""
}

def ensure_csv(name: str):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(MOCKS[name])

for fname in MOCKS:
    ensure_csv(fname)

# -------------------------
# Load data helpers
# -------------------------
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, name))

def load_all():
    return {
        "users": load_csv("users.csv"),
        "orgs": load_csv("organizations.csv"),
        "modules": load_csv("modules.csv"),
        "enrollments": load_csv("enrollments.csv"),
        "attendance": load_csv("attendance.csv"),
        "jobs": load_csv("jobs.csv"),
        "applications": load_csv("applications.csv"),
        "claims": load_csv("claims.csv"),
        "credit_map": load_csv("credit_map.csv"),
    }

data = load_all()

# -------------------------
# Utility: export & verify
# -------------------------
def save_export_text(filename: str, content: str):
    path = os.path.join(EXPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def save_export_json(filename: str, payload: dict):
    text = json.dumps(payload, indent=2)
    return save_export_text(filename, text)

def sha256_hexdigest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -------------------------
# UI Header
# -------------------------
st.title("📚 MySkillEdge – End-to-End Demo")
st.caption("Student ↔ Employer ↔ University ↔ Agency | Funding: PTPK & HRDF | Mock data, audit-friendly flows")
st.divider()

portal = st.sidebar.radio("Choose Portal", ["Student", "Employer", "University", "Agency"])

# -------------------------
# STUDENT PORTAL
# -------------------------
if portal == "Student":
    st.header("🎓 Student Portal")

    students = data["users"][data["users"]["role"] == "student"].copy()
    student_name = st.selectbox("Select Student", students["name"].tolist(), index=0)
    srow = students[students["name"] == student_name].iloc[0]
    st.write("**Profile**", dict(srow))

    my_enr = data["enrollments"][data["enrollments"]["user_id"] == srow["user_id"]].copy()
    st.subheader("Enrollments")
    st.dataframe(my_enr, use_container_width=True)

    my_att = data["attendance"][data["attendance"]["enroll_id"].isin(my_enr["enroll_id"].tolist())]
    total_hours = 0 if my_att.empty else float(my_att["hours"].sum())
    st.metric("Total Attendance Hours (logged)", f"{total_hours:.0f}")

    st.subheader("Generate PTPK Evidence Pack (mock)")
    ptpk_modules = data["modules"][data["modules"]["claim_agency"] == "PTPK"]["module_code"].tolist()
    if ptpk_modules:
        mod_code = st.selectbox("Module", ptpk_modules, index=0)
        if st.button("Generate PTPK Pack"):
            mod = data["modules"][data["modules"]["module_code"] == mod_code].iloc[0]
            pack = {
                "type": "PTPK_EVIDENCE_PACK",
                "generated_at": datetime.now().isoformat(),
                "student": {
                    "name": srow["name"],
                    "user_id": srow["user_id"],
                    "cohort": srow.get("cohort", "")
                },
                "module": {
                    "code": mod["module_code"],
                    "name": mod["module_name"],
                    "hours_required": int(mod["hours"]),
                    "outcomes": mod["outcomes"],
                    "fee_rm": int(mod["fee_rm"]),
                },
                "attendance_logged_hours": int(total_hours),
                "attachments": [
                    "attendance_log.csv",
                    "syllabus.pdf",
                    "trainer_credentials.pdf",
                    "invoice.pdf",
                    "student_kyc.pdf"
                ],
                "note": "Mock export for demo purposes."
            }
            filename = f"PTPK_{srow['user_id']}_{mod_code}.json"
            path = save_export_json(filename, pack)
            st.success(f"Generated: {path}")
            st.download_button("Download PTPK Pack (JSON)", json.dumps(pack, indent=2), file_name=filename, mime="application/json")
            st.code(json.dumps(pack, indent=2), language="json")
    else:
        st.info("No PTPK-claimable modules in mock data.")

# -------------------------
# EMPLOYER PORTAL
# -------------------------
elif portal == "Employer":
    st.header("🏢 Employer Portal")

    emps = data["orgs"][data["orgs"]["type"] == "employer"].copy()
    org_name = st.selectbox("Select Employer", emps["org_name"].tolist(), index=0)
    org = emps[emps["org_name"] == org_name].iloc[0]
    st.metric("HRDF Levy Balance (RM)", f"{int(org['levy_balance_rm']):,}")

    st.subheader("Open Jobs")
    jobs = data["jobs"][data["jobs"]["org_id"] == org["org_id"]]
    st.dataframe(jobs, use_container_width=True)

    st.subheader("AI-style Candidate Matches (simple keyword overlap)")
    req_skills = st.text_input("Paste Job Skills (comma-separated)", "python, sql, pandas, electronics")
    if st.button("Find Candidates"):
        wanted = [s.strip().lower() for s in req_skills.split(",") if s.strip()]
        # naive scoring: overlap between wanted skills and module outcomes text
        merged = data["enrollments"].merge(data["modules"], on="module_code", how="left")
        merged = merged.merge(data["users"][data["users"]["role"] == "student"], left_on="user_id", right_on="user_id", how="left")
        merged = merged.dropna(subset=["name"])
        def score_row(row):
            blob = (str(row["module_name"]) + " " + str(row["outcomes"])).lower()
            return sum(1 for w in wanted if w in blob)
        merged["match_score"] = merged.apply(score_row, axis=1)
        ranked = merged.groupby(["user_id", "name"], as_index=False)["match_score"].sum().sort_values("match_score", ascending=False)
        st.dataframe(ranked, use_container_width=True)

    st.subheader("Generate HRDF Claim Pack (mock)")
    person_opts = data["users"][data["users"]["role"] == "student"]["name"].tolist()
    student_name = st.selectbox("Select Trainee", person_opts, index=0)
    mod_code = st.selectbox("Module Trained", data["modules"]["module_code"].tolist(), index=1)
    amount = st.number_input("Amount (RM)", value=5000, step=500, min_value=0)
    if st.button("Generate HRDF Pack"):
        pack = {
            "type": "HRDF_CLAIM_PACK",
            "generated_at": datetime.now().isoformat(),
            "employer": {
                "org_id": org["org_id"],
                "org_name": org["org_name"]
            },
            "trainee": student_name,
            "module_code": mod_code,
            "amount_rm": int(amount),
            "checklist": [
                "attendance_sheet.csv",
                "syllabus.pdf",
                "trainer_qualification.pdf",
                "invoice.pdf",
                "employer_approval_memo.pdf"
            ],
            "note": "Mock export for demo purposes."
        }
        filename = f"HRDF_{org['org_id']}_{mod_code}.json"
        path = save_export_json(filename, pack)
        st.success(f"Generated: {path}")
        st.download_button("Download HRDF Pack (JSON)", json.dumps(pack, indent=2), file_name=filename, mime="application/json")
        st.code(json.dumps(pack, indent=2), language="json")

# -------------------------
# UNIVERSITY PORTAL
# -------------------------
elif portal == "University":
    st.header("🎓 University Portal")
    uni = data["orgs"][data["orgs"]["type"] == "university"].iloc[0]
    st.write("**Partner University**:", uni["org_name"])

    st.subheader("Credit Transfer Map")
    st.dataframe(data["credit_map"], use_container_width=True)

    st.subheader("Issue Co-branded Certificate (mock)")
    students = data["users"][data["users"]["role"] == "student"]["name"].tolist()
    student = st.selectbox("Student", students, index=0)
    mod_code = st.selectbox("Completed Module", data["modules"]["module_code"].tolist(), index=0)
    if st.button("Issue Certificate"):
        payload = {
            "type": "CO_BRANDED_CREDENTIAL",
            "student": student,
            "module_code": mod_code,
            "issuer": f"MySkillEdge + {uni['org_name']}",
            "issued_at": datetime.now().isoformat()
        }
        # Create a verifiable hash
        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = sha256_hexdigest(payload_str)
        credential = {
            **payload,
            "sha256": payload_hash,
            "verify_instructions": "Recompute SHA-256 over the sorted JSON payload and match with 'sha256'."
        }
        filename = f"Credential_{student.replace(' ','_')}_{mod_code}.json"
        path = save_export_json(filename, credential)
        st.success(f"Issued: {path}")
        st.download_button("Download Credential (JSON)", json.dumps(credential, indent=2), file_name=filename, mime="application/json")
        st.code(json.dumps(credential, indent=2), language="json")

# -------------------------
# AGENCY DASHBOARD
# -------------------------
else:
    st.header("🏛 Agency Dashboard (HRDF / PTPK)")

    # KPIs
    st.subheader("Key Metrics (mock)")
    # Completion rate: % of enrollments with status == completed
    comp = (data["enrollments"]["status"] == "completed").mean() if len(data["enrollments"]) else 0.0
    # Placement rate: % applications at 'offer'
    place = (data["applications"]["stage"] == "offer").mean() if len(data["applications"]) else 0.0
    c1, c2 = st.columns(2)
    c1.metric("Completion Rate", f"{comp*100:.1f}%")
    c2.metric("Placement (Offer) Rate", f"{place*100:.1f}%")

    st.subheader("Claim Status")
    st.dataframe(data["claims"], use_container_width=True)

    st.subheader("Funding Mix by Agency (sum of module fees)")
    mix = data["modules"].groupby("claim_agency")["fee_rm"].sum().reset_index()
    if not mix.empty:
        mix = mix.rename(columns={"fee_rm": "total_fee_rm"})
        st.dataframe(mix, use_container_width=True)
        st.bar_chart(mix.set_index("claim_agency"))
    else:
        st.info("No modules in dataset.")
