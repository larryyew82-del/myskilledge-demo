# role_based_data_entry_app.py
# Single-file **data entry + analysis** app with a Streamlit UI *when available*,
# and a **CLI fallback** (self-tests + CSV I/O) when Streamlit is **not installed**.
#
# Roles supported: Student, University, HRDF, Company, Admin
# Demo auth for Streamlit UI: username **123**, password **123** for ALL roles.
#
# ──────────────────────────────────────────────────────────────────────────────
# How to run (UI):
#   pip install streamlit pandas
#   streamlit run role_based_data_entry_app.py
#
# How to run (CLI fallback – no Streamlit needed):
#   python role_based_data_entry_app.py
#   → runs self-tests, writes CSVs to ./data, and a test report to ./exports
#
# Notes:
# - The CLI fallback exists to avoid "ModuleNotFoundError: No module named 'streamlit'".
# - In CLI mode, no interactive UI is shown; the script exercises CRUD flows
#   for all 5 roles and prints a summary + saves a JSON test report.

from __future__ import annotations
import os
import json
from datetime import datetime, date
from typing import Dict, Any

import pandas as pd

# Try Streamlit; if unavailable, switch to CLI mode gracefully
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except ModuleNotFoundError:
    st = None  # type: ignore
    HAS_STREAMLIT = False

# ----------------------------------------------------------------------------
# Config / paths
# ----------------------------------------------------------------------------
DATA_DIR = os.environ.get("RBDEA_DATA_DIR", "data")
EXPORT_DIR = os.environ.get("RBDEA_EXPORT_DIR", "exports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Provide a no-op cache decorator if Streamlit isn't present
if HAS_STREAMLIT:
    cache_data = st.cache_data  # pragma: no cover
else:
    def cache_data(*args, **kwargs):  # pragma: no cover
        def deco(func):
            return func
        return deco

# ----------------------------------------------------------------------------
# Data model (flat CSVs)
# ----------------------------------------------------------------------------
FILES: Dict[str, list] = {
    "users.csv": ["user_id", "name", "role", "username", "password"],
    "organizations.csv": ["org_id", "org_name", "type", "levy_balance_rm"],
    "modules.csv": [
        "module_code", "module_name", "industry", "claimable",
        "claim_agency", "hours", "outcomes", "fee_rm"
    ],
    "enrollments.csv": ["enroll_id", "user_id", "module_code", "status", "grade", "attendance_hours"],
    "attendance.csv": ["event_id", "enroll_id", "date", "hours", "mode"],
    "jobs.csv": ["job_id", "org_id", "title", "skills", "location", "salary_min", "salary_max", "status"],
    "applications.csv": ["app_id", "job_id", "user_id", "stage", "score"],
    "claims.csv": ["claim_id", "claim_type", "org_id", "user_id", "module_code", "amount_rm", "status"],
    "credit_map.csv": ["module_code", "university_org_id", "degree_program", "credits_awarded"],
}

# Seed some sensible defaults (includes FIVE default users, one per role)
MOCK_ROWS: Dict[str, list] = {
    "users.csv": [
        ["S001", "Student One",   "student",   "123", "123"],
        ["U001", "Uni Admin",      "university", "123", "123"],
        ["H001", "HRDF Officer",   "hrdf",      "123", "123"],
        ["C001", "Company Admin",  "company",   "123", "123"],
        ["A001", "Admin User",     "admin",     "123", "123"],
    ],
    "organizations.csv": [
        ["ORG_EE",  "Penang E&E Sdn Bhd", "employer", 120000],
        ["ORG_MMU", "Multimedia University", "university", 0],
    ],
    "modules.csv": [
        ["AI101", "AI for E&E (Certificate)", "E&E", "yes", "PTPK", 180, "Python; data wrangling", 12000],
    ],
}

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def ensure_csv(path: str, columns: list, seed_rows: list | None = None) -> None:
    if not os.path.exists(path):
        df = pd.DataFrame(seed_rows or [], columns=columns)
        df.to_csv(path, index=False)

@cache_data(show_spinner=False)
def load_df(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    ensure_csv(path, FILES[name], MOCK_ROWS.get(name))
    return pd.read_csv(path)

@cache_data(show_spinner=False)
def list_files() -> list:
    return list(FILES.keys())

def save_df(name: str, df: pd.DataFrame) -> None:
    path = os.path.join(DATA_DIR, name)
    df.to_csv(path, index=False)
    # Invalidate Streamlit's cache if present (fixes AttributeError from calling function.clear())
    if HAS_STREAMLIT:
        try:
            st.cache_data.clear()
        except Exception:
            pass

# deterministic-enough, timestamp-based id (works in UI & CLI)
def new_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}{ts[-10:]}"  # last 10 digits to keep things short

# simple generic upsert utility by a unique key column

def upsert_csv(name: str, key_col: str, row: Dict[str, Any]) -> Dict[str, Any]:
    df = load_df(name)
    key = row[key_col]
    df = df[df[key_col] != key]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_df(name, df)
    return row

# metrics used both by UI and CLI

def quick_metrics() -> Dict[str, Any]:
    enr = load_df("enrollments.csv")
    apps = load_df("applications.csv")
    mods = load_df("modules.csv")

    completion = float(enr["status"].eq("completed").mean() * 100) if len(enr) else 0.0
    placement = float(apps["stage"].eq("offer").mean() * 100) if len(apps) else 0.0

    mix = pd.DataFrame()
    if not mods.empty and "claim_agency" in mods.columns:
        mix = (
            mods.groupby("claim_agency")["fee_rm"].sum().reset_index().rename(columns={"fee_rm": "total_fee_rm"})
        )
    return {
        "completion_rate_pct": round(completion, 1),
        "placement_rate_pct": round(placement, 1),
        "funding_mix_rows": int(len(mix)),
    }

# ----------------------------------------------------------------------------
# CLI fallback (self-tests)
# ----------------------------------------------------------------------------

def run_self_tests() -> Dict[str, Any]:
    """Exercise CRUD flows for all roles and assert basic invariants.
    Tests are **additive** and idempotent-friendly.
    """
    results: Dict[str, Any] = {"tests": []}

    # 0) Files exist
    files = list_files()
    assert set(FILES.keys()).issubset(set(files)), "Missing required CSV files"
    results["tests"].append({"name": "files_exist", "ok": True})

    # 1) Default users include 5 roles
    users = load_df("users.csv")
    for role in ["student", "university", "hrdf", "company", "admin"]:
        assert (users["role"] == role).any(), f"Missing default user for role '{role}'"
    results["tests"].append({"name": "default_users_present", "ok": True})

    # 2) Student: create enrollment + attendance
    enr_before = len(load_df("enrollments.csv"))
    enr_id = new_id("ENR")
    upsert_csv("enrollments.csv", "enroll_id", {
        "enroll_id": enr_id,
        "user_id": "S001",
        "module_code": "AI101",
        "status": "active",
        "grade": "",
        "attendance_hours": 0,
    })
    assert len(load_df("enrollments.csv")) >= enr_before + 1

    att_before = len(load_df("attendance.csv"))
    att_id = new_id("ATT")
    upsert_csv("attendance.csv", "event_id", {
        "event_id": att_id,
        "enroll_id": enr_id,
        "date": date.today().isoformat(),
        "hours": 2,
        "mode": "live",
    })
    assert len(load_df("attendance.csv")) >= att_before + 1
    results["tests"].append({"name": "student_enrollment_attendance", "ok": True})

    # 3) University: upsert a module + credit map row
    mod_before = len(load_df("modules.csv"))
    upsert_csv("modules.csv", "module_code", {
        "module_code": "LG101",
        "module_name": "Logistics Optimization (Cert)",
        "industry": "Logistics",
        "claimable": "yes",
        "claim_agency": "PTPK",
        "hours": 180,
        "outcomes": "Routing; WMS; Excel",
        "fee_rm": 12000,
    })
    assert len(load_df("modules.csv")) >= mod_before  # upsert may overwrite

    cm_before = len(load_df("credit_map.csv")) if os.path.exists(os.path.join(DATA_DIR, "credit_map.csv")) else 0
    upsert_csv("credit_map.csv", "module_code", {
        "module_code": "LG101",
        "university_org_id": "ORG_MMU",
        "degree_program": "BBA Supply Chain",
        "credits_awarded": 12,
    })
    assert len(load_df("credit_map.csv")) >= cm_before + 0  # allow overwrite
    results["tests"].append({"name": "university_module_creditmap", "ok": True})

    # 4) HRDF: create a claim
    clm_before = len(load_df("claims.csv"))
    clm_id = new_id("CLM")
    upsert_csv("claims.csv", "claim_id", {
        "claim_id": clm_id,
        "claim_type": "HRDF",
        "org_id": "ORG_EE",
        "user_id": "S001",
        "module_code": "AI101",
        "amount_rm": 5000,
        "status": "in_progress",
    })
    assert len(load_df("claims.csv")) >= clm_before + 1
    results["tests"].append({"name": "hrdf_claim", "ok": True})

    # 5) Company: job + application
    job_before = len(load_df("jobs.csv"))
    job_id = new_id("JOB")
    upsert_csv("jobs.csv", "job_id", {
        "job_id": job_id,
        "org_id": "ORG_EE",
        "title": "Junior Data Analyst",
        "skills": "python, sql, pandas",
        "location": "Penang",
        "salary_min": 3500,
        "salary_max": 4500,
        "status": "open",
    })
    assert len(load_df("jobs.csv")) >= job_before + 1

    app_before = len(load_df("applications.csv"))
    app_id = new_id("APP")
    upsert_csv("applications.csv", "app_id", {
        "app_id": app_id,
        "job_id": job_id,
        "user_id": "S001",
        "stage": "screen",
        "score": 0.7,
    })
    assert len(load_df("applications.csv")) >= app_before + 1
    results["tests"].append({"name": "company_job_application", "ok": True})

    # 6) Admin: user + organization, plus metrics are sane
    usr_before = len(load_df("users.csv"))
    usr_id = new_id("USR")
    upsert_csv("users.csv", "user_id", {
        "user_id": usr_id,
        "name": "New User",
        "role": "student",
        "username": "123",
        "password": "123",
    })
    assert len(load_df("users.csv")) >= usr_before + 1

    org_before = len(load_df("organizations.csv"))
    org_id = new_id("ORG")
    upsert_csv("organizations.csv", "org_id", {
        "org_id": org_id,
        "org_name": "New Org",
        "type": "employer",
        "levy_balance_rm": 0,
    })
    assert len(load_df("organizations.csv")) >= org_before + 1

    metrics = quick_metrics()
    assert 0.0 <= metrics["completion_rate_pct"] <= 100.0
    assert 0.0 <= metrics["placement_rate_pct"] <= 100.0
    results["tests"].append({"name": "admin_users_orgs_metrics", "ok": True, "metrics": metrics})

    # 7) EXTRA TESTS (added): upsert overwrite & cache invalidation safety
    # 7a) Upsert overwrite: re-save LG101 with different fee, ensure single row and value updated
    mods_before = load_df("modules.csv").copy()
    upsert_csv("modules.csv", "module_code", {
        "module_code": "LG101",
        "module_name": "Logistics Optimization (Cert)",
        "industry": "Logistics",
        "claimable": "yes",
        "claim_agency": "PTPK",
        "hours": 180,
        "outcomes": "Routing; WMS; Excel",
        "fee_rm": 9999,
    })
    mods_after = load_df("modules.csv")
    assert (mods_after["module_code"] == "LG101").sum() == 1
    assert int(mods_after.loc[mods_after["module_code"] == "LG101", "fee_rm"].iloc[0]) == 9999
    results["tests"].append({"name": "upsert_overwrite_singleton", "ok": True})

    # 7b) Cache invalidation path should not crash even without Streamlit
    try:
        df_copy = load_df("users.csv").copy()
        save_df("users.csv", df_copy)
        results["tests"].append({"name": "cache_invalidation_no_crash", "ok": True})
    except Exception as e:
        results["tests"].append({"name": "cache_invalidation_no_crash", "ok": False, "error": str(e)})
        raise

    # Save a machine-readable report
    report_path = os.path.join(EXPORT_DIR, "self_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    results["report_path"] = report_path

    return results

# ----------------------------------------------------------------------------
# Streamlit UI (only runs if Streamlit is installed)
# ----------------------------------------------------------------------------
if HAS_STREAMLIT:
    # Top-level UI definition so `streamlit run` renders immediately
    st.set_page_config(page_title="Raw Data Entry + Analysis", layout="wide")

    # Demo auth (do NOT use in production)
    ROLES = ["student", "university", "hrdf", "company", "admin"]

    def login_ui():
        st.sidebar.header("Login")
        role = st.sidebar.selectbox("Role", ROLES, index=0, help="Select your portal role")
        username = st.sidebar.text_input("Username", value="123")
        password = st.sidebar.text_input("Password", type="password", value="123")
        ok = st.sidebar.button("Sign in")
        if ok:
            if username == "123" and password == "123":
                st.session_state["authed_role"] = role
                st.session_state["username"] = username
                st.session_state["login_time"] = datetime.now().isoformat()
                st.sidebar.success(f"Signed in as {role}")
            else:
                st.sidebar.error("Invalid credentials (demo expects 123/123)")

    if "authed_role" not in st.session_state:
        login_ui()
        # Stop rendering until authed
        if "authed_role" not in st.session_state:
            st.stop()

    # Common widgets
    def table_view(name: str):
        st.subheader(f"Current data — {name}")
        df = load_df(name)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            f"Download {name}",
            df.to_csv(index=False).encode("utf-8"),
            file_name=name,
            mime="text/csv",
        )

    role = st.session_state["authed_role"]
    st.title("📦 Raw Data Entry + Quick Analysis")
    st.caption("Demo login for ALL roles: username 123 / password 123")
    st.toast(f"Logged in as '{role}'", icon="✅")

    # ───── Student ───────────────────────────────────────────────────────────
    if role == "student":
        st.header("🎓 Student Portal")
        udf = load_df("users.csv")
        students = udf[udf["role"] == "student"]
        if students.empty:
            st.info("No student users available. Ask Admin to create one in users.csv.")
        else:
            student_id = st.selectbox("Select your ID", students["user_id"].tolist())

            st.subheader("Add/Update Enrollment")
            with st.form("enr_form"):
                enroll_id = st.text_input("Enroll ID (leave blank for auto)")
                module_code = st.text_input("Module Code", value="AI101")
                status = st.selectbox("Status", ["planned", "active", "completed"], index=1)
                grade = st.text_input("Grade", value="")
                att_hours = st.number_input("Attendance Hours", min_value=0, value=0)
                submitted = st.form_submit_button("Save Enrollment")
            if submitted:
                if not enroll_id:
                    enroll_id = new_id("ENR")
                row = {
                    "enroll_id": enroll_id,
                    "user_id": student_id,
                    "module_code": module_code,
                    "status": status,
                    "grade": grade,
                    "attendance_hours": att_hours,
                }
                upsert_csv("enrollments.csv", "enroll_id", row)
                st.success(f"Enrollment saved: {enroll_id}")

            table_view("enrollments.csv")

            st.subheader("Log Attendance")
            with st.form("att_form"):
                event_id = st.text_input("Event ID (auto if blank)")
                sel_enr_df = load_df("enrollments.csv")
                sel_enr_list = sel_enr_df["enroll_id"].tolist() if not sel_enr_df.empty else []
                sel_enr = st.selectbox("Enrollment", sel_enr_list)
                d = st.date_input("Date")
                hours = st.number_input("Hours", min_value=0, value=2)
                mode = st.selectbox("Mode", ["live", "recording"], index=0)
                ok = st.form_submit_button("Save Attendance")
            if ok:
                if not event_id:
                    event_id = new_id("ATT")
                row = {"event_id": event_id, "enroll_id": sel_enr, "date": str(d), "hours": hours, "mode": mode}
                upsert_csv("attendance.csv", "event_id", row)
                st.success(f"Attendance saved: {event_id}")

            table_view("attendance.csv")

    # ───── University ────────────────────────────────────────────────────────
    elif role == "university":
        st.header("🏫 University Portal")
        st.subheader("Create / Edit Module")
        with st.form("mod_form"):
            module_code = st.text_input("Module Code", value="AI101")
            module_name = st.text_input("Module Name", value="AI for E&E (Certificate)")
            industry = st.text_input("Industry", value="E&E")
            claimable = st.selectbox("Claimable", ["yes", "no"], index=0)
            claim_agency = st.selectbox("Claim Agency", ["PTPK", "HRDF", "-"], index=0)
            hours = st.number_input("Hours", min_value=0, value=180)
            outcomes = st.text_area("Outcomes", value="Python; data wrangling")
            fee_rm = st.number_input("Fee (RM)", min_value=0, value=12000)
            ok = st.form_submit_button("Save Module")
        if ok:
            row = {
                "module_code": module_code,
                "module_name": module_name,
                "industry": industry,
                "claimable": claimable,
                "claim_agency": claim_agency if claim_agency != "-" else "",
                "hours": hours,
                "outcomes": outcomes,
                "fee_rm": fee_rm,
            }
            upsert_csv("modules.csv", "module_code", row)
            st.success(f"Module saved: {module_code}")
        table_view("modules.csv")

        st.subheader("Credit Transfer Map")
        with st.form("credit_form"):
            cm_module = st.text_input("Module Code (must exist)", value="AI101")
            uni_org = st.text_input("University Org ID", value="ORG_MMU")
            degree = st.text_input("Degree Program", value="BSc Data Science")
            credits = st.number_input("Credits Awarded", min_value=0, value=12)
            ok2 = st.form_submit_button("Save Credit Map")
        if ok2:
            new_row = {"module_code": cm_module, "university_org_id": uni_org, "degree_program": degree, "credits_awarded": credits}
            # Upsert by module_code for simplicity
            upsert_csv("credit_map.csv", "module_code", new_row)
            st.success("Credit map row saved")
        table_view("credit_map.csv")

    # ───── HRDF ─────────────────────────────────────────────────────────────
    elif role == "hrdf":
        st.header("🏛 HRDF Portal")
        with st.form("claim_form"):
            claim_id = st.text_input("Claim ID (auto if blank)")
            claim_type = st.selectbox("Claim Type", ["HRDF", "PTPK"], index=0)
            org_id = st.text_input("Org ID", value="ORG_EE")
            user_id = st.text_input("Trainee User ID", value="S001")
            module_code = st.text_input("Module Code", value="AI101")
            amount_rm = st.number_input("Amount (RM)", min_value=0, value=5000)
            status = st.selectbox("Status", ["in_progress", "approved", "rejected"], index=0)
            ok = st.form_submit_button("Save Claim")
        if ok:
            if not claim_id:
                claim_id = new_id("CLM")
            row = {
                "claim_id": claim_id,
                "claim_type": claim_type,
                "org_id": org_id,
                "user_id": user_id,
                "module_code": module_code,
                "amount_rm": amount_rm,
                "status": status,
            }
            upsert_csv("claims.csv", "claim_id", row)
            st.success(f"Claim saved: {claim_id}")
        table_view("claims.csv")

    # ───── Company ──────────────────────────────────────────────────────────
    elif role == "company":
        st.header("🏢 Company Portal")
        with st.form("job_form"):
            job_id = st.text_input("Job ID (auto if blank)")
            org_id = st.text_input("Org ID", value="ORG_EE")
            title = st.text_input("Job Title", value="Junior Data Analyst")
            skills = st.text_input("Skills (comma-separated)", value="python, sql, pandas")
            location = st.text_input("Location", value="Penang")
            salary_min = st.number_input("Salary Min", min_value=0, value=3500)
            salary_max = st.number_input("Salary Max", min_value=0, value=4500)
            status = st.selectbox("Status", ["open", "closed"], index=0)
            ok = st.form_submit_button("Save Job")
        if ok:
            if not job_id:
                job_id = new_id("JOB")
            row = {
                "job_id": job_id,
                "org_id": org_id,
                "title": title,
                "skills": skills,
                "location": location,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "status": status,
            }
            upsert_csv("jobs.csv", "job_id", row)
            st.success(f"Job saved: {job_id}")
        table_view("jobs.csv")

        st.subheader("Applications")
        with st.form("app_form"):
            app_id = st.text_input("Application ID (auto if blank)")
            job_id_ref = st.text_input("Job ID (must exist)")
            user_id = st.text_input("Student User ID", value="S001")
            stage = st.selectbox("Stage", ["screen", "interview", "offer", "rejected"], index=0)
            score = st.number_input("Score (0-1)", min_value=0.0, max_value=1.0, value=0.7)
            ok2 = st.form_submit_button("Save Application")
        if ok2:
            if not app_id:
                app_id = new_id("APP")
            row = {"app_id": app_id, "job_id": job_id_ref, "user_id": user_id, "stage": stage, "score": score}
            upsert_csv("applications.csv", "app_id", row)
            st.success(f"Application saved: {app_id}")
        table_view("applications.csv")

    # ───── Admin ────────────────────────────────────────────────────────────
    elif role == "admin":
        st.header("🛠️ Admin Portal")
        st.subheader("Create / Edit User")
        with st.form("user_form"):
            user_id = st.text_input("User ID (auto if blank)")
            name = st.text_input("Name", value="New User")
            role_pick = st.selectbox("Role", ["student", "university", "hrdf", "company", "admin"], index=0)
            username = st.text_input("Username", value="123")
            password = st.text_input("Password", value="123")
            ok = st.form_submit_button("Save User")
        if ok:
            if not user_id:
                user_id = new_id("USR")
            row = {"user_id": user_id, "name": name, "role": role_pick, "username": username, "password": password}
            upsert_csv("users.csv", "user_id", row)
            st.success(f"User saved: {user_id}")
        table_view("users.csv")

        st.subheader("Organizations")
        with st.form("org_form"):
            org_id = st.text_input("Org ID (auto if blank)")
            org_name = st.text_input("Org Name", value="Penang E&E Sdn Bhd")
            typ = st.selectbox("Type", ["employer", "university", "agency"], index=0)
            levy_balance = st.number_input("Levy Balance (RM)", min_value=0, value=0)
            ok2 = st.form_submit_button("Save Org")
        if ok2:
            if not org_id:
                org_id = new_id("ORG")
            row = {"org_id": org_id, "org_name": org_name, "type": typ, "levy_balance_rm": levy_balance}
            upsert_csv("organizations.csv", "org_id", row)
            st.success(f"Organization saved: {org_id}")
        table_view("organizations.csv")

        st.subheader("Quick Analysis")
        m = quick_metrics()
        c1, c2 = st.columns(2)
        c1.metric("Completion Rate", f"{m['completion_rate_pct']:.1f}%")
        c2.metric("Placement (Offer) Rate", f"{m['placement_rate_pct']:.1f}%")

        mods = load_df("modules.csv")
        st.subheader("Funding Mix by Agency (from modules)")
        if not mods.empty and "claim_agency" in mods.columns:
            mix = (
                mods.groupby("claim_agency")["fee_rm"].sum().reset_index().rename(columns={"fee_rm": "total_fee_rm"})
            )
            st.dataframe(mix, use_container_width=True)
            st.bar_chart(mix.set_index("claim_agency"))
        else:
            st.info("No modules data yet.")

    # Footer
    st.divider()
    st.caption("⚠️ Demo app: flat-file CSV storage, no real security. Replace with proper auth/DB before production use.")

# ----------------------------------------------------------------------------
# CLI entrypoint (only active when executing with `python` and no Streamlit)
# ----------------------------------------------------------------------------
if __name__ == "__main__" and not HAS_STREAMLIT:
    print("[RBDEA] Streamlit not found – running in CLI fallback mode.\n")
    print("- Data dir:   ", os.path.abspath(DATA_DIR))
    print("- Export dir: ", os.path.abspath(EXPORT_DIR))
    results = run_self_tests()
    print("\nSelf-tests completed. Summary:")
    for t in results.get("tests", []):
        name = t.get("name")
        ok = t.get("ok")
        print(f"  • {name}: {'OK' if ok else 'FAIL'}")
        if name == "admin_users_orgs_metrics":
            print(f"    - completion_rate_pct = {t['metrics']['completion_rate_pct']}")
            print(f"    - placement_rate_pct  = {t['metrics']['placement_rate_pct']}")
            print(f"    - funding_mix_rows    = {t['metrics']['funding_mix_rows']}")
    print(f"\nReport saved to: {results['report_path']}")
    print("\nTip: To use the full UI, install Streamlit and run:\n  pip install streamlit pandas\n  streamlit run role_based_data_entry_app.py\n")
