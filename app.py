import streamlit as st
import pandas as pd

st.set_page_config(page_title="MySkillEdge Demo", layout="wide")

st.sidebar.title("MySkillEdge Demo")
page = st.sidebar.radio("Choose a portal", ["Student", "Employer", "University", "Agency"])

if page == "Student":
    st.title("Student Portal")
    st.write("Demo: Students can view their modules, attendance, and generate funding packs (PTPK).")

elif page == "Employer":
    st.title("Employer Portal")
    st.write("Demo: Employers can search for candidates, conduct interviews, and generate HRDF claim packs.")

elif page == "University":
    st.title("University Portal")
    st.write("Demo: Universities can issue co-branded certificates and verify stackable modules.")

elif page == "Agency":
    st.title("Agency Portal")
    st.write("Demo: Agencies can view reports, funding KPIs, and approve claims.")
