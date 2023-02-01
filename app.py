import streamlit as st
import pandas as pd
import numpy as np
import backend.job_search

@st.cache
def find_jobs(id):
    search = backend.job_search.JobSearch([id], False, False)
    suggested_jobs = search.get_best_matches(5)
    suggested_job_list = [x.job2.job_name for x in suggested_jobs]
    return suggested_job_list


st.title('MCC Sussex tool')


tab = pd.read_csv('occupations_en.csv')
options = pd.Series(tab['preferredLabel'])

form = st.form("my_form")
selected_job = form.selectbox(label = 'Select your job', options = options)
job_id = options[options == selected_job].index[0]
submit = form.form_submit_button("Submit")



if submit:
    st.write(f"Your selected job is {selected_job} with an ID of {job_id}")

    expander = st.expander("Suggested jobs")
    suggested_jobs = find_jobs(job_id)
    expander.write(f'Your suggested jobs transtions are to {suggested_jobs}')