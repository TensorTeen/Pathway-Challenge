import requests
import streamlit as st
import os
import time
from typing import List

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(page_title="Finance QA", layout="wide")

st.title("📊 Finance QA System")

tabs = st.tabs(["Documents", "Ask", "Explain"])

with tabs[0]:
    st.header("Documents")
    upl = st.file_uploader("Upload PDF (async)", type=['pdf'])
    if upl is not None and st.button("Start Upload"):
        files = {'file': (upl.name, upl.read(), upl.type)}
        r = requests.post(f"{API_URL}/upload_async", files=files)
        if r.ok:
            job_id = r.json()['job_id']
            st.info(f"Upload started. Job ID: {job_id}")
            progress_ph = st.empty()
            events_ph = st.empty()
            finished = False
            while not finished:
                evr = requests.get(f"{API_URL}/jobs/{job_id}")
                if evr.ok:
                    events = evr.json().get('events', [])
                    pct = 0.0
                    for e in events:
                        if e['event'] == 'progress':
                            pct = e['data'].get('pct', pct)
                        if e['event'] == 'job_finished':
                            finished = True
                    progress_ph.progress(pct if pct <= 1 else 1.0, text=f"Progress: {pct*100:.1f}%")
                    # show last few events
                    events_ph.json(events[-5:])
                time.sleep(0.7)
            st.success("Upload completed")
        else:
            st.error(r.text)
    if st.button("Refresh File List"):
        pass
    r = requests.get(f"{API_URL}/files")
    if r.ok:
        files = r.json().get('files', [])
        st.write(files)
        del_choice = st.selectbox("Delete file", [""] + files)
        if del_choice and st.button("Confirm Delete"):
            dr = requests.delete(f"{API_URL}/files/{del_choice}")
            if dr.ok:
                st.warning(f"Deleted {del_choice}")
            else:
                st.error(dr.text)

with tabs[1]:
    st.header("Ask a Question (async)")
    question = st.text_input("Enter your finance question")
    answer_container = st.container()
    trace_container = st.container()
    if st.button("Submit Question") and question:
        r = requests.post(f"{API_URL}/question_async", json={'question': question})
        if not r.ok:
            st.error(r.text)
        else:
            job_id = r.json()['job_id']
            st.info(f"Question processing started. Job ID: {job_id}")
            progress_ph = st.empty()
            events_ph = st.empty()
            finished = False
            trace_id = None
            pct = 0.0
            while not finished:
                evr = requests.get(f"{API_URL}/jobs/{job_id}")
                if evr.ok:
                    events = evr.json().get('events', [])
                    for e in events:
                        if e['event'] == 'progress':
                            pct = e['data'].get('pct', pct)
                        if e['event'] == 'job_finished':
                            finished = True
                        if e['event'] == 'info' and e['data'].get('trace_id'):
                            trace_id = e['data']['trace_id']
                    progress_ph.progress(pct if pct <= 1 else 1.0, text=f"Progress: {pct*100:.1f}%")
                    events_ph.json(events[-8:])
                time.sleep(0.8)
            st.success("Question processing complete")
            if trace_id:
                # fetch full trace (contains final_answer)
                tr_full = requests.get(f"{API_URL}/trace/{trace_id}")
                if tr_full.ok:
                    full = tr_full.json()
                    final_answer = full.get('final_answer', {}).get('answer') if isinstance(full.get('final_answer'), dict) else full.get('final_answer')
                    reasoning = full.get('final_answer', {}).get('reasoning') if isinstance(full.get('final_answer'), dict) else None
                    if final_answer:
                        answer_container.subheader("Final Answer")
                        answer_container.write(final_answer)
                        if reasoning:
                            with answer_container.expander("Model Reasoning"):
                                st.write(reasoning)
                    # show structured trace below
                    trace_container.subheader("Trace Steps")
                    steps = full.get('steps', [])
                    for s in steps:
                        with trace_container.expander(f"Loop {s.get('loop')} - {s.get('type')}"):
                            st.json(s)
                else:
                    st.error("Could not fetch trace")

with tabs[2]:
    st.header("Explain a Trace")
    trace_id = st.text_input("Trace ID", value=st.session_state.get('last_trace', {}).get('id', ''))
    if st.button("Explain") and trace_id:
        r = requests.post(f"{API_URL}/explain", json={'trace_id': trace_id})
        if r.ok:
            st.write(r.json().get('explanation'))
        else:
            st.error(r.text)
