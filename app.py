import streamlit as st
from pdf_parser import extract_text_from_pdf
from qa_chain import setup_qa_chain

# UI Theme
st.set_page_config(page_title="MedAI Bot", page_icon="🩺", layout="wide")
st.markdown("""
    <style>
        .main {background-color:#D1DFE3;}
        .stButton>button {background-color: #7AA8C6; color: white; border-radius: 8px; padding: 8px 16px;}
        .stTextInput>div>div>input {background-color: #E3E6EB; color: black;}
        .stMarkdown {color: black;}
        .stChatMessage {background-color: #DDE2EB; padding: 10px; border-radius: 5px; margin-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

st.title("MedAI Bot")

# Session state for uploaded report & QA chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "uploaded_report" not in st.session_state:
    st.session_state.uploaded_report = None

# New Conversation Button
if st.button(" New Conversation"):
    st.session_state.qa_chain = None
    st.session_state.uploaded_report = None
    st.rerun() 
# File Upload
uploaded_file = st.file_uploader("Upload a medical report (PDF)", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    report_text = extract_text_from_pdf("temp.pdf")

    if report_text != st.session_state.uploaded_report:
        st.session_state.qa_chain = setup_qa_chain(report_text)
        st.session_state.uploaded_report = report_text
        st.success("New medical report processed. Ask your questions!")

# Chat Interface
if st.session_state.qa_chain:
    user_question = st.text_input("Ask a question about the report:")

    if user_question:
        response = st.session_state.qa_chain({"question": user_question})

        if not response or not response["answer"].strip():
            st.warning("I can only answer questions related to the uploaded report.")
        else:
            st.markdown(f"<div class='stChatMessage'><b>Response:</b> {response['answer']}</div>", unsafe_allow_html=True)
