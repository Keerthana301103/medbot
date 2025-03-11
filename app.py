import streamlit as st
import requests
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from requests.exceptions import ConnectionError, Timeout

# Initialize session state
if "qa_chain" not in st.session_state:
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings()
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(embeddings)
    except ConnectionError:
        st.error("Failed to connect to Ollama API. Ensure the service is running.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error initializing QA chain: {e}")
        st.stop()

st.title("MedBot - AI Health Assistant")

user_question = st.text_input("Ask a question:")

if user_question:
    try:
        with st.spinner("Processing..."):
            response = st.session_state.qa_chain({"question": user_question})
            st.write(response)

    except ConnectionError:
        st.error("Connection error: Unable to reach the API. Check your internet or API service.")
    except Timeout:
        st.error("Request timed out. Try again later.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
