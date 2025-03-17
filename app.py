import streamlit as st
import os
from pathlib import Path
import matplotlib.pyplot as plt
from qa_chain import process_user_query

# ✅ Set page config
st.set_page_config(
    page_title="MedAI - Medical Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ✅ Custom CSS for Maroon & Red Theme
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
        }
        .stApp {
            background-color: white;
        }
        .stFileUploader {
            background-color: #f8f8f8;
            border: 2px solid maroon;
            padding: 10px;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            border: 2px solid maroon;
            border-radius: 5px;
            padding: 8px;
        }
        .stButton>button {
            background-color: white;
            color: maroon;
            border: 2px solid maroon;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: maroon;
            color: white;
        }
        .stAlert {
            border-left: 5px solid #d9534f;
        }
        .maroon-text {
            color: maroon;
            font-size: 20px;
            font-weight: bold;
        }
        .red-text {
            color: #d9534f;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Upload & Analyze"])

# ✅ Ensure the uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ✅ Home Page
if page == "Home":
    st.markdown("<h1 class='maroon-text'>MedAI - AI-Powered Medical Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='red-text'>MedAI helps analyze medical reports and brain MRI scans using AI.</p>", unsafe_allow_html=True)
    
    # 🔹 **Reduced banner size**
    st.image(r"C:\Users\s.anumandla\Desktop\medBot\medbanner.png", use_container_width=False, width=500)

    st.markdown("""
        <p class="maroon-text">Features:</p>
        <ul class="red-text">
            <li>Upload a <b>PDF medical report</b> and ask AI-based questions.</li>
            <li>Upload a <b>brain MRI scan</b> for AI-driven insights.</li>
            <li>Get <b>AI-generated explanations</b> for MRI scans.</li>
        </ul>
    """, unsafe_allow_html=True)
    st.success("Select 'Upload & Analyze' from the sidebar to begin.")

# ✅ Upload & Analyze Page
elif page == "Upload & Analyze":
    st.markdown("<h1 class='maroon-text'>Upload a File for Analysis</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a medical report (PDF) or MRI scan (PNG, JPG, JPEG):",
        type=["pdf", "png", "jpg", "jpeg"]
    )
    
    query = st.text_input("Ask a question about the file:")

    # ✅ Process File If Uploaded & Query is Provided
    if uploaded_file and query:
        file_path = UPLOAD_DIR / uploaded_file.name
        file_extension = file_path.suffix.lower()

        # ✅ Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ✅ Process the File
        with st.spinner("Processing... Please wait."):
            try:
                if file_extension == ".pdf":
                    st.markdown("<h2 class='maroon-text'>Medical Report Analysis</h2>", unsafe_allow_html=True)
                    response = process_user_query(query, file_path, "pdf")
                    st.markdown(f"<p class='red-text'>{response}</p>", unsafe_allow_html=True)

                elif file_extension in [".png", ".jpg", ".jpeg"]:
                    st.markdown("<h2 class='maroon-text'>Brain MRI Analysis</h2>", unsafe_allow_html=True)

                    # Process MRI and get response + explanation
                    response, explanation_fig = process_user_query(query, file_path, "mri")

                    st.markdown(f"<p class='red-text'>{response}</p>", unsafe_allow_html=True)
                    st.markdown("<h3 class='maroon-text'>AI Explanation</h3>", unsafe_allow_html=True)

                    # ✅ Check if explanation is a valid Matplotlib figure before rendering
                    if isinstance(explanation_fig, plt.Figure):
                        st.pyplot(explanation_fig)
                    else:
                        st.warning("MRI explanation could not be generated.")

                else:
                    st.error("Unsupported file type.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
