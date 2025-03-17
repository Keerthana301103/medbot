from ollama_model import chat_with_ollama
from doc_process import extract_text_from_pdf
from mri_classification import predict_mri
from explainability import generate_gradcam_explanation

def process_user_query(query, uploaded_file, file_type):
    """Processes user queries based on uploaded file type."""
    
    if file_type == "pdf":
        text_content = extract_text_from_pdf(uploaded_file)
        response = chat_with_ollama(f"Summarize this medical report and answer: {query}\n{text_content}")
        return response, None

    elif file_type == "mri":
        # Ensure MRI model allows gradient computation for Grad-CAM
        prediction, output = predict_mri(uploaded_file)  

        # Generate Grad-CAM explanation
        explanation_fig = generate_gradcam_explanation(uploaded_file)  

        # Construct response
        response = chat_with_ollama(f"The MRI prediction is '{prediction}'. User asks: {query}")

        return response, explanation_fig  

    return "Invalid file type", None  
