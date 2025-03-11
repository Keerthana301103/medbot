import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract text from a multi-page PDF."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:  # Loop through all pages
            text += page.extract_text() + "\n"
    return text.strip()
