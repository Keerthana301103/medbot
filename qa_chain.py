from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_qa_chain(report_text):
    """Creates a medical report Q&A system handling multi-page reports."""
    llm = Ollama(model="mistral")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Improved text splitting for multi-page reports
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([report_text])

    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,  # ✅ Pass retriever correctly
        memory=memory
    )

    return qa_chain
