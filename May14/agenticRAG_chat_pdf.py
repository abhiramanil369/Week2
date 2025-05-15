import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  

load_dotenv()

os.environ["GOOGLE_API_KEY"] = "AIzaSyDBJp_ZXsvzTVpClNNOD0i5D84tRLTBsfU"
genai.configure()  

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_data = pdf.read()
            if not pdf_data:
                raise ValueError(f"File {pdf.name} is empty or invalid.")

            with fitz.open(stream=io.BytesIO(pdf_data), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            raise ValueError(f"Failed to process {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_index"  
    )
    vector_store.persist()
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant designed to answer questions based only on the content of a set of PDF documents provided to you.

    INSTRUCTIONS:
    - Only use the information from the documents to answer.
    - If the information is not present in the documents, respond with: "The information you're asking for is not available in the provided documents."
    - Be concise and accurate. Provide the most relevant excerpt(s) when possible.
    - If asked for summaries or comparisons, do so using only the PDF data.
    - Always cite the page number(s) when referencing information directly.

    DOCUMENT CONTEXT:
    {context}

    USER QUESTION:
    {question}

    YOUR RESPONSE:
    """
    try:
        model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-002", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing conversational chain: {e}")
        return None

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = Chroma(persist_directory="chroma_index", embedding_function=embeddings)
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return

    try:
        docs = new_db.similarity_search(user_question)
        if not docs:
            st.warning("No relevant documents found for your query.")
            return
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return

    try:
        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("### Reply:")
            st.write(response["output_text"])
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using Google Gemini")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload at least one PDF file.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
