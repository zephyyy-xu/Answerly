import os
import tempfile
import streamlit as st
import sys
import logging
from typing import List
import time
import requests
import ollama


sys.path.append(r"C:\Users\krish\AppData\Local\Programs\Python\Python311\Lib\site-packages")


import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from streamlit.runtime.uploaded_file_manager import UploadedFile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# System prompt for the LLM
SYSTEM_PROMPT = """You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a context  contex will be passed as "Context:" user question will be passed as "Question:"  To answer the question:
1. Throughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehesive, covering all the relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet point or numbered lists where appropriate to break down complex information.
4. If relevant, include any heading or subheadings to structure your response.
5. Ensure proper grammer, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. And if the user ask anything like translation or simple tasks then do it except adult content"""

def set_custom_style():
    """Set custom CSS styles for the application"""
    st.markdown("""
        <style>
        /* Modern Theme Colors */
        :root {
            --primary: #6C63FF;
            --secondary: #8B85FF;
            --background: #FF7B7B;
            --text: #000103;
            --accent: #84ED77;
            --surface: #ffffff;
            --border: #e5e7eb;
        }

        /* Blinking placeholder animation */
        @keyframes blink {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }

        /* Global styles */
        .stApp {
            background-color: var(--background);
            font-family: 'Inter', -apple-system, sans-serif;
        }

        /* Clean Guide Box */
        .quick-guide {
            background: var(--accent);
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
            border: none;
            transition: all 0.2s ease;
        }

        .guide-title {
            color: var(--text);
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }

        .guide-step {
            color: var(--text);
            margin: 8px 0;
            font-size: 13px;
            display: flex;
            align-items: center;
            opacity: 0.8;
        }

        /* Status message styles */
        .status-message {
            font-size: 14px;
            font-weight: 500;
            margin: 8px 0;
            text-align: center;
            padding: 8px;
            border-radius: 6px;
        }

        .status-success {
            color: #84ED77 !important;
        }

        .status-error {
            color: #FF4B4B !important;
        }

        /* Modern Query Interface */
        .query-interface {
            background: var(--primary);
            padding: 24px;
            border-radius: 16px;
            margin-bottom: 24px;
            text-align: center;
        }

        .neural-title {
            font-family: 'Inter', sans-serif;
            color: white;
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
            justify-content: center;
            text-align: center;
        }

        /* Text Area styling */
        .stTextArea > div > div {
            background-color: white !important;
            border: none !important;
            border-radius: 12px;
            color: black !important;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            transition: all 0.2s ease;
            margin-bottom: 20px;
        }

        /* Ensure textarea input text is black */
        .stTextArea textarea {
            color: black !important;
        }

        /* Blinking placeholder */
        .stTextArea textarea::placeholder {
            color: #6B7280 !important;
            opacity: 1;
            animation: blink 1.5s infinite;
        }

        /* Button styling */
        .stButton > button {
            background-color: var(--primary) !important;
            color: white !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
            font-size: 14px;
            text-transform: none;
        }

        /* File uploader styling */
        .stFileUploader > div {
            background-color: #1E1E1E !important;
            border: 2px dashed var(--border) !important;
            border-radius: 12px;
            padding: 16px !important;
        }

        /* Hide empty containers */
        div:empty {
            display: none !important;
        }

        /* Style success messages specifically */
        .stSuccess {
            background-color: var(--accent) !important;
            color: black !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
        }
        </style>
    """, unsafe_allow_html=True)

def check_ollama_status():
    """Check if Ollama is running and responsive"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def process_document(uploaded_file: UploadedFile) -> List[Document]:
    """Process uploaded document with smaller chunks"""
    try:
        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ".", "?", "!", " "]
        )
        return text_splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Error processing document: {e}")
        logging.error(f"Error processing document: {e}")
        return []
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def get_vector_collection() -> chromadb.Collection:
    """Get or create a vector collection"""
    try:
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text"
        )
        chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
        return chroma_client.get_or_create_collection(
            name="rag_app",
            embedding_function=ollama_ef
        )
    except Exception as e:
        st.error(f"Error initializing vector collection: {e}")
        logging.error(f"Error initializing vector collection: {e}")
        raise


def add_documents_in_batches(collection, documents, metadatas, ids, batch_size=5):
    """Add documents to collection in small batches"""
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch_docs = documents[i:end_idx]
        batch_meta = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                collection.upsert(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
                progress = (end_idx) / len(documents)
                progress_bar.progress(progress)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 * (attempt + 1))

def add_to_vector_collection(all_splits: List[Document], file_name: str):
    """Add documents to vector collection"""
    try:
        if not check_ollama_status():
            st.error("Ollama service is not running. Please start Ollama first.")
            return

        collection = get_vector_collection()
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")

        add_documents_in_batches(collection, documents, metadatas, ids, batch_size=5)
        
        return True
    except Exception as e:
        logging.error(f"Error adding to vector store: {e}")
        return False

def query_collection(prompt: str, n_results: int = 10):
    """Query the vector collection"""
    try:
        if not check_ollama_status():
            st.error("Ollama service is not running. Please start Ollama first.")
            return None

        collection = get_vector_collection()
        results = collection.query(
            query_texts=[prompt],
            n_results=n_results
        )
        return results
    except Exception as e:
        st.error(f"Error querying vector store: {e}")
        logging.error(f"Error querying vector store: {e}")
        return None

def call_llm(context: str, prompt: str):
    """Call LLaMA model with streaming response"""
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {prompt}",
                },
            ],
        )
        
        for chunk in response:
            if chunk["done"] is False:
                yield chunk["message"]["content"]
            else:
                break
    except Exception as e:
        yield f"Error generating response: {str(e)}"

def main():
    st.set_page_config(page_title="RAG Q&A", layout="wide")
    set_custom_style()
    
    if not check_ollama_status():
        st.error("⚠️ Ollama service is not running. Please start Ollama first.")
        st.info("Run 'ollama run nomic-embed-text' and 'ollama run llama2:13b' in your terminal to start the services.")
        return

    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("""
            <div class="quick-guide">
                <div class="guide-title">QUICK GUIDE</div>
                <div class="guide-step">Upload your PDF files</div>
                <div class="guide-step">Process the documents</div>
                <div class="guide-step">Ask your questions</div>
                <div class="guide-step">Get AI responses</div>
            </div>
            """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)
        
        # Create a placeholder for the status message
        status_placeholder = st.empty()
        
        if uploaded_files:
            process = st.button("Process Files")
            
            if process:
                try:
                    global progress_bar
                    progress_bar = st.progress(0)
                    
                    success = True
                    for uploaded_file in uploaded_files:
                        normalize_uploaded_file_name = uploaded_file.name.translate(
                            str.maketrans({"-": "_", ".": "_", " ": "_"})
                        )
                        all_splits = process_document(uploaded_file)

                        if all_splits:
                            if not add_to_vector_collection(all_splits, normalize_uploaded_file_name):
                                success = False
                                break
                    
                    progress_bar.empty()
                    
                    if success:
                        status_placeholder.markdown('<p class="status-message status-success">File Processed Successfully! Go and Rock! 🚀</p>', unsafe_allow_html=True)
                    else:
                        status_placeholder.markdown('<p class="status-message status-error">Sorry that\'s too much for me... 😅</p>', unsafe_allow_html=True)
                except Exception as e:
                    status_placeholder.markdown('<p class="status-message status-error">Sorry that\'s too much for me... 😅</p>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="query-interface">
                <div class="neural-title">⚡Answerly</div>
                <p style='color: white; font-family: "Inter", sans-serif; font-size: 14px; opacity: 0.9;'>
                    Ask questions about your documents
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        prompt = st.text_area("", height=120, placeholder="Type your question here...")
        
        col_space1, col_button, col_space2 = st.columns([2, 1, 2])
        with col_button:
            ask = st.button("Ask")

        if ask and prompt:
            results = query_collection(prompt)
            if results and results['documents']:
                context = " ".join(results['documents'][0])
                response_container = st.empty()
                full_response = ""
                
                for chunk in call_llm(context=context, prompt=prompt):
                    full_response += chunk
                    response_container.markdown(full_response + "▮")
                response_container.markdown(full_response)
            else:
                st.info("No matching information found. Try another question.")
                
if __name__ == "__main__":
    main()
    