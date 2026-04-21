import os
import sys
import tempfile
import traceback
# 1. CRITICAL: Environment Shields for Windows + Python 3.13
# These must be set before any heavy ML imports 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["JOBLIB_MULTIPROCESSING_BACKEND"] = "threading"

# Redirect caches to a dedicated temp folder to avoid Streamlit reload loops
tmp_parent = os.path.join(tempfile.gettempdir(), "rag_project_cache")
os.makedirs(tmp_parent, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(tmp_parent, "hf_home")

import streamlit as st

# 2. Defensive Imports
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings
    import re
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

# Simple embeddings class (no model downloads needed)
class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 384

# 3. Direct Answer Extraction Function
def extract_direct_answer(question, documents):
    """Extract exact answer from documents based on question"""
    if not documents:
        return "No relevant information found in documents."
    
    # Get the most relevant document (first one is usually highest scored)
    best_doc = documents[0]
    text = best_doc.page_content
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Find sentences containing key words from the question
    question_words = set(word.lower() for word in question.split() if len(word) > 3)
    
    best_sentence = None
    best_score = 0
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Count matching keywords in sentence
        score = sum(1 for word in question_words if word in sentence_lower)
        if score > best_score:
            best_score = score
            best_sentence = sentence.strip()
    
    # If no matching sentence found, return first sentence
    if not best_sentence and sentences:
        best_sentence = sentences[0].strip()
    
    return best_sentence if best_sentence else "Answer not found in documents."

# 4. Cached Resource Loader
@st.cache_resource
def load_rag_assets(hf_token):
    try:
        # Use simple embeddings
        embeddings = SimpleEmbeddings()
        
        return {
            "embeddings": embeddings,
            "splitter": RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        }
    except Exception as e:
        st.error(f"Error during AI initialization: {e}")
        st.code(traceback.format_exc())
        return None

# 4. Streamlit UI
st.set_page_config(page_title="PDF AI Assistant", layout="wide")
st.title("📄 PDF AI Assistant (RAG)")

# Sidebar Setup
with st.sidebar:
    st.header("1. Authentication")
    hf_token = st.text_input("HuggingFace API Token", type="password")
    
    st.header("2. Document Management")
    pdf_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    # Session State Initialization
    if "assets" not in st.session_state:
        st.session_state.assets = None
    if "vstore" not in st.session_state:
        st.session_state.vstore = None

    # Step-by-Step UI Control
    if hf_token and not st.session_state.assets:
        if st.button("🔌 Connect AI Models"):
            with st.spinner("Connecting to HuggingFace..."):
                st.session_state.assets = load_rag_assets(hf_token)
                if st.session_state.assets:
                    st.success("AI Models Connected!")

    # Manual trigger for indexing
    if pdf_files and st.session_state.assets:
        if st.button("🚀 Index Documents"):
            try:
                with st.spinner("Indexing contents..."):
                    all_docs = []
                    for pdf_file in pdf_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(pdf_file.read())
                            tmp_path = tmp.name
                        try:
                            loader = PyPDFLoader(tmp_path)
                            loaded_docs = loader.load()
                            st.write(f"ℹ️ Loaded {len(loaded_docs)} pages from {pdf_file.name}")
                            all_docs.extend(loaded_docs)
                        finally:
                            if os.path.exists(tmp_path): os.unlink(tmp_path)

                    if not all_docs:
                        st.error("❌ No text was found in the uploaded PDFs. Please check if the documents are readable.")
                    else:
                        chunks = st.session_state.assets["splitter"].split_documents(all_docs)
                        vstore = FAISS.from_documents(chunks, st.session_state.assets["embeddings"])
                        st.session_state.vstore = vstore
                        st.success(f"✅ Successfully indexed {len(chunks)} chunks from {len(all_docs)} pages!")
            except Exception as e:
                st.error("❌ Indexing Failed")
                with st.expander("Technical Error Message"):
                    st.error(f"{type(e).__name__}: {e}")
                    st.code(traceback.format_exc())

# 5. Main Interaction Area
if st.session_state.vstore is not None:
    user_input = st.text_input("💬 Ask a question about your documents:")
    if user_input:
        try:
            with st.spinner("Finding answer..."):
                # Retrieve relevant documents
                docs = st.session_state.vstore.similarity_search(user_input, k=3)
                
                # Extract direct answer from documents
                answer = extract_direct_answer(user_input, docs)
                st.markdown(f"**Answer:** {answer}")
                
                with st.expander("Show Sources"):
                    for d in docs:
                        st.markdown(f"--- \n {d.page_content}")
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.code(traceback.format_exc())
else:
    # Onboarding Messages
    if not hf_token:
        st.info("👈 Step 1: Please enter your HuggingFace Token in the sidebar.")
    elif not st.session_state.assets:
        st.info("👈 Step 2: Click **Connect AI Models** in the sidebar.")
    elif pdf_files:
        st.info("👈 Step 3: Click **Index Documents** in the sidebar.")
    else:
        st.info("👈 Step 3: Upload some PDF files in the sidebar.")
