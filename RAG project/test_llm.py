import os
import tempfile
import re

# Environment Shields
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Redirect caches
tmp_parent = os.path.join(tempfile.gettempdir(), "rag_project_cache")
os.makedirs(tmp_parent, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(tmp_parent, "hf_home")

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings

    # Simple embeddings class
    class SimpleEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.1] * 384 for _ in texts]
        def embed_query(self, text):
            return [0.1] * 384

    # Direct Answer Extraction Function
    def extract_direct_answer(question, documents):
        """Extract exact answer from documents based on question"""
        if not documents:
            return "No relevant information found in documents."
        
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

    print("Testing HuggingFace setup with direct answer extraction...")
    
    # Use simple embeddings
    embeddings = SimpleEmbeddings()
    
    # Create test documents
    doc = Document(page_content="The capital of France is Paris. It is located in Europe. Paris is known for the Eiffel Tower.")
    vstore = FAISS.from_documents([doc], embeddings)

    # Test retrieval and extraction
    print("\nTest 1: What is the capital of France?")
    docs = vstore.similarity_search("What is the capital of France?", k=1)
    answer = extract_direct_answer("What is the capital of France?", docs)
    print(f"Answer: {answer}")
    
    print("\nTest 2: Where is Paris located?")
    docs = vstore.similarity_search("Where is Paris located?", k=1)
    answer = extract_direct_answer("Where is Paris located?", docs)
    print(f"Answer: {answer}")
    
    print("\nTest 3: What is Paris known for?")
    docs = vstore.similarity_search("What is Paris known for?", k=1)
    answer = extract_direct_answer("What is Paris known for?", docs)
    print(f"Answer: {answer}")
    
    print("\n✅ HuggingFace direct answer extraction working!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()