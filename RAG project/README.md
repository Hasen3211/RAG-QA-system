# RAG Document Question Answering System

## 📌 Project Overview

This project is a **Retrieval-Augmented Generation (RAG) based Document Question Answering System** built using Python and modern NLP frameworks. The system allows users to upload documents and ask questions related to the document content. It retrieves the most relevant information and provides accurate answers based on the document context.

\---

## 🚀 Features

* Upload and process PDF documents
* Convert documents into embeddings
* Store embeddings using FAISS vector database
* Ask questions related to the document
* Retrieve context-aware answers from the document
* Simple and interactive web interface using Streamlit

\---

## 🛠️ Technologies Used

* Python
* Streamlit
* LangChain
* HuggingFace Transformers
* Sentence Transformers
* FAISS Vector Database
* PyPDF

\---

## 🧠 How the System Works

1. User uploads a document (PDF).
2. The document text is extracted.
3. The text is split into smaller chunks.
4. Each chunk is converted into vector embeddings.
5. Embeddings are stored in a FAISS vector database.
6. When a user asks a question, the system retrieves the most relevant document chunks.
7. The retrieved content is used to generate the final answer.

\---

## 📂 Project Structure

```
rag-document-qa-system
│
├── app.py                # Streamlit application
├── test\\\_llm.py           # LLM testing script
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

\---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Hasen3211/RAG-QA-system

```

Navigate to the project folder:

```
cd rag-document-qa-system
```

Install required dependencies:

```
pip install -r requirements.txt
```

\---

## ▶️ Running the Application

Run the Streamlit app:

```
streamlit run app.py
```

## The application will start in your browser.

## 📊 Future Improvements

* Support for multiple document formats
* Improved answer generation using advanced LLMs
* Deployment on cloud platforms
* Multi-user support

\---

## 👩‍💻 Author

**Hasen Basha**
GitHub:
https://github.com/Hasen3211
---

## ⭐ Acknowledgements

This project uses open-source libraries and frameworks such as LangChain, HuggingFace, FAISS, and Streamlit to build an efficient document-based question answering system.

