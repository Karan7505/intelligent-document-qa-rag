# ============================================
# STREAMLIT USER INTERFACE
# For complete beginners
# ============================================

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from responsible_ai import ResponsibleAI

# Load environment variables
load_dotenv()

# ============================================
# SESSION STATE
# ============================================

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="📚 Smart Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# PAGE TITLE AND INTRODUCTION
# ============================================

st.title("📚 Intelligent Document Q&A System")
st.markdown("""
    ### Welcome! 👋
    
    This app uses **Retrieval-Augmented Generation (RAG)** to answer questions about your documents with:
    - ✅ Accurate answers based on YOUR documents
    - ✅ Source citations showing where the answer came from
    - ✅ Confidence scores for answer reliability
    - ✅ Hallucination detection to catch made-up information
    
    **How it works**:
    1. Upload a PDF document
    2. Ask any question about it
    3. Get an accurate answer with sources
""")

st.markdown("---")

# ============================================
# SIDEBAR - CONFIGURATION
# ============================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model Settings")
    temperature = st.slider(
        "Temperature (Lower = More Focused)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower values = more focused answers. Higher values = more creative."
    )
    
    st.subheader("Retrieval Settings")
    num_chunks = st.slider(
        "Number of Document Chunks to Retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant sections to show the AI"
    )

    chunk_size = st.selectbox(
        "Chunk Size",
        options=[500, 1000, 1500],
        index=1,
        help="Choose how large each document chunk should be."
    )

# ============================================
# MAIN INTERFACE
# ============================================

# Two columns layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF document you want to ask questions about"
    )

with col2:
    st.subheader("❓ Ask Questions")
    
    user_question = st.text_input(
        "What would you like to know?",
        placeholder="Example: What are the main topics in this document?",
        help="Ask any question about your uploaded document"
    )

st.markdown("---")

def create_custom_prompt():

    template = """
You are a helpful AI assistant.

Answer questions ONLY using the information provided in the document context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer ONLY using the provided context.
2. Do NOT make assumptions.
3. If the answer is not found, say:
   "I don't have this information in the uploaded document."
4. Keep your answer clear and concise.
5. Mention the relevant document information whenever possible.

ANSWER:
"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# ============================================
# PROCESS AND DISPLAY RESULTS
# ============================================

if uploaded_file:

    # Process the document only if it's a new upload
    if st.session_state.uploaded_file_name != uploaded_file.name:

        with st.spinner("🔄 Processing document..."):

            try:
                # Save uploaded file
                with open("temp_document.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load PDF
                loader = PyPDFLoader("temp_document.pdf")
                documents = loader.load()

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=100
                )
                chunks = splitter.split_documents(documents)
                st.success(
                    f"✅ Created {len(chunks)} chunks using chunk size {chunk_size}"
                )

                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )

                # Vector Store
                vector_store = FAISS.from_documents(
                    chunks,
                    embeddings
                )

                # Retriever
                retriever = vector_store.as_retriever(
                    search_kwargs={"k": num_chunks}
                )

                # LLM
                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile",
                    temperature=temperature
                )

                # Prompt
                prompt = create_custom_prompt()

                # QA Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": prompt
                    }
                )

                st.session_state.qa_chain = qa_chain
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.document_processed = True

                st.success("✅ Document processed successfully!")

            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()

    # Ask questions only after processing
    if user_question and st.session_state.qa_chain:

        with st.spinner("💭 Thinking..."):

            result = st.session_state.qa_chain.invoke(
                {"query": user_question}
            )

        st.markdown("---")
        st.header("📊 Results")

        st.subheader("💬 Answer")
        st.info(result["result"])

        sources = result.get("source_documents", [])

        confidence = ResponsibleAI.calculate_confidence(
            sources
        )

        safety_report = ResponsibleAI.comprehensive_safety_check(
            user_question,
            result["result"],
            sources,
            confidence
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "📊 Confidence",
                f"{confidence * 100:.1f}%"
            )

        with col2:
            st.metric(
                "📚 Sources",
                len(sources)
            )

        with col3:
            st.metric(
                "🛡️ Reliability",
                "Safe"
                if not safety_report["potential_hallucination"]
                else "Caution"
            )

        st.subheader("📚 Sources")

        if sources:

            for i, doc in enumerate(sources, 1):

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                clean_text = " ".join(doc.page_content.split())

                with st.expander(f"📄 Source {i} (Page {page})"):

                    st.write(clean_text)

        else:

            st.warning("No sources found.")

        st.subheader("💡 Recommendation")

        st.info(
            safety_report["recommendation"]
        )

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
    ### 🎯 How This Works:
    1. **RAG (Retrieval-Augmented Generation)**: Finds relevant document sections
    2. **Vector Database**: Uses AI to understand meaning (not just keywords)
    3. **LLM**: Generates accurate answers based on context
    4. **Responsible AI**: Checks reliability and prevents hallucinations
    
    Made with ❤️ using LangChain, Groq, and Streamlit
""")

# Cleanup
import os
if os.path.exists("temp_document.pdf"):
    os.remove("temp_document.pdf")