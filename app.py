# ============================================
# DOCUMENT Q&A SYSTEM WITH RAG
# For Complete Beginners
# ============================================

# Import statements (bringing in the libraries we installed)
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

# ============================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ============================================
# This reads our .env file and gets the API keys

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if keys exist
if not groq_api_key or not google_api_key:
    print("❌ ERROR: API keys not found in .env file!")
    print("Make sure your .env file has:")
    print("  GROQ_API_KEY=your_key")
    print("  GOOGLE_API_KEY=your_key")
    exit()

print("✅ API keys loaded successfully!")

# ============================================
# STEP 2: DOCUMENT LOADER FUNCTION
# ============================================
# This function reads a PDF file

def load_pdf(file_path):
    """
    Load a PDF document
    Input: file_path - path to your PDF file
    Output: list of document pieces
    """
    print(f"📄 Loading PDF: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"✅ PDF loaded! Contains {len(documents)} pages")
        return documents
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return None

# ============================================
# STEP 3: TEXT SPLITTING FUNCTION
# ============================================
# This breaks the document into smaller chunks

def split_documents(documents):
    """
    Split documents into smaller chunks
    Why? Because we can't process huge documents at once
    
    Analogy: Instead of reading a 300-page book at once,
    we read it in 500-word chunks
    
    Input: documents from load_pdf()
    Output: list of document chunks
    """
    print("\n✂️ Splitting documents into chunks...")
    
    # RecursiveCharacterTextSplitter = Smart chunking
    # chunk_size=1000 = each chunk has 1000 characters
    # overlap=100 = chunks overlap by 100 characters (so we don't lose context)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ============================================
# STEP 4: CREATE EMBEDDINGS
# ============================================
# Convert text to numbers that represent meaning

def create_embeddings():
    """
    Create embeddings model
    
    What's an embedding?
    - Takes text like "I love pizza"
    - Converts to list of numbers: [0.23, -0.45, 0.67, ...]
    - Similar meanings = similar numbers
    - We use HuggingFace's free model
    
    Output: embeddings model
    """
    print("\n🧠 Creating embeddings model...")
    
    # Using 'all-MiniLM-L6-v2' - a free, fast model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    print("✅ Embeddings model ready!")
    return embeddings

# ============================================
# STEP 5: CREATE VECTOR DATABASE
# ============================================
# Store embeddings in a searchable database

def create_vector_store(chunks, embeddings):
    """
    Create FAISS vector database
    
    What's FAISS?
    - Facebook AI Similarity Search
    - Super fast database for finding similar things
    
    Analogy: Like a library catalog system
    - Input: A book topic (embedding)
    - Output: All similar books found quickly
    
    Input: chunks + embeddings
    Output: vector_store (searchable database)
    """
    print("\n🗄️ Creating vector database...")
    print(f"Adding {len(chunks)} chunks to database...")
    
    # Create FAISS from documents
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print("✅ Vector database created!")
    return vector_store

# ============================================
# STEP 6: CREATE RETRIEVER
# ============================================
# This searches the database for relevant chunks

def create_retriever(vector_store):
    """
    Create a retriever from vector store
    
    What's a retriever?
    - Searches the vector database
    - Finds top 5 most relevant chunks
    - Returns them to the LLM
    
    Input: vector_store
    Output: retriever (search engine)
    """
    print("\n🔍 Creating retriever...")
    
    # Convert vector store to retriever
    # search_kwargs={"k": 5} means "find top 5 results"
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    
    print("✅ Retriever ready!")
    return retriever

# ============================================
# STEP 7: CREATE LLM (AI Model)
# ============================================
# The brain that answers questions

def create_llm():
    """
    Create Groq LLM
    
    What's an LLM?
    - Large Language Model
    - An AI that understands text and generates responses
    
    Why Groq?
    - Free API
    - Super fast
    - Reliable
    
    Output: llm (the AI brain)
    """
    print("\n🤖 Initializing Groq LLM...")
    
    # Using llama-3.1-70b-versatile model
    # It's free and very powerful
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3  # Lower = more focused answers
    )
    
    print("✅ LLM ready!")
    return llm

# ============================================
# STEP 8: CREATE CUSTOM PROMPT
# ============================================
# Tell the AI how to behave

def create_custom_prompt():
    """
    Create a custom prompt template
    
    What's a prompt?
    - Instructions for the AI
    - Like "Answer ONLY from these documents"
    
    We'll tell the AI:
    1. Use only the provided context
    2. Cite sources
    3. Say "I don't know" if unsure
    
    Output: prompt template
    """
    print("\n📝 Creating custom prompt...")
    
    # Custom template
    template = """
You are a helpful AI assistant. Answer questions based ONLY on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer ONLY using information from the context
2. If the answer is not in the context, say "I don't have this information in the documents"
3. Always cite which document part you're referencing
4. Do NOT make assumptions or calculations
5. If the exact answer is not stated, say:
   "The document does not explicitly state this."
6. Be concise and clear

ANSWER:
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    print("✅ Prompt ready!")
    return prompt

# ============================================
# STEP 9: CREATE QA CHAIN
# ============================================
# Connect retriever + LLM

def create_qa_chain(llm, retriever):
    """
    Create the complete RAG chain
    
    This connects:
    1. Retriever (finds relevant documents)
    2. LLM (generates answer)
    
    Flow:
    Question → Retriever → Find documents → LLM → Answer
    
    Input: llm + retriever
    Output: qa_chain (the complete system)
    """
    print("\n⛓️ Creating QA chain...")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    
    print("✅ QA chain ready!")
    return qa_chain

# ============================================
# MAIN FUNCTION
# ============================================
# This runs everything in order

def main():
    """
    Main function - runs the entire system
    """
    print("=" * 60)
    print("🚀 DOCUMENT Q&A SYSTEM - INITIALIZATION")
    print("=" * 60)
    
    # Step 1: Load PDF
    pdf_path = "sample_document.pdf"  # We'll create this
    documents = load_pdf(pdf_path)
    if not documents:
        return
    
    # Step 2: Split documents
    chunks = split_documents(documents)
    
    # Step 3: Create embeddings
    embeddings = create_embeddings()
    
    # Step 4: Create vector store
    vector_store = create_vector_store(chunks, embeddings)
    
    # Step 5: Create retriever
    retriever = create_retriever(vector_store)
    
    # Step 6: Create LLM
    llm = create_llm()
    
    # Step 7: Create QA chain
    qa_chain = create_qa_chain(llm, retriever)
    
    # ============================================
    # TEST THE SYSTEM
    # ============================================
    print("\n" + "=" * 60)
    print("💬 TESTING THE SYSTEM")
    print("=" * 60)
    
    # Test questions
    print("\nType your questions below.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask your question: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        print("-" * 60)

        # ===========================
        # Responsible AI - Validate Query
        # ===========================

        validation = ResponsibleAI.validate_query(question)

        if not validation["is_valid"]:
            print(f"\n❌ Query Rejected")
            print(f"Reason: {validation['reason']}")
            continue

        try:

            # ===========================
            # Ask the RAG system
            # ===========================

            result = qa_chain.invoke({"query": question})

            answer = result["result"]

            source_documents = result.get("source_documents", [])

            # ===========================
            # Responsible AI Checks
            # ===========================

            confidence = ResponsibleAI.calculate_confidence(
                source_documents
            )

            safety_report = ResponsibleAI.comprehensive_safety_check(
                question,
                answer,
                source_documents,
                confidence
            )

            # ===========================
            # Display Answer
            # ===========================

            print("\n✅ Answer")
            print(answer)

            print(f"\n📊 Confidence Score: {confidence * 100:.1f}%")

            print()

            print(
                ResponsibleAI.format_sources(source_documents)
            )

            print(
                f"\n💡 Recommendation: {safety_report['recommendation']}"
            )

        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM WORKING! Ready for Streamlit UI")
    print("=" * 60)

# ============================================
# RUN THE PROGRAM
# ============================================

if __name__ == "__main__":
    main()