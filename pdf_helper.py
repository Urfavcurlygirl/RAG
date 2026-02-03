import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- FONCTION POUR PDFS MULTIPLES ---


def create_db_from_pdfs(pdf_files):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)

    for pdf in pdf_files:
        # Sauvegarde temporaire car PyPDFLoader lit un chemin fichier
        with open("temp.pdf", "wb") as f:
            f.write(pdf.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        all_docs.extend(text_splitter.split_documents(pages))

    db = FAISS.from_documents(all_docs, embeddings)
    return db

# --- FONCTION POUR YOUTUBE ---


def create_db_from_youtube(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db

# --- FONCTION RÉPONSE AVEC MÉMOIRE ---


def get_chat_response(db, user_query, chat_history):
    # Recherche sémantique
    docs = db.similarity_search(user_query, k=3)
    context = " ".join([d.page_content for d in docs])

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    # Prompt Template qui gère le contexte ET l'historique
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tu es un assistant expert. Réponds en utilisant UNIQUEMENT le contexte suivant :\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "input": user_query,
        "context": context,
        "chat_history": chat_history
    })

    return response.content
