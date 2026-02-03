import streamlit as st
import pdf_helper as helper
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="RAG Chatbot Multi-Source", layout="wide")
st.title("💬 Super Assistant RAG")

# --- INITIALISATION DES VARIABLES ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Pour stocker la conversation
if "db" not in st.session_state:
    st.session_state.db = None  # Pour stocker la mémoire documentaire

# --- BARRE LATÉRALE : CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Source de données :", ["Plusieurs PDFs", "Vidéo YouTube"])

    if mode == "Plusieurs PDFs":
        files = st.file_uploader(
            "Upload tes fichiers PDF", type="pdf", accept_multiple_files=True)
        if st.button("Indexer les documents"):
            if files:
                with st.spinner("Analyse en cours..."):
                    st.session_state.db = helper.create_db_from_pdfs(files)
                st.success("PDFs indexés !")
            else:
                st.error("Ajoute au moins un PDF.")

    else:
        url = st.text_input("URL YouTube :")
        if st.button("Extraire la transcription"):
            if url:
                with st.spinner("L'IA écoute la vidéo..."):
                    st.session_state.db = helper.create_db_from_youtube(url)
                st.success("Vidéo prête !")

    if st.button("🗑️ Effacer le chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- ZONE DE CHAT (STYLE CHAT GPT) ---

# Afficher les anciens messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Saisie du nouveau message
if prompt := st.chat_input("Pose-moi une question..."):
    if st.session_state.db is None:
        st.warning(
            "⚠️ Charge d'abord une source (PDF ou YouTube) dans la barre latérale.")
    else:
        # 1. Afficher et stocker le message utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Obtenir la réponse de l'IA
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                full_response = helper.get_chat_response(
                    st.session_state.db,
                    prompt,
                    st.session_state.chat_history
                )
                st.markdown(full_response)

        # 3. Mettre à jour l'historique
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=full_response))
