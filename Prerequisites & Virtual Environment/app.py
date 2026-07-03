import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

st.set_page_config(page_title="CloudNex AI Portal", page_icon="🤖", layout="centered")
st.title("🤖 CloudNex AI Compliance Portal")
st.caption("Secure, offline corporate policy intelligence engine with real-time streaming")
st.markdown("---")

base_dir = os.path.dirname(os.path.abspath(__file__))
db_storage_dir = os.path.join(base_dir, "db_storage")

@st.cache_resource
def initialize_rag_backend():
    if not os.path.exists(db_storage_dir):
        return None, None
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=db_storage_dir, embedding_function=embeddings)
    # Uses resource-optimized local architectures (e.g., llama3.1:8b or phi3)
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.2)
    return vector_db, llm

vector_db, llm = initialize_rag_backend()

if vector_db is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a policy or compliance question..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            results = vector_db.similarity_search(user_query, k=2)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
            
            # Ironclad Prompt Guardrails to Prevent Hallucinations
            system_prompt = f"You are an internal corporate AI assistant for Xcelerate. Answer accurately using ONLY the context provided below. If the answer cannot be found, reply exactly with: 'I cannot find that information in the current compliance documents.'\n\nCONTEXT:\n{context_text}\n\nUSER QUESTION:\n{user_query}\n\nREPLY:\n"
            
            placeholder = st.empty()
            full_response = ""
            
            # Real-time token streaming loop
            for chunk in llm.stream(system_prompt):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})
