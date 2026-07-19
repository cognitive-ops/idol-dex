"""Streamlit GUI for JAV RAG chatbot."""
import streamlit as st
import requests
import json
from datetime import datetime

# Streamlit config
st.set_page_config(
    page_title="JAV Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stChatMessage { margin: 1rem 0; }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar config
with st.sidebar:
    st.title("⚙️ Configuration")
    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="FastAPI server URL"
    )
    top_k = st.slider("Results per query", 1, 10, 5)
    st.divider()
    st.info("💡 Tip: Make sure FastAPI server is running first!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False

# Check API health
def check_api():
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Main chat interface
st.title("🎬 JAV Information Chatbot")
st.markdown("*Powered by Claude RAG + FAISS semantic search*")

# Health check
if not check_api():
    st.error("❌ FastAPI server not reachable. Start it with: `python main.py`")
    st.stop()
else:
    st.success("✅ Connected to API")

st.divider()

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for i, src in enumerate(msg["sources"], 1):
                        with st.container():
                            st.markdown(f"**{i}. {src.get('title', 'Unknown')}**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"👥 **Actors:** {src.get('actors', 'N/A')}")
                                st.markdown(f"📅 **Date:** {src.get('date', 'N/A')}")
                            with col2:
                                if src.get("url"):
                                    st.markdown(f"[🔗 View on R18]({src['url']})")
                                similarity = src.get("similarity", 0)
                                st.progress(similarity, text=f"Relevance: {similarity:.1%}")

st.divider()

# Query input
query = st.chat_input("Ask about JAV titles, actors, releases...")

if query:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching database..."):
            try:
                response = requests.post(
                    f"{api_url}/chat",
                    json={"query": query},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Display answer
                st.markdown(data.get("answer", "No answer generated"))

                # Display sources
                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved {len(sources)} sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**{i}. {src.get('title', 'Unknown')}**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"👥 Actors: {src.get('actors', 'N/A')}")
                                st.markdown(f"📅 Date: {src.get('date', 'N/A')}")
                            with col2:
                                if src.get("url"):
                                    st.markdown(f"[🔗 View]({src['url']})")
                                similarity = src.get("similarity", 0)
                                st.progress(similarity, text=f"Relevance: {similarity:.1%}")

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data.get("answer", ""),
                    "sources": sources
                })

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try a simpler query.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Is the server running?")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Sidebar actions
st.sidebar.divider()
st.sidebar.subheader("🛠️ Tools")

if st.sidebar.button("🔄 Ingest Data", use_container_width=True):
    with st.spinner("Scraping sources and building index (this may take a few minutes)..."):
        try:
            response = requests.post(f"{api_url}/ingest", timeout=60)
            data = response.json()
            st.sidebar.success(f"✅ Indexed {data.get('indexed', 0)} documents")
        except Exception as e:
            st.sidebar.error(f"❌ Ingest failed: {str(e)}")

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("""
### About
**JAV RAG Chatbot** — Information retrieval for Japanese Adult Videos.

- **Search**: Semantic search via FAISS
- **Retrieve**: Metadata from r18.com, javlibrary, dmm
- **Answer**: Claude LLM with RAG context

Built with FastAPI + FAISS + Claude + Streamlit
""")
