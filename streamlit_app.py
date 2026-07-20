"""Streamlit GUI for IMDb RAG chatbot."""
import logging
import streamlit as st
import requests
import json
from datetime import datetime

from src.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Streamlit config
st.set_page_config(
    page_title="IMDb Chatbot",
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
        value=settings.api_base_url,
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
        logger.debug("health check %s -> %d", api_url, response.status_code)
        return response.status_code == 200
    except Exception as e:
        logger.debug("health check %s failed: %s", api_url, e)
        return False

# Main chat interface
st.title("🎬 IMDb Information Chatbot")
st.markdown("*Powered by Claude RAG + FAISS semantic search*")

# Health check
if not check_api():
    st.error("❌ FastAPI server not reachable. Start it with: `python main.py`")
    st.stop()
else:
    st.success("✅ Connected to API")

st.divider()


def render_source(i: int, src: dict):
    """Render one source card — person profile or title, depending on payload shape."""
    st.markdown(f"**{i}. {src.get('title', 'Unknown')}**")
    col1, col2 = st.columns(2)
    is_person = "name" in src or "birth_year" in src
    with col1:
        if is_person:
            st.markdown(f"🎂 **Born:** {src.get('birth_year', 'N/A')}")
            if src.get("death_year"):
                st.markdown(f"⚰️ **Died:** {src.get('death_year')}")
            st.markdown(f"🎭 **Professions:** {src.get('professions', 'N/A')}")
        else:
            st.markdown(f"👥 **Cast:** {src.get('actors', 'N/A')}")
            st.markdown(f"📅 **Year:** {src.get('year', 'N/A')}")
            st.markdown(f"⭐ **Rating:** {src.get('rating', 'N/A')}")
    with col2:
        if src.get("url"):
            st.markdown(f"[🔗 View on IMDb]({src['url']})")
        if is_person and src.get("known_for"):
            known_for = ", ".join(src["known_for"][:5])
            st.markdown(f"🎬 **Known for:** {known_for}")
        similarity = src.get("similarity", 0)
        st.progress(similarity, text=f"Relevance: {similarity:.1%}")


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
                            render_source(i, src)

st.divider()

# Query input
query = st.chat_input("Ask about movies, TV shows, actors, actresses...")

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
                logger.debug("POST %s/chat query=%r", api_url, query)
                response = requests.post(
                    f"{api_url}/chat",
                    json={"query": query},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                logger.debug("/chat response: %d sources", len(data.get("sources", [])))

                # Display answer
                st.markdown(data.get("answer", "No answer generated"))

                # Display sources
                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved {len(sources)} sources"):
                        for i, src in enumerate(sources, 1):
                            render_source(i, src)

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data.get("answer", ""),
                    "sources": sources
                })

            except requests.exceptions.Timeout:
                logger.warning("/chat request timed out")
                st.error("⏱️ Request timed out. Try a simpler query.")
            except requests.exceptions.ConnectionError:
                logger.warning("/chat connection error to %s", api_url)
                st.error("❌ Cannot connect to API. Is the server running?")
            except Exception as e:
                logger.error("/chat error: %s", e)
                st.error(f"❌ Error: {str(e)}")

# Sidebar actions
st.sidebar.divider()
st.sidebar.subheader("🛠️ Tools")

if st.sidebar.button("🔄 Ingest Titles + Cast", use_container_width=True):
    with st.spinner("Downloading IMDb datasets and building index (this may take a few minutes)..."):
        try:
            logger.debug("POST %s/ingest", api_url)
            response = requests.post(f"{api_url}/ingest", timeout=600)
            data = response.json()
            logger.info("/ingest response: indexed=%s", data.get("indexed", 0))
            st.sidebar.success(f"✅ Indexed {data.get('indexed', 0)} documents")
        except Exception as e:
            logger.error("/ingest failed: %s", e)
            st.sidebar.error(f"❌ Ingest failed: {str(e)}")

if st.sidebar.button("👤 Ingest People Only", use_container_width=True):
    with st.spinner("Loading actor/actress profiles..."):
        try:
            logger.debug("POST %s/ingest/people", api_url)
            response = requests.post(f"{api_url}/ingest/people", timeout=600)
            data = response.json()
            logger.info("/ingest/people response: indexed=%s", data.get("indexed", 0))
            st.sidebar.success(f"✅ Indexed {data.get('indexed', 0)} person profiles")
        except Exception as e:
            logger.error("/ingest/people failed: %s", e)
            st.sidebar.error(f"❌ Ingest failed: {str(e)}")

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("""
### About
**IMDb RAG Chatbot** — Ask about movies, TV shows, actors, and actresses.

- **Search**: Semantic search via FAISS
- **Retrieve**: IMDb non-commercial datasets (datasets.imdbws.com)
- **Answer**: Claude LLM with RAG context

Built with FastAPI + FAISS + Claude + Streamlit
""")
