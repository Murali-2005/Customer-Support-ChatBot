import os
import tempfile
import streamlit as st

from update_knowledge import update_vector_db
from image_helper import get_image_context
from langchain_helper import (
    create_vector_db,
    get_multimodal_response,
    clear_history
)

st.set_page_config(
    page_title="Customer Service Chatbot",
    page_icon="🤖"
)

st.title("🤖 Multi-Modal Customer Support Chatbot")

VECTOR_DB_PATH = "faiss_index"

# -----------------------------
# Knowledge Base
# -----------------------------

if not os.path.exists(VECTOR_DB_PATH):

    st.warning("Knowledge Base is not initialized. Please click the button below to create it.")

    if st.button("Create Knowledge Base"):

        with st.spinner("Creating Knowledge Base..."):

            create_vector_db()

        st.success("Knowledge Base Created!")
        st.rerun()

else:

    st.success("Knowledge Base Ready")

    if st.button("Update Knowledge Base"):

        with st.spinner("Updating Knowledge Base..."):

            message = update_vector_db()

        st.success(message)
        st.rerun()

st.divider()

# -----------------------------
# User Input
# -----------------------------

db_ready = os.path.exists(VECTOR_DB_PATH)

question = st.text_input(
    "Ask your question",
    disabled=not db_ready,
    placeholder="Please initialize the knowledge base first" if not db_ready else "Type your query here..."
)

uploaded_image = st.file_uploader(
    "Upload Image (Optional)",
    type=["png", "jpg", "jpeg"]
)

image_context = ""

if uploaded_image:

    st.image(
        uploaded_image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

        tmp.write(uploaded_image.getbuffer())

        image_path = tmp.name

    try:
        with st.spinner("Understanding image..."):
            image_context = get_image_context(image_path)
    finally:
        try:
            os.remove(image_path)
        except Exception:
            pass

# -----------------------------
# Final Reasoning
# -----------------------------

if question:

    with st.spinner("Thinking..."):

        response = get_multimodal_response(
            question=question,
            image_context=image_context
        )

    st.subheader("Answer")

    st.write(response)

if st.sidebar.button("Clear Conversation"):
    clear_history()
    st.success("Conversation cleared.")