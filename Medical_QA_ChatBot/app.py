import streamlit as st

from medical_langchain_helper import get_medical_response, clear_history

st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="🩺")

st.title("🩺 Medical Q&A Chatbot")
st.caption(
    "Ask a medical question. This is for informational purposes only — "
    "not a substitute for professional medical advice."
)

# Messages shown on screen. Kept separately from the helper module's own
# chat_history (which it uses internally for retrieval context) so the
# UI doesn't need to know about that implementation detail.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        clear_history()
        st.rerun()

# Replay past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
question = st.chat_input("Ask a medical question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_medical_response(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
