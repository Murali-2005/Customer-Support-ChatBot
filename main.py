from update_knowledge import update_vector_db
import streamlit as st
import os
from langchain_helper import create_vector_db, get_qa_chain

st.set_page_config(
    page_title="Customer Service Chatbot",
    page_icon="🤖",
)

st.title("🤖 Customer Service Chatbot")
VECTOR_DB_PATH = "faiss_index"

# Show Create button only if FAISS doesn't exist
if not os.path.exists(VECTOR_DB_PATH):

    if st.button("Create Knowledge Base"):

        with st.spinner("Creating Knowledge Base..."):

            create_vector_db()

        st.success("Knowledge Base Created!")

else:

    st.success("Knowledge Base Ready")

    if st.button("Update Knowledge Base"):

        with st.spinner("Updating Knowledge Base..."):

            message = update_vector_db()

        st.success(message)
  
question = st.text_input("Ask your question:")

if question:
    with st.spinner("Thinking..."):
        chain = get_qa_chain()

        response = chain.invoke({
            "input": question
        })

    st.subheader("Answer")

    st.write(response["answer"])