import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.set_page_config(
    page_title="Customer Service Chatbot",
    page_icon="🤖",
)

st.title("🤖 Customer Service Chatbot")

if st.button("Create Knowledge Base"):
    with st.spinner("Creating Vector Database..."):
        create_vector_db()
    st.success("Knowledge Base Created Successfully!")

question = st.text_input("Ask your question:")

if question:
    with st.spinner("Thinking..."):
        chain = get_qa_chain()

        response = chain.invoke({
            "input": question
        })

    st.subheader("Answer")

    st.write(response["answer"])