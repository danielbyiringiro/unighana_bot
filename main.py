import streamlit as st
import langchain_helper as lch
import textwrap

st.title("UniGhana Chatbot")

about = st.sidebar.header("Ask any question about Ashesi University")
query = st.sidebar.text_area(label="Query",max_chars=200)  # Adjust max_chars as needed
button = st.sidebar.button("Ask")

if button:
    # Add a loading indicator
    with st.spinner("Analyzing your query..."):
        try:
            db = lch.create_vector_db_from_text()
            response, docs = lch.get_response_from_query(db, query)
            st.text(textwrap.fill(response, width=85))
        except Exception as e:
            st.error("An error occurred while processing your request.")
            st.write(str(e))