import streamlit as st
import requests

st.title("Biomedical Literature QA")
st.write("Ask questions about CFTR-related medical literature.")

# Input box for query
query = st.text_input("Enter your question:")
if st.button("Submit"):
    if query:
        try:
            response = requests.post("http://127.0.0.1:8000/ask", json={"question": query})
            if response.status_code == 200:
                st.success(f"Answer: {response.json()['answer']}")
            else:
                st.error(f"Error: {response.status_code} - {response.json()['detail']}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend. Make sure it's running.")
    else:
        st.warning("Please enter a question.")
