import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# FastAPI backend URLs
UPLOAD_FILE_URL = os.getenv("UPLOAD_FILE_URL")
CHAT_API_URL = os.getenv("CHAT_API_URL")

# Streamlit Interface
def pdf_chatbot_interface():
    st.title("Document Chatbot")
    
    # Initialize session state variables if they are not already set
    if "file_uploaded" not in st.session_state:
        st.session_state["file_uploaded"] = False    # Tracks if a FILE has been uploaded
        st.session_state["uploaded_file_name"] = None   # Stores the uploaded file name
        st.session_state["chat_history"] = []   # Stores the chat history
    
    # File uploader widget to upload a file
    uploaded_file = st.file_uploader("Upload a FILE", type=["pdf", "docx", "txt"])
    
    # Process the uploaded file if it hasn't been uploaded before
    if uploaded_file is not None and not st.session_state["file_uploaded"]:
        with st.spinner("Uploading FILE..."): # Display a loading spinner
            # Determine the MIME type based on file extension for the API request
            filename = uploaded_file.name.lower()
            
            if filename.endswith(".pdf"):
                mime_type = "application/pdf"
            elif filename.endswith(".docx"):
                mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif filename.endswith(".txt"):
                mime_type = "text/plain"
                
            files = {"file": (uploaded_file.name, uploaded_file, mime_type)}
            response = requests.post(UPLOAD_FILE_URL, files=files)   # Send a request to upload the FILE
            
            # Check the API response
            if response.status_code == 200:
                st.success("FILE uploaded successfully!") # Show success message
                st.session_state["file_uploaded"] = True # Update session state to indicate successful upload
                st.session_state["uploaded_file_name"] = uploaded_file.name # Store the uploaded file name
            else:
                st.error("Failed to upload FILE. Please try again.") # Show error message
                st.session_state["file_uploaded"] = False    # Reset upload state

    # If a FILE has been successfully uploaded, enable the chat interface
    if st.session_state.get("file_uploaded"):  
        st.subheader("Chat with FILE")   # Display a subheading

        # Ensure chat history is initialized in session state
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Display chat history
        for q, a, sources in st.session_state["chat_history"]:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)
                if sources:
                    st.markdown(f"**Sources:** {', '.join(sources)}")

        # User input for the query
        if user_input := st.chat_input("Enter your question:"):
            st.session_state["chat_history"].append((user_input, "", []))
            with st.chat_message("user"):
                st.markdown(user_input)

            # Send properly formatted chat history
            formatted_chat_history = [(q, a) for q, a, _ in st.session_state["chat_history"]]

            with st.spinner("Thinking..."):
                response = requests.post(CHAT_API_URL, json={
                    "question": user_input,
                    "chat_history": formatted_chat_history  # Send correctly formatted history
                })
            if response.status_code == 200:
                result = response.json()
                assistant_response = result["answer"]
                sources = result.get("sources", [])  

                st.session_state["chat_history"][-1] = (user_input, assistant_response, sources)

                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
                    if sources:
                        st.markdown(f"**Sources:** {', '.join(sources)}")


# Run the interface
if __name__ == "__main__":
    pdf_chatbot_interface()
