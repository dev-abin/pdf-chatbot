import streamlit as st
import requests

# FastAPI backend URLs (update these with your actual backend URLs)
UPLOAD_PDF_URL = "http://localhost:8000/upload-pdf/"
CHAT_API_URL = "http://localhost:8000/chat/"


# Streamlit Interface
def pdf_chatbot_interface():
    st.title("PDF Chatbot")
    
    # Initialize session state variables if they are not already set
    if "pdf_uploaded" not in st.session_state:
        st.session_state["pdf_uploaded"] = False    # Tracks if a PDF has been uploaded
        st.session_state["uploaded_file_name"] = None   # Stores the uploaded file name
        st.session_state["chat_history"] = []   # Stores the chat history
    
    # File uploader widget to upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    # Process the uploaded file if it hasn't been uploaded before
    if uploaded_file is not None and not st.session_state["pdf_uploaded"]:
        with st.spinner("Uploading PDF..."): # Display a loading spinner
            # Prepare the file for sending via API request
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(UPLOAD_PDF_URL, files=files)   # Send a request to upload the PDF
            
            # Check the API response
            if response.status_code == 200:
                st.success("PDF uploaded successfully!") # Show success message
                st.session_state["pdf_uploaded"] = True # Update session state to indicate successful upload
                st.session_state["uploaded_file_name"] = uploaded_file.name # Store the uploaded file name
            else:
                st.error("Failed to upload PDF. Please try again.") # Show error message
                st.session_state["pdf_uploaded"] = False    # Reset upload state

    # If a PDF has been successfully uploaded, enable the chat interface
    if st.session_state.get("pdf_uploaded"):  
        st.subheader("Chat with PDF")   # Display a subheading

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

            with st.spinner("Fetching answer..."):
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
