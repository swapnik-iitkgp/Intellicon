import streamlit as st
import os
import glob
import shutil
import base64
import fitz 
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

directories = ["npy", "txt", "pdfs"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

def clean_directories_and_index():
    for directory in directories:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    if os.path.exists("vector_store.index"):
        os.remove("vector_store.index")

def move_pdfs_to_folder():
    pdf_files = glob.glob("*.pdf")
    for pdf_file in pdf_files:
        shutil.move(pdf_file, os.path.join("pdfs", pdf_file))

if "initialized" not in st.session_state:
    clean_directories_and_index()
    st.session_state["initialized"] = True

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

user_avatar_base64 = get_base64_image("user.png")
robot_avatar_base64 = get_base64_image("bot.png")

st.set_page_config(page_title="Intellicon", page_icon="🐧", layout="centered")

st.title("Ask Intellicon!")
st.write("Your AI-powered assistant for document insights")

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "bot", "content": "Hi there! <br> This is Intellicon 🤖 <br> Ready to get started? <br> Upload your PDFs for detailed insights."})

uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and "uploaded_files_processed" not in st.session_state:
    texts = []
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join("pdfs", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyMuPDFLoader(pdf_path)
        doc_pages = loader.load()
        text = "\n".join([page.page_content for page in doc_pages])
        texts.append(text)
    
    embeddings = embeddings_model.embed_documents(texts)
    vectorstore = FAISS.from_texts(texts, embeddings_model)

    vectorstore.save_local("vector_store")

    st.session_state["messages"].append({"role": "bot", "content": "PDFs are uploaded and processed successfully... <br> You can now ask questions based on the uploaded documents."})
    
    st.session_state["uploaded_files_processed"] = True
    
    st.rerun()

else:
    st.markdown(
        """
        <style>
        .chat-container {
            background-color: #2D2D2D;
            padding: 10px;
            border-radius: 10px;
            max-height: 70vh;
            overflow-y: auto;
            margin-bottom: 40px; /* To make space for the input box */
        }
        .user-message, .bot-message {
            display: flex;
            align-items: center;
            margin-bottom: 20px; /* Increased space between messages */
        }
        .user-message {
            justify-content: flex-end;
        }
        .bot-message {
            justify-content: flex-start;
        }
        .message-content {
            padding: 10px;
            border-radius: 20px;
            max-width: 70%;
        }
        .user-message .message-content {
            background-color: #FF4B4B;
            color: white;
        }
        .bot-message .message-content {
            background-color: #FFD700;
            color: black.
        }
        .avatar {
            width: 45px;  /* Size for zoom effect */
            height: 45px; /* Size for zoom effect */
            border-radius: 30%;
            margin: 0 10px;
        }
        .input-container {
            position: fixed;
            bottom: 0;
            width: 100%.
        }
        .input-container form {
            bottom: 0.
            display: flex.
            align-items: center.
        }
        .input-container input {
            flex: 1.
            padding: 10px.
            border-radius: 5px 0 0 5px.
            border: none.
        }
        .input-container button {
            padding: 10px 20px.
            background-color: #FF4B4B.
            color: white.
            border: none.
            border-radius: 0 5px 5px 0.
            white-space: nowrap.
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    model_choice = st.selectbox("Choose the model to generate insights:", ["Flan-T5", "GPT-2"])

    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">'
                f'<div class="message-content">{message["content"]}</div>'
                f'<img src="data:image/png;base64,{user_avatar_base64}" class="avatar">'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-message">'
                f'<img src="data:image/png;base64,{robot_avatar_base64}" class="avatar">'
                f'<div class="message-content">{message["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="input_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])

        with col1:
            query = st.text_input("Enter your query", key="input", label_visibility="collapsed")

        with col2:
            submit_button = st.form_submit_button(label="Send")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit_button and query:
        st.session_state["messages"].append({"role": "user", "content": query})
        
        with st.spinner("Generating insights..."):
            vectorstore = FAISS.load_local("vector_store", embeddings_model, allow_dangerous_deserialization=True)
            
            retriever = vectorstore.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in retrieved_docs])
            
            if model_choice == "GPT-2":
                inputs = gpt2_tokenizer.encode(query + context, return_tensors="pt", max_length=512, truncation=True)
                outputs = gpt2_model.generate(inputs, max_length=2048, num_return_sequences=1)
                insights = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                input_text = "question: {} context: {}".format(query, context)
                inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=2048, truncation=True)
                outputs = t5_model.generate(inputs, max_length=2048, num_return_sequences=1)
                insights = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.session_state["messages"].append({"role": "bot", "content": insights})
        
        st.rerun()