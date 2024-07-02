import streamlit as st
from extract_text import extract_text_from_uploaded_pdf
from generate_embeddings import generate_embeddings
from store_vector import store_embeddings
from query_engine import search_index
from generate_t5_insights import generate_t5_insights
from generate_gpt_insights import generate_gpt_insights
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import base64
import os
import fitz 
import glob 
import shutil

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

user_avatar_base64 = get_base64_image("./imgs/user.png")
robot_avatar_base64 = get_base64_image("./imgs/bot.png")

index = None
if os.path.exists("vector_store.index"):
    index = faiss.read_index("vector_store.index")
else:
    index = faiss.IndexFlatL2(384)

model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Intellicon", page_icon="🐧", layout="centered")

st.title("Ask Intellicon!")
st.write("Your AI-powered assistant for document insights")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "bot", "content": "Hi there! <br> This is Intellicon 🤖 <br> Ready to get started? <br> Upload your PDFs for detailed insights."})

def extract_text_from_uploaded_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if not os.path.exists("txt"):
    os.makedirs("txt")

if not os.path.exists("npy"):
    os.makedirs("npy")

if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and "uploaded_files_processed" not in st.session_state:
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join("pdfs", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with open(pdf_path, "rb") as f:
            text = extract_text_from_uploaded_pdf(f)
        with open(f"txt/{uploaded_file.name}.txt", "w", encoding="utf-8") as text_file:
            text_file.write(text)
        
        embeddings = generate_embeddings(text)
        np.save(f"txt/{uploaded_file.name}_embeddings.npy", embeddings)
        
        if index.ntotal == 0:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension) 
        
        store_embeddings(embeddings, index, uploaded_file.name)
        
    move_pdfs_to_folder()

    faiss.write_index(index, "vector_store.index")

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
            color: black;
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
            width: 100%;
        }
        .input-container form {
            bottom: 0;
            display: flex;
            align-items: center;
        }
        .input-container input {
            flex: 1;
            padding: 10px;
            border-radius: 5px 0 0 5px;
            border: none;
        }
        .input-container button {
            padding: 10px 20px;
            background-color: #FF4B4B;
            color: white;
            border: none;
            border-radius: 0 5px 5px 0;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    model_choice = st.selectbox("Choose the model to generate insights:", ["GPT-2", "Flan-T5"])

    all_texts = []
    for file_path in glob.glob("txt/*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            all_texts.extend(file.read().split('\n'))

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
            results = search_index(query, index, model)
            
            if isinstance(results[0], np.ndarray): 
                relevant_texts = [all_texts[i] for i in results[0] if i < len(all_texts)]
            else:
                relevant_texts = [all_texts[results[0]]]

            if model_choice == "Flan-T5":
                insights = generate_t5_insights(query, relevant_texts)
            else:
                insights = generate_gpt_insights(query, relevant_texts)
            
            st.session_state["messages"].append({"role": "bot", "content": insights})
        
        st.rerun()