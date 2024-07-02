import streamlit as st
# from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from extract_text import extract_text_from_uploaded_pdf
from generate_embeddings import generate_embeddings
from store_vector import store_embeddings
from query_engine import search_index
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import base64
import os
import fitz  # PyMuPDF
import glob  # Import glob to read all text files
import shutil

# Ensure directories exist
directories = ["npy", "txt", "pdfs"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Function to empty directories and remove the vector store index
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

# Clean directories and index only during the initial stage
if "initialized" not in st.session_state:
    clean_directories_and_index()
    st.session_state["initialized"] = True

# Helper function to get base64 image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Convert images to base64
user_avatar_base64 = get_base64_image("./imgs/user.png")
robot_avatar_base64 = get_base64_image("./imgs/bot.png")

# Load vector store or initialize a new one
index = None
if os.path.exists("vector_store.index"):
    index = faiss.read_index("vector_store.index")
else:
    index = faiss.IndexFlatL2(384)  # Initialize with a dummy dimension which will be overwritten later

model = SentenceTransformer('all-MiniLM-L6-v2')

# Set page configuration
st.set_page_config(page_title="Intellicon", page_icon="🐧", layout="centered")

# Title and subtitle
st.title("Ask Intellicon!")
st.write("Your AI-powered assistant for document insights")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "bot", "content": "Hi there! <br> This is Intellicon 🤖 <br> Ready to get started? <br> Upload your PDFs for detailed insights."})

# Load T5 model and tokenizer only once and store in session state
# if "t5_model" not in st.session_state:
#     st.session_state["t5_model"] = T5ForConditionalGeneration.from_pretrained('t5-small')
#     st.session_state["t5_tokenizer"] = T5Tokenizer.from_pretrained('t5-small', legacy=False)

# # Load GPT-2 model and tokenizer only once and store in session state
# if "gpt_model" not in st.session_state:
#     st.session_state["gpt_model"] = GPT2LMHeadModel.from_pretrained('gpt2')
#     st.session_state["gpt_tokenizer"] = GPT2Tokenizer.from_pretrained('gpt2', legacy=False)

# def generate_t5_insights(prompt, relevant_texts):
#     combined_prompt = prompt + "\n\n" + "\n\n".join(relevant_texts)
#     tokenizer = st.session_state["t5_tokenizer"]
#     model = st.session_state["t5_model"]
    
#     inputs = tokenizer.encode("summarize: " + combined_prompt, return_tensors='pt', max_length=512, truncation=True)
    
#     outputs = model.generate(inputs, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2)
    
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     response_lines = response.split('. ')
#     unique_lines = []
#     for line in response_lines:
#         if line and line not in unique_lines:
#             unique_lines.append(line)
    
#     structured_response = '. '.join(unique_lines) + '.'
    
#     return structured_response

# def generate_gpt_insights(prompt, relevant_texts):
#     combined_prompt = prompt + "\n\n" + "\n\n".join(relevant_texts)
#     tokenizer = st.session_state["gpt_tokenizer"]
#     model = st.session_state["gpt_model"]
    
#     inputs = tokenizer.encode(combined_prompt, return_tensors='pt')
    
#     outputs = model.generate(inputs, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2)
    
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     response = response.replace(prompt, '').strip()
    
#     response_lines = response.split('. ')
#     unique_lines = []
#     for line in response_lines:
#         if line and line not in unique_lines and not line.endswith(','):
#             unique_lines.append(line)
    
#     structured_response = '. '.join(unique_lines) + '.'
    
#     return structured_response

import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_TOKEN = "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"  # Replace with your Hugging Face API token

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_gpt_insights(prompt, relevant_texts):
    combined_prompt = prompt + "\n\n" + "\n\n".join(relevant_texts)
    payload = {
        "inputs": "summarize: " + combined_prompt,
        "parameters": {
            "max_length": 300,
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 2
        }
    }
    response = query(payload)
    response_text = response[0]['generated_text']
    response_lines = response_text.split('. ')
    unique_lines = []
    for line in response_lines:
        if line and line not in unique_lines:
            unique_lines.append(line)
    
    structured_response = '. '.join(unique_lines) + '.'
    
    return structured_response

# Function to extract text from PDF
def extract_text_from_uploaded_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Ensure directories exist
os.makedirs("txt", exist_ok=True)
os.makedirs("npy", exist_ok=True)
os.makedirs("pdfs", exist_ok=True)

# Move any existing PDFs to the pdfs directory
for pdf_file in glob.glob("*.pdf"):
    shutil.move(pdf_file, os.path.join("pdfs", os.path.basename(pdf_file)))

# Upload PDFs
uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and "uploaded_files_processed" not in st.session_state:
    for uploaded_file in uploaded_files:
        # Save the uploaded PDF to the pdfs directory
        pdf_path = os.path.join("pdfs", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text from PDF
        text = extract_text_from_uploaded_pdf(uploaded_file)
        with open(f"txt/{uploaded_file.name}.txt", "w", encoding="utf-8") as text_file:
            text_file.write(text)
        
        # Generate embeddings
        embeddings = generate_embeddings(text)
        np.save(f"txt/{uploaded_file.name}_embeddings.npy", embeddings)
        
        # Check dimension of embeddings and reinitialize the index if needed
        if index.ntotal == 0:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)  # Reinitialize with correct dimension
        
        # Add embeddings to vector store
        store_embeddings(embeddings, index, uploaded_file.name)
        
    # Save the updated vector store
    faiss.write_index(index, "vector_store.index")

    # Inform user of successful upload
    st.session_state["messages"].append({"role": "bot", "content": "PDFs are uploaded and processed successfully... <br> You can now ask questions based on the uploaded documents."})
    
    # Set flag to indicate files have been processed
    st.session_state["uploaded_files_processed"] = True
    
    # Rerun the script to update the state
    st.rerun()

else:
    # Chat container styling
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

    # Add a selectbox for model selection
    model_choice = st.selectbox("Choose the model to generate insights:", ["Flan-T5", "GPT-2"])

    # Load and combine text from all files in the txt folder
    all_texts = []
    for file_path in glob.glob("txt/*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            all_texts.extend(file.read().split('\n'))

    # Display chat history
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

    # Input box at the bottom
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="input_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])

        # Add a text input in the first column
        with col1:
            query = st.text_input("Enter your query", key="input", label_visibility="collapsed")

        # Add a submit button in the second column
        with col2:
            submit_button = st.form_submit_button(label="Send")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit_button and query:
        with st.spinner("Generating insights..."):
            # Search the index for relevant text segments
            results = search_index(query, index, model)
            
            # Load the original text segments
            if isinstance(results[0], np.ndarray):  # Ensure results[0] is iterable
                relevant_texts = [all_texts[i] for i in results[0] if i < len(all_texts)]
            else:
                relevant_texts = [all_texts[results[0]]]

            # Generate insights based on the relevant texts
            if model_choice == "GPT-2":
                insights = generate_gpt_insights(query, relevant_texts)
            elif model_choice == "Flan-T5":
                insights = generate_gpt_insights(query, relevant_texts)
            
            # Add user query and response to chat history
            st.session_state["messages"].append({"role": "user", "content": query})
            st.session_state["messages"].append({"role": "bot", "content": insights})
            
            # Rerun the script to display the updated chat
            st.experimental_rerun()