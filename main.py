import os
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
import streamlit as st
from transformers import BitsAndBytesConfig
import torch

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Define a function to convert messages to a prompt
def messages_to_prompt(messages):
    prompt = "\n".join([f"\n{message.content}\n" for message in messages if message.role in ['system', 'user', 'assistant']])
    prompt = "\n\n" + prompt if not prompt.startswith("\n") else prompt
    return prompt + "\n"

# Initialize HuggingFaceLLM
llm = HuggingFaceLLM(
    model_name="stabilityai/stablelm-zephyr-3b",
    tokenizer_name="stabilityai/stablelm-zephyr-3b",
    query_wrapper_prompt=PromptTemplate("\n\n\n{query_str}\n\n"),
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.3},
    device_map="auto",
)

# Service Context for the LLM Model
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

# Function to check if index files exist
def index_files_exist():
    return os.path.exists("index")

# Function to create files and index automatically if they don't exist
def create_files_and_index():
    # Check if the "files" folder exists
    if not os.path.exists("files"):
        os.makedirs("files")

    # Check if the index already exists
    if os.path.exists("index"):
        return None

    # If index doesn't exist, create it
    documents = SimpleDirectoryReader("files").load_data()
    vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    vector_index.storage_context.persist("index")

# Function to upload files through the Streamlit app and save them in the "files" directory
def upload_files():
    st.sidebar.title(":blue[QuestifyAI: Your Conversational AI Partner]")
    st.sidebar.header("Upload Files")
    uploaded_files = st.sidebar.file_uploader("", type=["txt", "pdf", "docx", "pptx", "ppt", "jpg", "png", "jpeg", "ipynb"], accept_multiple_files=True)
    if uploaded_files:
        # Remove existing files in the "files" folder
        for file_name in os.listdir("files"):
            file_path = os.path.join("files", file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # Save uploaded files to the "files" folder
        for file in uploaded_files:
            with open(os.path.join("files", file.name), "wb") as f:
                f.write(file.getvalue())
        st.sidebar.success("Files uploaded successfully.")
        # Trigger the creation of files and index
        create_files_and_index()

# Function to ask questions
def ask_question(query):
    if index_files_exist():
        prompt = f'given the context: "answer_from_rag", reply to the user\'s question: "{query}"'
        response = query_engine.query(prompt)
        return response.response
    else:
        return "Please upload files to get answers from them."

# Function to prompt the user for a question using the Stable LLM model
def prompt_question():
    user_question = st.text_input("Ask your question")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
            query_result = ask_question(user_question)
        with st.chat_message("assistant"):
            st.markdown(query_result)

# Call the function to upload files
upload_files()

# Load index and set up query engine
storage_context = StorageContext.from_defaults(persist_dir="index")
index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
query_engine = index.as_query_engine(similarity_top_k=5)

# Call the function to prompt the user for a question
prompt_question()
