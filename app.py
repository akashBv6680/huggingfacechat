import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# --- Securely get your Hugging Face API token from Streamlit Secrets ---
# The key 'HF_TOKEN' must match what you set in your Streamlit secrets.toml file.
try:
    hf_token = st.secrets["HF_TOKEN"]
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
except KeyError:
    st.error("Error: Hugging Face token not found in Streamlit Secrets. "
             "Please add it with the key 'HF_TOKEN'.")
    st.stop()

# --- Cache the model loading to prevent reloading on every interaction ---
@st.cache_resource
def load_llm_and_chain():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    
    # Use st.spinner to show a loading message
    with st.spinner(f"Loading model: {model_name}... This might take a moment."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        repetition_penalty=1.1,
        do_sample=True,
        temperature=0.7
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    prompt_template = """You are a helpful AI assistant. Answer the questions naturally.
    User: {question}
    Assistant:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"]
    )
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

# --- Main Streamlit App Interface ---
st.title("General AI Chatbot")
st.markdown("Ask me anything!")

# Load the LLMChain once and cache it
llm_chain = load_llm_and_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get a response from the LLMChain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm_chain.invoke(prompt)
            clean_response = response['text'].split("Assistant:")[1].strip()
            st.markdown(clean_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": clean_response})
