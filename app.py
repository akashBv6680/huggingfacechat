import streamlit as st
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# --- Securely get Hugging Face API token from Streamlit Secrets ---
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
    # Use a smaller, more memory-efficient model
    model_name = "distilgpt2" 
    
    with st.spinner(f"Loading model: {model_name}..."):
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=128,
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

llm_chain = load_llm_and_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the LLM chain to get a response
            response = llm_chain.invoke(prompt)
            # The output for distilgpt2 might be different, so let's clean it up
            clean_response = response['text'].split(prompt)[-1].strip()
            st.markdown(clean_response)

    st.session_state.messages.append({"role": "assistant", "content": clean_response})
