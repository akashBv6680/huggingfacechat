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
    # Use a small, conversation-optimized model
    # This is a T5 model, great for Q&A and dialogue, and still very small
    model_name = "google/flan-t5-small"
    
    with st.spinner(f"Loading conversational model: {model_name}..."):
        # We need a different pipeline task for T5 models
        text_gen_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=128,
            temperature=0.7
        )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    # The prompt should be simple for T5 models
    prompt_template = """Question: {question}
    Answer:"""
    
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
            response = llm_chain.invoke(prompt)
            # The T5 pipeline output is cleaner, so no complex post-processing is needed
            st.markdown(response['text'])

    st.session_state.messages.append({"role": "assistant", "content": response['text']})
