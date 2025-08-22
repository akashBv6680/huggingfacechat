import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
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
    model_name = "MaziyarPanahi/Llama-3-8B-Instruct-bnb-4bit" # A small, quantized Llama 3 model
    
    with st.spinner(f"Loading conversational model: {model_name}..."):
        # We need a different pipeline task for Llama 3 models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # Use 4-bit quantization to fit into memory
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto"
        )
        
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
        )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    # Use a prompt that is optimized for instruction-tuned models like Llama 3
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant. Answer the questions naturally.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
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
            # Post-process the response to remove the prompt template and extra text
            clean_response = response['text'].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            st.markdown(clean_response)

    st.session_state.messages.append({"role": "assistant", "content": clean_response})
