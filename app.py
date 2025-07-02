import streamlit as st
from src.textSummarizer.components.model_init import load_model
from src.textSummarizer.components.infer import summarize

st.title("📝 Text Summarizer")

text = st.text_area("Enter dialogue or paragraph")
if st.button("Summarize"):
    tokenizer, model = load_model()
    result = summarize(text, model, tokenizer)
    st.success(result)
