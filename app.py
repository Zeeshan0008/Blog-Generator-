import streamlit as st
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM  # Direct import from ctransformers


def getLlamaResponse(input_text, no_words):
    # Load the model using ctransformers
    llm = AutoModelForCausalLM.from_pretrained('/home/zeeshan/DsProjects/Blog_Creation/llama-2-7b-chat.Q8_0.gguf', 
                                               model_type='llama')

    # Define the prompt template
    template = """
        WRITE A BLOG ON THE TOPIC {input_text} within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["input_text", "no_words"], template=template)

    # Generate the response using the formatted prompt
    response = llm(prompt.format(input_text=input_text, no_words=no_words))
    print(response)
    return response


# Streamlit app layout
st.set_page_config(page_title="Blog Generation", layout='centered', initial_sidebar_state='collapsed')

st.header("GENERATE BLOGS")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('No of words')

submit = st.button("Generate")

if submit:
    st.write(getLlamaResponse(input_text, no_words))
