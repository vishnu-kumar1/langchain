import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm1 = ChatGroq(
    model = 'llama-3.3-70b-versatile',
    temperature=0,
    api_key=''
)
prompt = ChatPromptTemplate.from_messages([
    ('system','You are a helpfull assistance your work is to '
    '1.check the user given code is correct or not'
    '2.if not correct provide that line with comment whats the error'
    '3.Do not provide any explanation just code and use comment to point error'),
    ('human','this is the code {query}')
])
chain1 = prompt | llm1 | StrOutputParser()

prompt2 = ChatPromptTemplate.from_messages([
    ('system','You are a helpfull assistance your work is '
    '1.optimized the given user code Output must be code only'
    '2.Before optimizing the code provide error part with comment'),
    ('human','this is the code {query}')
])
chain2 = prompt2 | llm1 | StrOutputParser()

full_chain = chain1 | chain2 | StrOutputParser()



st.title("Code Optimizer")

code = st.text_area("Enter your code here:", height=200,width=700)

if st.button("Optimize"):
    if code:
        # Initialize the Grok model
        result = full_chain.invoke({'query': code})

        # Optimize the code
        optimized_code = result

        st.subheader("Optimized Code:")
        st.code(optimized_code, language="python")
    else:
        st.warning("Please enter some code to optimize.")