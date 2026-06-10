import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["GOOGLE_API_KEY"] = ""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=os.environ["GOOGLE_API_KEY"]
)

question_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI tutor. Generate ONE question for {level}."
    ),
    (
        "human",
        "Topic: {topic}"
    )
])

question_chain = question_prompt | llm | StrOutputParser()

validation_chain = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert tutor.
        - Check if the user's answer is correct don't do anything
        - If answer incorrect then explain the mistake otherwise don't do anything
        - If answer incorrect then Provide the correct answer
        - If answer incorrect then Give a real-world example
          Return structured explanation.
          Return simple markdown only.
            Do not use tables.
            Do not use code blocks
        """
    ),
    (
        "human",
        """
        Question: {question}

        Answer: {answer}
        """
    )
])

final = validation_chain | llm | StrOutputParser()

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="AI Tutor", page_icon="📚")

st.title("📚 AI Tutor")
st.write("Generate questions and validate answers.")

topic = st.text_input("Topic")
level = st.selectbox(
    "Level",
    ["Beginner", "Intermediate", "Advanced"]
)

if "question" not in st.session_state:
    st.session_state.question = ""

if st.button("Generate Question"):
    if topic:
        question = question_chain.invoke({ 
            "topic": topic,
            "level": level
        })

        st.session_state.question = question

if st.session_state.question:
    st.subheader("Question")
    st.write(st.session_state.question)

    answer = st.text_area("Enter your answer",key="answer")

    if st.button("Submit Answer"):
        result = final.invoke({
            "question": st.session_state.question,
            "answer": answer
        })

        st.subheader("Evaluation")
        st.write(result)