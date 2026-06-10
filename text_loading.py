from langchain_community.document_loaders import TextLoader
txt_loader = TextLoader(file_path='content.txt')
doc = txt_loader.load()  #(method) def load() -> list[Document]

import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
nlp = spacy.load("en_core_web_sm")
text = '''
Artificial Intelligence (AI) is one of the most transformative technologies of the modern era. It refers to the simulation of human intelligence in machines that are programmed to think, learn, and make decisions. AI systems are designed to perform tasks such as problem-solving, speech recognition, decision-making, and language translation—tasks that traditionally required human intelligence.

AI is not a single technology but a broad field that includes several subfields such as machine learning, deep learning, natural language processing, computer vision, and robotics. These areas work together to create intelligent systems capable of handling complex tasks.
'''

doc = nlp(text)
processed_text = " ".join([ token.lemma_ for token in doc if not token.is_punct ])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(processed_text)

from langchain_huggingface import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #Teh default size of all-MiniLM-L6-v2 is 384 dimensions

embeddings = embed_model.embed_documents(chunks)

result5 = splitter.create_documents([processed_text])    

for doc, emb in zip(result5, embeddings):
    doc.metadata["hf_embedding"] = emb


from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages  import HumanMessage, SystemMessage

vectorstore = FAISS.from_documents(result5, embed_model)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    api_key="AIzaSyDO4GsJio6Flrs5gxOezoSfuqvGI_EKpFs"
)
def rag_query(query: str, k: int = 4):
    # build context and prompt, then call the model using .generate()
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer only using the context. If the answer is not contained in the context, say you do not know."
    )

    result = chat_model.generate([[SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)]])

    # extract text from possible return shapes
    try:
        rag_answer = result.generations[0][0].text
    except Exception:
        rag_answer = result.generations[0][0].message.content

    return rag_answer

print("Number of documents in result:", len(result5))
try:
    print("Number of vectors in FAISS index:", vectorstore.index.ntotal)
except Exception as e:
    print("Could not read FAISS vector count:", e)

query = "What tasks can AI perform?"
print("\nRAG answer:",rag_query(query))




















