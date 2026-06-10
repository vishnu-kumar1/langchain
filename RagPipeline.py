class RAG:
    def text_loader(self, file_path):
        from langchain_community.document_loaders import TextLoader
        txt_loader = TextLoader(file_path=file_path)
        doc = txt_loader.load()
        print("Document loaded successfully.")
        print("Enter the Yes if you want to normalize the text:")
        if input().lower() == "yes":
            doc[0].page_content = self.normalize_text(doc[0].page_content)
            print("Text normalized successfully.")
        else:
            print("Skipping text normalization.")
            self.split_text(doc[0].page_content)
    
    
    def normalize_text(self, text):
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        processed_text = " ".join([ token.lemma_ for token in doc if not token.is_punct ])
        self.split_text(processed_text)
    
        
    def split_text(self, text, chunk_size=100, chunk_overlap=20):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents([text])
        self.embed_documents(chunks)
        
    
    def embed_documents(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings
        embed_model = HuggingFaceEmbeddings(model_name=model_name)
        embeddings = embed_model.embed_documents(chunks)
        self.create_documents_with_embeddings(chunks, embeddings)
        
    
    def create_documents_with_embeddings(self, doc, embeddings):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        result5 = RecursiveCharacterTextSplitter().create_documents([doc])  
        for docs, emb in zip(result5, embeddings):
            docs.metadata["hf_embedding"] = emb
        self.create_vectorstore(result5, self.embed_model())

    
    def embed_model(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings
        embed_model = HuggingFaceEmbeddings(model_name=model_name)
        return embed_model
    
    def create_vectorstore(self, documents, embed_model):
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.from_documents(documents, embed_model)
        rag_answer = self.rag_query("What is the main topic of the document?", vectorstore, self.create_chat_model())
        print("RAG Answer:", rag_answer)        
    
    def create_chat_model(self, model="gemini-2.5-flash", temperature=0, api_key=""):
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat_model = ChatGoogleGenerativeAI(
            model=model, 
            temperature=temperature, 
            api_key=api_key
        )
        return chat_model
    
    def rag_query(self, query, vectorstore, chat_model, k=4):
        from langchain_core.messages  import HumanMessage, SystemMessage
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer only using the context. If the answer is not contained in the context, say you do not know."
        )

        result = chat_model.generate([[SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)]])

        try:
            rag_answer = result.generations[0][0].text
        except Exception:
            rag_answer = result.generations[0][0].message.content

        return rag_answer
    
    def run_pipeline(self):
        print("Enter your text file path :")
        file_path = input()
        doc = self.text_loader(file_path)
        print(doc)

obj = RAG()
obj.run_pipeline()