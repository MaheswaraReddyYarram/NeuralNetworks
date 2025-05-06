import streamlit as st
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(uploaded_file, openai_api_key, query_text):
    #load document
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        #split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.create_documents(documents)
        #choose llm
        llm = ChatOpenAI(model= 'gpt4-0',openai_api_key = openai_api_key)
        #select embeddings
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key = openai_api_key)
        # create vector store from embeddings
        database = Chroma.from_documents(texts, embeddings)
        #create retriever
        retriever = database.as_retriever()
        prompt = hub.pull('rlm/rag-prompt')
        rag_chain = (
            {'context': retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser
        )

        # create QA chain
        response = rag_chain.invoke(query_text)
        return response



if __name__ == '__main__':
    #upload file
    uploaded_file = st.file_uploader('Upload a file', type = ['txt', 'pdf'])
    # query text
    query_text = st.text_input('Enter your question', placeholder=' Please provide a short summary', disabled=not uploaded_file)
    #form input and query
    result = None
    with st.form('myForm', clear_on_submit=True, border=False):
        openai_api_key = st.text_input('Enter your OpenAI API Key', type="password", disabled=not (uploaded_file and query_text))
        submitted = st.form_submit_button('Submit', disabled = not (uploaded_file and query_text))
        if submitted and openai_api_key.startswith('sk-'):
            with st.spinner('Calculating...'):
                response = generate_response(uploaded_file, query_text, openai_api_key)
                result= response
    if result:
        st.info(result)
        
    
