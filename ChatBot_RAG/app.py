import streamlit as st
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
import base64

from htmlTemplates import css, expander_css, user_template, bot_template


def process_file(uploaded_file):
    model_name = 'thenlper/gte-small'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    #select embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name = model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)

    #create vector database
    print(f'type of uploaded_file is {type(uploaded_file)}')
    print(f'uploaded_file content is {uploaded_file[0]}')
    database = Chroma.from_documents(uploaded_file, embeddings)


    # create RAG chain
    rag_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
                                                      retriever=database.as_retriever(search_kwargs = {"k":2}),
                                                      return_source_documents = True)
    return rag_chain

def handle_userinput(query):
    response = st.session_state.conversation({"question": query, 'chat_history': st.session_state.chat_history}, return_only_outputs = True)
    st.session_state.chat_history += [(query, response)]

    st.session_state.N = list(response['source_documents'][0])[1][1]['page']
    for i, message in enumerate(st.session_state.chat_history):
        print(f'message is {message}')
        st.session_state.expander1.write(user_template.replace("{{MSG}}", message[0]), unsafe_allow_html=True)
        st.session_state.expander1.write(bot_template.replace("{{MSG}}", message[1].get('answer')), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(layout="wide",page_title="Interactive PDF Reader", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "N" not in st.session_state:
        st.session_state.N = 0


    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    st.session_state.col1.header("Interactive Reader :books:")
    user_question = st.session_state.col1.text_input("Ask a question on the contents of the uploaded PDF:")
    st.session_state.expander1 = st.session_state.col1.expander('Your Chat', expanded=True)
    st.session_state.col1.markdown(expander_css, unsafe_allow_html=True)


    # load and process PDF files
    st.session_state.col1.subheader("Your Documents")
    st.session_state.pdf_doc = st.session_state.col1.file_uploader("Upload your PDF file here  and click 'Process'", type= 'pdf')

    if st.session_state.col1.button("Process", key= 'a'):
        with st.spinner("Processing"):
            if st.session_state.pdf_doc is not None:
                with NamedTemporaryFile(suffix='pdf') as temp:
                    temp.write(st.session_state.pdf_doc.getvalue())
                    temp.seek(0)
                    loader = PyPDFLoader(temp.name)
                    print(f'loader name is {loader}')
                    pdf = loader.load()
                    #print(f'pdf page_content is {pdf}')
                    st.session_state.conversation = process_file(pdf)
                    st.session_state.col1.markdown("Done processing. You may now ask a question")



    if user_question:
        handle_userinput(user_question)
        with NamedTemporaryFile(suffix='pdf') as temp:
            temp.write(st.session_state.pdf_doc.getvalue())
            temp.seek(0)
            reader = PdfReader(temp.name)

            pdf_writer = PdfWriter()
            start = max(st.session_state.N-2,0)
            end = min(st.session_state.N+2, len(reader.pages)-1)
            while start <= end:
                pdf_writer.add_page(reader.pages[start])
                start+=1
            with NamedTemporaryFile(suffix='pdf') as temp2:
                pdf_writer.write(temp2.name)
                with open(temp2.name, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#page={3}"\
                                        width="100%" height="900" type="application/pdf frameborder="0"></iframe>'

                    st.session_state.col2.markdown(pdf_display, unsafe_allow_html=True)



if __name__ == '__main__':
    main()



