# see: https://www.youtube.com/watch?v=dXxQ0LR-3Hg

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface \
    import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import \
    ConversationalRetrievalChain
from html_templates import css, bot_template, user_template
from langchain.llms.huggingface_hub import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ''

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name='hkunlp/instructor-xl')

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conservation_chain(vectorestore):
    llm = HuggingFaceHub(
        repo_id='google/flan-t5-xxl',
        model_kwargs={
            'temerature': 0.5,
            'max_length': 512
        }
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True)

    conservation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorestore.as_retriever(),
        memory=memory
    )

    return conservation_chain


def handle_user_input(input):
    response = st.session_state.conservation(
        {'question': input}
    )

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace('{{MSG}}', message.content),
                unsafe_allow_html=True)

        else:
            st.write(
                bot_template.replace('{{MSG}}', message.content),
                unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(
        page_title='Chat with multiple PDFs',
        page_icon=':books:')

    st.write(css, unsafe_allow_html=True)

    if 'conservation' not in st.session_state:
        st.session_state.conservation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple PDFs :books:')
    user_questions = st.text_input('Ask a questions about your documents:')

    if user_questions:
        handle_user_input(user_questions)

    with st.sidebar:
        st.subheader('Your documents')

        pdf_docs = st.file_uploader(
            'Upload your PDFs here and click on "Process"',
            accept_multiple_files=True)

        if st.button('Process'):
            with st.spinner('Processing...'):

                # get pdfs text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                # get the embeddings
                vectorstore = get_vectorstore(text_chunks)

                # create conservation chain
                st.session_state.conservation = \
                    get_conservation_chain(vectorstore)


if __name__ == '__main__':
    main()
