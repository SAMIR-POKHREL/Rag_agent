from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
import os

if "messages" not in st.session_state:
    st.session_state.messages=[]

if "agent" not in st.session_state:
    st.session_state.agent=None


if "vector_store" not in st.session_state:
    st.session_state.vector_store=False

if "document" not in st.session_state:
    st.session_state.document=False


def document_processor(document_path):

    # loading documents from a folder/path
    print("loading document \n")
    document_loader=PyPDFDirectoryLoader(document_path)
    loaded_docs=document_loader.load() # documents loaded
    print('documents loaded successfully.... \n')


    # splitting the documents
    print("splitting documents \n")
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splitted_docs=splitter.split_documents(loaded_docs)
    print("document spliitted successfully \n")


    # embeddings and storing in vector db

    @st.cache_resource
    def load_embedding_model():
        
        print("loading embedding model \n")
        embedder = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
            )
        print("embedding model loaded")
        return embedder
    
    embedder=load_embedding_model()
    vector_db=InMemoryVectorStore.from_documents(embedding=embedder,documents=splitted_docs)

    print("embeddings generated successfully and stored in vector db")

    # llm
    print("llm loaded")
    llm=ChatGroq(model='openai/gpt-oss-120b',temperature=0)

    print("tool loading \n")
    print('context generating........ \n')
    @tool
    def return_context(query:str):
        """"
                retrieve documents relevant to query  from the knowledge base.
        """

        context=""

        docs=vector_db.similarity_search(query=query,k=3)

        if not docs:
            return "no relevant info found"

        for doc in docs:
            context+=doc.page_content +'\n\n'

        print("context generated and ready for use \n")
        return context





    system_prompt=""""
        You are a helpful  assistant to retrieve information and provide answer .
    
        Always use the 'return_context' tool to retrieve relevant
        information from the documents before answering questions
        about its contents.
    
        Answer based on the retrieved context.
        If the answer is not available in the context, clearly say
        that the information is not available in the document.
    
        Do not invent information.
    
    
    """
    # agent creation
    memory=InMemorySaver()
    agent=create_agent(
        model=llm,
        tools=[return_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )


    st.session_state.agent=agent
    st.session_state.document=True
    print("agent loaded successfully ......")


#############################################################################
if not st.session_state.document:
    document_uploaded=st.file_uploader('upload documents',accept_multiple_files=True)

    if document_uploaded:
        path='./doc_fileeeee'
        os.makedirs(path,exist_ok=True)

        # saving the file uploaded by user in a folder path('./doc_fileeee)
        for files in document_uploaded:
            
            with open(os.path.join(path,files.name),'wb') as f:
                f.write(files.getvalue())
                print(f"file {files} uploaded in {path}--> as {os.path.join(path,files.name)} \n")

        print("file uploaded successfully \n")
        print(f"current directory: {os.getcwd()} \n")
        print(f"files list in {path} is = {os.listdir(path)} \n")


        # processing document
        print("Document processing has started \n")
        document_processor(path)
        st.rerun() # from here the whole code from start will run again 




if st.session_state.agent and st.session_state.document:

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])


    query=st.chat_input("ask anything related to documents you uploaded ?")

    if query:

        st.session_state.messages.append({
            'role':'user',
            'content':query
        })
        st.chat_message("user").markdown(query)

        with st.spinner("Thinking..........."):
            response=st.session_state.agent.invoke(
                {'messages':[{"role":"user",'content':query}]},
                {'configurable':{'thread_id':'agent1'}}
                )
            llm_response=response['messages'][-1].content
            st.chat_message("Miski").markdown(llm_response)

            st.session_state.messages.append({
                'role':"ai",
                'content':llm_response
            })


