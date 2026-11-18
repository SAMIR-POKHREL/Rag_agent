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

from langchain_chroma import Chroma

st.set_page_config(
    page_title="Miski AI",
    page_icon="🌟",
    layout='wide',
    initial_sidebar_state='expanded'
)

document_path='./your_documents'
os.makedirs(document_path,exist_ok=True)


if "messages" not in st.session_state:
    st.session_state.messages=[]

if "agent" not in st.session_state:
    st.session_state.agent=None


if "vector_store" not in st.session_state:
    st.session_state.vector_store=False

if "document" not in st.session_state:
    st.session_state.document=False

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files=[]


def document_processor(document_path):

    # loading documents from a folder/path
    with st.spinner("Loading documents..."):
        print("loading document \n")
        document_loader=PyPDFDirectoryLoader(document_path)
        loaded_docs=document_loader.load() # documents loaded

        if not loaded_docs:
            raise ValueError("no redable pdf documents found")
        print('documents loaded successfully.... \n')
        st.success(f"loaded {len(loaded_docs)} document pages")


    # splitting the documents
    with st.spinner("splitting documents into chunks"):
        print("splitting documents \n")

        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        splitted_docs=splitter.split_documents(loaded_docs)
        print("document spliitted successfully \n")

    st.success(f"created {len(splitted_docs)} chunks")


    # embeddings and storing in vector db

    @st.cache_resource
    def load_embedding_model():
        with st.spinner("generating embeddings..."):
            print("loading embedding model \n")
            embedder = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
                )
            print("embedding model loaded")
        return embedder

    
    embedder=load_embedding_model()
    vector_db=Chroma.from_documents(embedding=embedder,documents=splitted_docs,collection_name="miski_knowledge",persist_directory='./chroma_db')
    st.success("embeddings generated successfully")
    print("embeddings generated successfully and stored in vector db")

    # llm
    api_key=(
    st.secrets['GROQ_API_KEY']
    if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
    )
    if not api_key:
        raise ValueError("groq api key is missing")


    print("llm loaded")
    llm=ChatGroq(model='openai/gpt-oss-120b',temperature=0,api_key=api_key)

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
    st.session_state.vector_store=vector_db

    print("agent loaded successfully ......")
    return agent



with st.sidebar:
    st.title("Miski AI")
    st.caption("your intelligent document companion")

    st.divider()

    if st.button("New Conversation",use_container_width=True):
        st.session_state.messages=[]
        st.rerun()

    if st.button("Clear Knowledge Base",use_container_width=True):

        for file in os.listdir(document_path):
            file_path=os.path.join(document_path,file)

            if os.path.isfile(file_path) and file.lower().endswith('.pdf'):
                os.remove(file_path)

            st.session_state.messages=[]
            st.session_state.agent=None
            st.session_state.vector_store=None
            st.session_state.document=False
            st.session_state.uploaded_files=[]


        st.success("Knowledge base reset successfully")
        st.rerun()


    # document loading gareko
    st.divider()
    st.subheader("Knowledge Base")
    uploaded_files=st.file_uploader("upload pdf documents",type=['pdf'],accept_multiple_files=True)

    if st.button("Process Document",use_container_width=True):
        if not uploaded_files:
            st.warning("please upload files")
        else:
            try:
                with st.status("Processing...") as status:
                    for file in uploaded_files:
                        file_path=os.path.join(document_path,file.name)
                        with open(file_path,'wb') as f:
                            f.write(file.getvalue())

                st.write("documents uploaded successfully (your device)")
                document_processor(document_path=document_path)

                st.session_state.uploaded_files=[
                    f.name for f in uploaded_files
                ]
                status.update(label="knowledge base ready",state='complete')
                st.success("Miski is ready")
                st.rerun()



            except Exception as e:
                st.error(f"something went wrong {e}")





st.title("Miski personal assistant")
st.divider()

st.subheader('your documents')

pdf_files=[f for f in os.listdir(document_path) if f.lower().endswith('.pdf')]

if pdf_files:
    for pdf in pdf_files:
        with st.expander(f"{pdf}"):
            st.write(pdf)
else:
    st.info("please upload documents")

st.divider()

st.subheader("Settings")
# clear chat history button hai 
if st.button("Clear Chat History",use_container_width=True):
    st.session_state.messages=[]
    st.rerun()







for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


query=st.chat_input("ask Miski anything you like about your documents")

if query:
    if st.session_state.agent is None:
        st.warning("please upload and process pdf documents first .")
        st.stop()

    st.session_state.messages.append({
        'role':'user',
        'content':query
    })

    with st.chat_message("you"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Miski is thinking..."):
                res=st.session_state.agent.invoke({'messages':[{'role':'user','content':query}]},{'configurable':{'thread_id':'1'}})
                answer=res['messages'][-1].content
            st.write(answer)

            st.session_state.messages.append({'role':'assistant','content':answer})
        except Exception as e:
            st.error(f"something went wrong {e}")


