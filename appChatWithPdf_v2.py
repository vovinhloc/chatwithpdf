import streamlit as st
import pickle,os
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import random
import time,json 
# import openai
# import sys
# # Thêm đường dẫn thư mục chứa file MyClass.py vào sys.path
# sys.path.append('./pyclass/')
from MyPyClass.FileProcess import FileProcess
from MyPyClass.FAnswer import FAnswer


os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['HUGGING_FACE_HUB_API_KEY']=st.secrets["HUGGING_FACE_HUB_API_KEY"]


fileProcess = FileProcess()
fAnswer=FAnswer()
st.set_page_config(
    page_title="VietAI - Chat with pdf",
    page_icon=":robot_face:",
)
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "filename" not in st.session_state:
    st.session_state.filename = "chua co file"

if "docsSimilarity" not in st.session_state:
    st.session_state.docsSimilarity = []
    





# load_dotenv()
def main():
    with st.sidebar:
        st.title('LLM ChatPDF App')        
        pdf =st.file_uploader("Upload pdf file",type='pdf')
        
        
        optionAsk = st.selectbox(
                "Select a mode chat :",
                ("normal"
                 ,"retrieve_new_page=False",
                  "retrieve_diff_page=True",
                    "retrieve_choose_Page",
                    "retrieve_external_knowledge=True"),
                index=None,
                placeholder="Select a mode chat...",
                )

        
        if optionAsk == "retrieve_choose_Page":            
            chatWithPages = st.text_input('Chat with pages :', '',placeholder="ex: 1,2")
        # else:            
        #     st.empty()
        
        add_vertical_space(5)
        

        

    st.header("Chat with PDF")
    if pdf is not None:
        
        fileProcess.saveFile(pdf)

        pages = fileProcess.loadPdfFileToPagesThroughLangchain(pdf)    

        chunks=fileProcess.createChunks(pages)

        faissName=pdf.name[:-4]
        st.session_state.filename = pdf.name
        chunks_faiss_embedding=fileProcess.loadHFVectorEmbeddings(faissName,chunks)

        tab1, tab2, tab3 = st.tabs(["Chat", "pdfPages", "Logs"])

        with tab1:
            # st.write(f"Filename={store_name}")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            
            # st.write('The current movie title is', title)
            # agree = st.checkbox('I agree')
            # if agree:
            #     st.write('Great!')
            
            
            

        with tab2:
            # st.header("tab2")
            # st.write(pdf_reader)

            st.header(f"{len(pages)} pages")
            # st.write(pages[0].page_content)
            st.header(f" pages[:3]")
            st.write(pages[:3])
            # st.write(chunks_faiss_embedding)
            print("\n\n***********st.session_state.docsSimilarity")
            
            
                
            
            
            st.write(len(st.session_state.docsSimilarity))
            
            

        with tab3:            
            st.write(f"chat with file : {st.session_state.filename}")
            st.header(f"{len(chunks)} chunks")
            st.write(chunks)
            
            

    ##############
    
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = "" 
            # docsSimilarity=fileProcess.getSimilarity(chunks_faiss_embedding,prompt)
            # kq=fAnswer.getAnswerNormal(chunks_faiss_embedding,prompt)

            

            if (optionAsk=="retrieve_new_page=False"):
                print("\n\n*******************retrieve_new_page=False************************\n\n")
                kq=fAnswer.getAnswerWithOldSimilar(st.session_state.docsSimilarity,prompt)

            elif (optionAsk=="retrieve_diff_page=True"):
                print("\n\n*******************retrieve_diff_page=True************************\n\n")
                kq=fAnswer.getAnswerWithOldSimilar_diffPage(st.session_state.docsSimilarity,prompt)

            elif (optionAsk=="retrieve_choose_Page"):
                print("\n\n*******************retrieve_choose_Page************************\n\n")
                kq=fAnswer.getAnswerFromPageList(pages,chatWithPages,prompt)            

            elif (optionAsk=="retrieve_external_knowledge=True"):
                print("\n\n*******************retrieve_external_knowledge=True************************\n\n")
                kq=fAnswer.getAnswerFromExternal(prompt)

            else:
                print("\n\n*******************getAnswerNormal************************")
                kq=fAnswer.getAnswerNormal(chunks_faiss_embedding,prompt)
                docsSimilarity=kq["docsSimilarity"]
                st.session_state.docsSimilarity=docsSimilarity 

            

            full_response=kq["answer"]
            # pagelistSimilairy=fileProcess.getSimilarPages(docsSimilarity)            
            # full_response+=f" || pagelistSimilairy={pagelistSimilairy}"
            
            
            
            message_placeholder.markdown(full_response)

            
        st.session_state.messages.append({"role": "assistant", "content": full_response})


    ###############
if __name__ =='__main__':
    main()