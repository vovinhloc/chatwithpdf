from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle,os
from langchain.docstore.document import Document
class FileProcess:
    # def __init__(self, name):
    #     self.name = name

    # def say_hello(self):
    #     print(f"Hello, {self.name}!")

    # def add_numbers(self, a, b):
    #     return a + b
    def saveFile(self, pdf):
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        return 1
    def loadPdfFileToPagesThroughLangchain(self, pdf):        
        loader = PyPDFLoader(pdf.name)
        pages = loader.load()
        return pages
    def loadPdfFileToPagesByPyPDF2(self, pdf):        
        pdf_reader=PdfReader(pdf)
        text=""
        i=0
        for page in pdf_reader.pages:
            # st.markdown(f"##i")
            i=i+1
            text +=page.extract_text()
            
            if (i>1):
                break
        return text
    
    def createChunks(self, pages):        
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
            chunk_overlap=100,
            length_function=len
            )
        chunks = splitter.split_documents(pages[:3])
        return chunks
    
    def loadHFVectorEmbeddings(self,faissName,chunks):
        # store_name=pdf.name[:-4]
        # st.write(store_name)
        if(os.path.exists(f"{faissName}.pkl")):
            with open(f"{faissName}.pkl","rb") as f:
                VectorStore=pickle.load(f)            
        else:
            embeddings=HuggingFaceEmbeddings()
            VectorStore=FAISS.from_documents(chunks,embedding=embeddings)
            with open(f"{faissName}.pkl","wb") as f:
                pickle.dump(VectorStore,f) 
            
        return VectorStore
    
    def loadOpenAIVectorEmbeddings(self,faissName,chunks):
        # store_name=pdf.name[:-4]
        # st.write(store_name)
        if(os.path.exists(f"{faissName}.pkl")):
            with open(f"{faissName}.pkl","rb") as f:
                VectorStore=pickle.load(f)            
        else:
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{faissName}.pkl","wb") as f:
                pickle.dump(VectorStore,f) 
            
        return VectorStore
    
    def getSimilarity(self,chunks_faiss_embedding,query):  
        docs = chunks_faiss_embedding.similarity_search(query)
        # print("\n\n***********query")
        # print(query)
        # print("\n***********docs similarity")
        # print(docs)
        print(len(docs))
        i = 0
        pagelistSimilairy=''
        while i < len(docs):        
            # print(f"\n.........docs[{i}]")
            # print(docs[i].page_content)
            # print(docs[i].metadata)
            pagelistSimilairy+=f"{docs[i].metadata['page']},"
            i += 1
            print("\n")
        print(f"\n***pagelistSimilairy={pagelistSimilairy}")
        return docs
    
    def getAnswer(self,input_documents,question):
        question=self.refineQuestion(question=question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type='stuff')
        response = chain.run(input_documents=input_documents, question=question)
        return response
    
    def refineQuestion(self,question):
        # kq=f"Từ thông tin được cung cấp, vui lòng cho mình thông tin cho câu hỏi sau, trả lời bằng chuỗi json với key là \"Answer\" là câu trả lời, \"Source\" là nguồn thông tin cho câu hỏi này,\"page\" là  số trang (page) chứa thông tin của câu hỏi này, \"isExternal\" là 1 nếu thông tin được cung cấp từ bên ngoài và là 0 nếu thông tin được cung cấp từ tài liệu này,nếu bạn không biết thì \"Answer\" là '0', câu hỏi như sau : {question} "
        kq=f"vui lòng cho mình thông tin cho câu hỏi sau, nếu bạn không biết hoặc không có thông tin thì trả lời là '0', câu hỏi như sau : {question} "
        
        return question
    
    def getSimilarPages(self,docsSimilarity):
        pagelistSimilairy=""
        i=0
        while i < len(docsSimilarity):                 
            pagelistSimilairy+=f"{docsSimilarity[i].metadata['page']},"
            i += 1        
        return pagelistSimilairy
    
    def createDocs(self,pages,pageList):        
        document =  []
        list = pageList.split(",")
        for item in list:
            i=int(item)
            localText=pages[i].page_content
            localdoc =  Document(page_content=localText, metadata={"source": "createDocs","page":i})
            document.append(localdoc)
        return document
    
    def createDocsEmpty(self):
        localText=""
        localdoc =  Document(page_content=localText, metadata={"source": "createDocs","page":""})
        document =  []
        document.append(localdoc)
        return document
    