from MyPyClass.FileProcess import FileProcess
class FAnswer:
    
    def getAnswerNormal(self,chunks_faiss_embedding,question):
        print("\n\n*******************getAnswerNormal************************")
        fileProcess = FileProcess()
        kq={"docsSimilarity": [], "answer":""}
        kq["docsSimilarity"]=fileProcess.getSimilarity(chunks_faiss_embedding,question)
        print(f"\n\n *** Question={question}")
        print("\n docsSimilarity=")
        print(kq["docsSimilarity"])

        kq["answer"]=fileProcess.getAnswer(kq["docsSimilarity"][:1],question)
        print(f"\n\n *** kq['answer']")
        print(kq)
        return kq
    
    def getAnswerWithOldSimilar(self,docsSimilarity,question):
        print("\n\n*******************getAnswerWithOldSimilar************************")
        fileProcess = FileProcess()
        kq={"docsSimilarity": [], "answer":""}
        kq["docsSimilarity"]=docsSimilarity
        print(f"\n\n *** Question={question}")
        print("\n docsSimilarity=")
        print(kq["docsSimilarity"])

        kq["answer"]=fileProcess.getAnswer(kq["docsSimilarity"][:1],question)
        print(f"\n\n *** kq['answer']")
        print(kq)
        return kq
    
    def getAnswerWithOldSimilar_diffPage(self,docsSimilarity,question):
        print("\n\n*******************getAnswerWithOldSimilar_diffPage************************")
        fileProcess = FileProcess()
        kq={"docsSimilarity": [], "answer":""}
        kq["docsSimilarity"]=docsSimilarity
        print(f"\n\n *** Question={question}")
        print("\n docsSimilarity=")
        print(kq["docsSimilarity"])
        lenDocs=len(kq["docsSimilarity"])
        kq["answer"]=fileProcess.getAnswer(kq["docsSimilarity"][1:lenDocs],question)
        print(f"\n\n *** kq['answer']")
        print(kq)
        return kq
    
    def getAnswerFromExternal(self,question):
        print("\n\n*******************getAnswerFromExternal************************")
        fileProcess = FileProcess()
        kq={"docsSimilarity": [], "answer":""}
        kq["docsSimilarity"]=fileProcess.createDocsEmpty()
        print(f"\n\n *** Question={question}")
        print("\n docsSimilarity=")
        print(kq["docsSimilarity"])
        lenDocs=len(kq["docsSimilarity"])
        kq["answer"]=fileProcess.getAnswer(kq["docsSimilarity"],question)
        print(f"\n\n *** kq['answer']")
        print(kq)
        return kq
    
    def getAnswerFromPageList(self,pages,pageList,question):
        print("\n\n*******************getAnswerFromPageList************************")
        fileProcess = FileProcess()
        kq={"docsSimilarity": [], "answer":""}
        kq["docsSimilarity"]=fileProcess.createDocs(pages,pageList)      
        print(f"\n\n *** Question={question}")
        print("\n docsSimilarity=")
        print(kq["docsSimilarity"])
        lenDocs=len(kq["docsSimilarity"])
        kq["answer"]=fileProcess.getAnswer(kq["docsSimilarity"],question)
        print(f"\n\n *** kq['answer']")
        print(kq)
        return kq

