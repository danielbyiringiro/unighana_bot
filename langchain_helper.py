from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db_from_text() -> FAISS:
    transcript = open("data/ashesi.txt")
    transcript = transcript.readlines()
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #docs = text_splitter.split_text(transcript)

    # Embed documents using OpenAI

    db = FAISS.from_texts(transcript, embeddings)
    return db

def get_response_from_query(db: FAISS, query: str, k : int = 4) -> str:

    docs = db.similarity_search(query, k = k)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    llm = OpenAI(model = "text-davinci-003",  temperature=0.7)
    prompt = PromptTemplate(
        input_variables=['question', 'docs'],
        template="""
        
        You are a helpful assistant that that can answer questions about universities in Ghana. 
        based on the provided information about them.
        
        Answer the following question: {question}
        By searching the following documents: {docs}
        
        Only use the factual information from the documents to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm = llm, prompt = prompt)
    response = chain.run({'question': query, 'docs': docs_page_content})
    response = response.replace("\n", " ")
    return response, docs

#print(get_response_from_query(create_vector_db_from_text(), "What is the name of the university?"))