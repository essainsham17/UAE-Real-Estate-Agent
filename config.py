import os
import joblib
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.memory import MemorySaver



memory=MemorySaver()
load_dotenv()
embed_api=os.getenv("HF_API_KEY") # i saved in .env as HF_API_KEY


GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

search_tool=TavilySearch(
    max_results=5,
    api_key=os.getenv("TAVILY_API_KEY")
)
embed=HuggingFaceEndpointEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2',
                                   huggingfacehub_api_token=embed_api)
llm=ChatGroq(
    model='llama-3.3-70b-versatile', temperature=0
    )
vectorstore=Chroma(persist_directory="./chroma_db", embedding_function=embed)
retriever=vectorstore.as_retriever(search_kwargs={"k":3})

prediction_model=joblib.load("price_prediction_model.pkl")
columns=joblib.load("model_columns.pkl")