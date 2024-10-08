import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Load DASA data and create retriever
loader = WebBaseLoader("https://www.dasa.org")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(retriever, "DASA_search", "Search for information about DASA.")

# Load Tavily Search Tool
 # Ensure to set your API key here
search = TavilySearchResults()

# Define tools
tools = [retriever_tool, search]

# Load prompt from LangChain Hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Initialize the LLM
llm = ChatOpenAI()  # Ensure to set your API key here

# Create the AI agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit App
st.title("AI Agent with LangChain")
st.write("Ask anything about DASA or general queries!")

user_input = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_input:
        response = agent_executor.invoke({"input": user_input})
        st.write("AI Agent Response:")
        st.write(response['output'])
    else:
        st.write("Please enter a question.")
