import os
import streamlit as st
import operator
import functools

from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from agent_tools import *

load_dotenv(find_dotenv()) # Load .env file 

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']
LANGCHAIN_TRACING_V2 = os.environ['LANGCHAIN_TRACING_V2']
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']

# Create Tools 

# Tavily Search Tools
tavily_tool = TavilySearchResults(max_results=5)

# GA4 Knowledge Tool
ga4vectordb = Chroma(persist_directory='./chroma/ga4', embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
ga4retriever = ga4vectordb.as_retriever()

ga4_docs = create_retriever_tool(
    ga4retriever,
    "GA4_documentation",
    "Search and return information about GA4 implementation and set up",
)

# Meta Pixel Knowledge Tool
mpvectordb = Chroma(persist_directory='./chroma/meta_pixel', embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
mpretriever = mpvectordb.as_retriever()

mp_docs = create_retriever_tool(
    mpretriever,
    "Meta_Pixel_documentation",
    "Search and return information about Meta Pixel implementation and set up",
)

members = ["Researcher", "Google Expert", 'Meta Pixel Expert']
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4-1106-preview")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)



# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

google_agent = create_agent(
    llm,
    [ga4_docs],
    "You are a Google and GA4 expert who refers to documentation to anwer questions.",
)
google_node = functools.partial(agent_node, agent=google_agent, name="Google Expert")

meta_agent = create_agent(
    llm,
    [mp_docs],
    "You are a Meta Pixel and Facebook Export who refers to documentation to anwer questions.",
)
meta_node = functools.partial(agent_node, agent=meta_agent, name="Meta Pixel Expert")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Google Expert", google_node)
workflow.add_node("Meta Pixel Expert", meta_node)
workflow.add_node("supervisor", supervisor_chain)


for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# Streamlit UI elements
st.title("Experts Ensemble")
st.subheader("Your Experts: ")
st.subheader(members)

# Input from user
input_text = st.chat_input("Enter your text:")

if input_text:
    for s in graph.stream(
        {   
            "messages": [
                HumanMessage(content=input_text)
            ]
        }
    ):
        if "__end__" not in s:
            st.write(s)
            st.write("----")