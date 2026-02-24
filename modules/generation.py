import logging
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    vector_store: any


def format_docs(retrieved_docs):
    return "\n\n".join(
        f"[Time: {doc.metadata.get('timestamp', 'Unknown')}]\n{doc.page_content}" for doc in retrieved_docs)


# --- THE WORKER NODES ---

def retrieve_node(state: AgentState):
    print("---NODE: RETRIEVING CONTEXT---")
    latest_question = state["messages"][-1].content
    vector_store = state["vector_store"]
    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

    base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
    advanced_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=model)
    docs = advanced_retriever.invoke(latest_question)

    return {"context": format_docs(docs)}


def generate_node(state: AgentState):
    print("---NODE: GENERATING TRANSCRIPT ANSWER---")
    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant answering questions about a video. 
        Answer ONLY using the provided transcript context. ALWAYS cite the exact [Time: MM:SS].

        If the transcript context does NOT contain the answer, do not guess or hallucinate. 
        Instead, output EXACTLY this exact phrase and nothing else: NEED_WEB_SEARCH

        Context:
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = qa_prompt | model | StrOutputParser()
    response = chain.invoke({
        "messages": state["messages"],
        "context": state.get("context", "No specific context provided.")
    })

    return {"messages": [AIMessage(content=response)]}


def summarize_node(state: AgentState):
    print("---NODE: SUMMARIZING ENTIRE VIDEO---")
    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    all_docs = list(state["vector_store"].docstore._dict.values())
    full_transcript = format_docs(all_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert summarizer. Write a comprehensive summary of the following video transcript. Highlight the main topics.\n\nTranscript:\n{context}"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()
    latest_question = state["messages"][-1].content
    response = chain.invoke({"context": full_transcript, "question": latest_question})
    return {"messages": [AIMessage(content=response)]}


def web_search_node(state: AgentState):
    print("---NODE: FALLING BACK TO TAVILY WEB SEARCH---")
    latest_question = state["messages"][-2].content

    search = TavilySearchResults(max_results=3)
    web_context = search.invoke(latest_question)

    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

    prompt = ChatPromptTemplate.from_messages([
        ("system", """The user asked a question about a video, but the video transcript did not have the answer. We searched the internet. 
        Here are the clean search results from Tavily:
        {web_context}

        Answer the user's question using ONLY the provided web search results.
        You MUST start your answer with this exact phrase: 
        "**This information was not found in the video transcript, but according to web search:**"
        """),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"web_context": web_context, "question": latest_question})

    return {"messages": [AIMessage(content=response)]}


# --- THE ROUTERS ---

def route_question(state: AgentState):
    print("---ROUTING QUESTION---")
    latest_question = state["messages"][-1].content.lower()
    clean_text = latest_question.replace("?", "").replace("!", "").replace(".", "").replace(",", "")
    words = clean_text.split()

    casual_greetings = ["hi", "hello", "hey", "thanks", "thank you"]
    summary_keywords = ["summarize", "summary", "tldr", "overview", "gist"]

    if any(greet in words for greet in casual_greetings):
        print("--> ROUTE: Casual Conversation")
        return "generate"
    elif any(keyword in words for keyword in summary_keywords):
        print("--> ROUTE: Summarization")
        return "summarize"
    else:
        print("--> ROUTE: RAG Search")
        return "retrieve"


def check_web_fallback(state: AgentState):
    latest_message = state["messages"][-1].content
    if "NEED_WEB_SEARCH" in latest_message:
        print("--> FALLBACK: Missing Info Detected -> Executing Web Search")
        return "web_search"
    else:
        print("--> COMPLETE: Transcript Answer Sufficient")
        return "end"


# --- COMPILE THE GRAPH ---

def get_agent(vector_store):
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("web_search", web_search_node)

    workflow.add_conditional_edges(START, route_question,
                                   {"retrieve": "retrieve", "generate": "generate", "summarize": "summarize"})
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges("generate", check_web_fallback, {"web_search": "web_search", "end": END})
    workflow.add_edge("summarize", END)
    workflow.add_edge("web_search", END)

    compiled_agent = workflow.compile()

    def run_agent(question: str, chat_history: list):
        messages = chat_history + [HumanMessage(content=question)]
        final_state = compiled_agent.invoke({
            "messages": messages,
            "vector_store": vector_store
        })
        return final_state["messages"][-1].content

    return run_agent