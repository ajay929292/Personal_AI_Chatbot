"""A minimal LCEL chatbot using Google Gemini."""

import os
from datetime import datetime
from math import sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


model = ChatGoogleGenerativeAI(
    # Gemini 2.5 Flash is available on the Gemini API free tier, subject to quota.
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.4")),
)

embeddings = HuggingFaceEmbeddings(
    # Free local embedding model; it downloads automatically on the first run.
    model_name=os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
)

knowledge_files = ["personal_information.txt", "goal.txt"]


def build_knowledge_base():
    """Embed non-empty knowledge-file chunks in memory (no vector database)."""
    documents = []
    for filename in knowledge_files:
        path = Path(filename)
        if path.exists() and (content := path.read_text(encoding="utf-8").strip()):
            documents.append(Document(page_content=content, metadata={"source": filename}))

    if not documents:
        return []

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(
        documents
    )
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    return list(zip(chunks, vectors))


knowledge_base = build_knowledge_base()


def retrieve_context(question: str) -> str:
    """Retrieve the most relevant chunks from the local knowledge files."""
    if not knowledge_base:
        return "No knowledge-file content has been added yet."
    question_vector = embeddings.embed_query(question)

    def cosine_similarity(vector: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(question_vector, vector))
        question_length = sqrt(sum(value * value for value in question_vector))
        vector_length = sqrt(sum(value * value for value in vector))
        return dot_product / (question_length * vector_length) if vector_length else 0.0

    documents = [
        chunk
        for chunk, _ in sorted(
            knowledge_base, key=lambda item: cosine_similarity(item[1]), reverse=True
        )[:3]
    ]
    return "\n\n".join(
        f"Source: {document.metadata['source']}\n{document.page_content}"
        for document in documents
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise and helpful assistant. Use a tool whenever it "
            "is useful, especially for arithmetic. The retrieved context is "
            "supplementary personal knowledge: use it when it is relevant, but "
            "answer general-knowledge questions using your own knowledge. Do "
            "not claim you cannot answer merely because it is absent from the "
            "retrieved context.\n\nRetrieved context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


@tool
def calculator(expression: str) -> str:
    """Calculate an arithmetic expression using only +, -, *, /, and parentheses."""
    try:
        allowed_characters = "0123456789+-*/.() "
        compact_expression = expression.replace(" ", "")
        if (
            not compact_expression
            or any(char not in allowed_characters for char in expression)
            or "**" in compact_expression
            or "//" in compact_expression
        ):
            raise ValueError("Only numbers, +, -, *, /, and parentheses are allowed.")
        answer = eval(compact_expression, {"__builtins__": {}}, {})
        if not isinstance(answer, (int, float)):
            raise ValueError("The result must be a number.")
        return str(answer)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        return f"Calculation error: {error}"


@tool
def count_words(text: str) -> str:
    """Count the words and characters in supplied text."""
    return f"Words: {len(text.split())}; characters: {len(text)}"


@tool
def current_time(timezone: str = "Asia/Kolkata") -> str:
    """Get the current date and time for an IANA timezone, e.g. Asia/Kolkata or Europe/London."""
    try:
        return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return f"Invalid timezone: {timezone}"


tools = [calculator, count_words, current_time]
tools_by_name = {selected_tool.name: selected_tool for selected_tool in tools}
model_with_tools = model.bind_tools(tools)


def run_chatbot(inputs: dict) -> AIMessage:
    """Run the model and execute any requested tools before returning its final answer."""
    messages = prompt.format_messages(
        **inputs, context=retrieve_context(inputs["question"])
    )

    for _ in range(5):  # Prevent an accidental infinite tool-calling loop.
        response = model_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response

        for tool_call in response.tool_calls:
            selected_tool = tools_by_name.get(tool_call["name"])
            if selected_tool:
                result = selected_tool.invoke(tool_call["args"])
            else:
                result = f"Unknown tool: {tool_call['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    return AIMessage(content="I stopped after too many tool calls.")


# LCEL chain: tool-enabled Ollama model -> final AI message -> plain text
chain = RunnableLambda(run_chatbot) | StrOutputParser()

# Keeps chat history in memory for each session ID while the app is running.
chat_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


if __name__ == "__main__":
    session_id = "default-session"
    print("\n" + "=" * 58)
    print("                 AJAY'S PERSONAL AI ASSISTANT")
    print("=" * 58)
    print("  Memory  |  RAG Knowledge  |  Calculator  |  Time")
    print("-" * 58)
    print("  Type 'exit' or 'quit' to end the conversation.")
    print("=" * 58 + "\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            response = chatbot.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"Assistant: {response}")
