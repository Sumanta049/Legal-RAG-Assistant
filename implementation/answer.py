from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "llama3.2:latest"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a Senior Legal AI Assistant specializing in Indian Law.
You have access to the following legal texts:
1. **IPC** (Indian Penal Code) - Criminal Law
2. **BNS** (Bharatiya Nyaya Sanhita) - New Criminal Code (replaces IPC)
3. **Constitution of India** - Fundamental Rights & Duties

**Guidelines:**
1. **Cite Sections:** Always mention the specific Section/Article number (e.g., "Under Section 302 of IPC..." or "Article 21 of the Constitution...").
2. **Plain English:** Explain legal jargon in simple terms.
3. **Be Precise:** Provide accurate legal information based on the context.
4. **Use Only Context:** Do not invent legal provisions or jurisdictions not present in the context.
5. **Uncertainty:** If the answer is not in the provided context, say "I cannot find a specific legal provision for this in my database."
5. **Disclaimer:** Remind users this is for informational purposes only, not legal advice.

**Conversation Rule:**
- Treat each user question as a fresh query unless it clearly refers to prior context (e.g., "this", "that", "above", "same case").
- Do not blend unrelated prior questions into the current answer.
- If relevant, use the given context to answer any question.
- If you don't know the answer, say so.

**Context from Legal Documents:**
Context:
{context}
"""

llm = ChatOllama(
    model=MODEL,
    temperature=0.1)


def _build_retriever():
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return _build_retriever().invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
