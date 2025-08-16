import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load documents from knowledge base
loader = TextLoader("data/knowledge_base.txt")
documents = loader.load()

# Create embeddings and FAISS vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(documents, embeddings)

# Create RetrievalQA pipeline
retriever = db.as_retriever()
llm = OpenAI(openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Example queries for RAG
queries = [
    "Who founded Arista Networks?",
    "What is Vertex AI?",
    "How does LangChain help developers?",
    "Explain the purpose of FAISS.",
    "What is Confluence used for?",
    "What does GitLab do?"
]

results = []
for query in queries:
    result = qa_chain(query)
    answer = result["result"]
    sources = [doc.page_content for doc in result["source_documents"]]
    results.append((query, answer, sources))
    print(f"Q: {query}\nA: {answer}\nContext Snippet: {sources}\n{'-'*40}")

# Save results to output file
with open("output/sample_rag_results.txt", "w") as fout:
    for query, answer, sources in results:
        fout.write(f"Q: {query}\nA: {answer}\nContext Snippet: {sources}\n{'-'*40}\n")
