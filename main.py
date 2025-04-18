import os
from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA
from pinecone import Pinecone as PineconeClient

# === CHAVES
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")


# === Inicializa Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# === Cria o vectorstore
vectorstore = Pinecone(
    index=index,
    embedding=OpenAIEmbeddings(),
    text_key="text"
)

# === Inicializa modelo da OpenAI
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4"
)

# === RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# === Cria app Flask
app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return "✅ Gus Meta Agent rodando com sucesso com LangChain + Pinecone v3!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question")

    result = rag_chain(query)

    response = {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
    }

    return jsonify(response)

@app.route("/chat")
def serve_chat():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
