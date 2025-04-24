from flask import Flask, request, render_template
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone  # novo cliente
import os

app = Flask(__name__)

# Variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "chatbotgus"

# Inicializa Pinecone com novo padrão
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Configura vectorstore com Langchain
vectorstore = LangchainPinecone(
    index,
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    "text"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    retriever=vectorstore.as_retriever()
)

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        question = request.form["question"]
        answer = qa_chain.run(question)
        return render_template("chat.html", question=question, answer=answer)
    return render_template("chat.html", question=None, answer=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
