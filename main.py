from flask import Flask, request, render_template
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import os
import pinecone

app = Flask(__name__)

# Variáveis de ambiente (adicione no Render depois)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "gcp-starter"
PINECONE_INDEX = "chatbotgus"

# Inicializa Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)
vectorstore = Pinecone(index, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), "text")

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
    app.run(debug=True)
