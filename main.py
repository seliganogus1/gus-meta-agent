from flask import Flask, request, render_template
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone
import os

app = Flask(__name__)

# Variáveis de ambiente (Render configura via painel)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "chatbotgus")

# Inicializa cliente Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Cria vectorstore
vectorstore = LangchainPinecone(
    index=index,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    text_key="text"
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