from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def create_conv_chain(vectorstore):
    llm = OllamaLLM(model="gemma3:270m")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Fix: store only the 'answer' in memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return conv_chain
