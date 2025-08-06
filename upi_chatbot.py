import streamlit as st
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Configure page
st.set_page_config(page_title="UPI Expert Chatbot", page_icon="💳")
st.title("🏦 UPI Domain Expert Chatbot")

# Initialize components
@st.cache_resource
def init_components():
    embeddings = CohereEmbeddings(cohere_api_key="N7tFWrpDq4s2axWYO1PNYQzhSMqmi9IQbl0V4r8B", model="embed-english-v3.0")
    vectorstore = Chroma(persist_directory="upi_knowledge", embedding_function=embeddings)
    llm = ChatGroq(groq_api_key=st.secrets.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY"), model_name="mixtral-8x7b-32768", temperature=0.1)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), return_source_documents=True)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your UPI expert. Ask me anything about UPI payments, regulations, or technical details."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about UPI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        qa_chain = init_components()
        response = qa_chain({"query": f"As a UPI domain expert, provide detailed analysis: {prompt}"})
        answer = response["result"]
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})