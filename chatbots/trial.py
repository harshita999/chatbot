import os
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Set up Bedrock client and embeddings
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Define paths for storing FAISS index
FAISS_INDEX_DIR = 'faiss_index'
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, 'index.faiss')

def data_ingestion():
    """Loads PDF documents and splits them into smaller chunks."""
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    """Creates and saves a FAISS vector store from documents."""
    try:
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        
        # Ensure the directory exists
        if not os.path.exists(FAISS_INDEX_DIR):
            os.makedirs(FAISS_INDEX_DIR)
        
        # Save the FAISS index locally
        vectorstore_faiss.save_local(FAISS_INDEX_DIR)
        
        # Confirm the index has been saved correctly
        if os.path.exists(FAISS_INDEX_FILE) and os.path.getsize(FAISS_INDEX_FILE) > 0:
            print(f"FAISS index successfully saved at {FAISS_INDEX_FILE}")
        else:
            print(f"Failed to save FAISS index at {FAISS_INDEX_FILE}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def get_llama3_llm():
    """Returns the Meta Llama3 LLM."""
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Refined prompt template with more implicit instructions
prompt_template = """
Human: You are provided with some context and a question. Please answer the question using the information from the context. 
Ensure that your response is detailed and includes specific references to the sources, like document titles, page numbers, and other metadata.
Use the context provided below to frame your answer without explicitly repeating these instructions in your output.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    """Retrieves and returns the response from the LLM based on the query, including metadata."""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    
    result = answer['result']
    source_documents = answer['source_documents']
    
    # Create a detailed metadata response
    metadata_info = []
    for doc in source_documents:
        metadata = {
            "Document Title": doc.metadata.get('source', 'Unknown Document'),
            "Page Number": doc.metadata.get('page', 'Unknown Page'),
            "Paragraph": doc.metadata.get('paragraph', 'Unknown Paragraph'),
            "Chapter": doc.metadata.get('chapter', 'Unknown Chapter'),
            "Link": doc.metadata.get('link', 'No Link')
        }
        metadata_info.append(metadata)
    
    # Formatting metadata into a readable string
    metadata_str = "\n".join([f"Document: {meta['Document Title']}, Page: {meta['Page Number']}, Paragraph: {meta['Paragraph']}, Chapter: {meta['Chapter']}, Link: {meta['Link']}" for meta in metadata_info])

    return result, metadata_str

def main():
    st.set_page_config(page_title="Chat PDF using AWS Bedrock💁", layout='wide')
    
    st.header("Chat with PDF using AWS Bedrock")
    
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Update Vectors"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated successfully")
    
    def load_faiss_index():
        """Loads the FAISS index if it exists."""
        if os.path.exists(FAISS_INDEX_FILE):
            try:
                return FAISS.load_local(FAISS_INDEX_DIR, bedrock_embeddings)
            except Exception as e:
                st.error(f"Error loading FAISS index: {e}")
        else:
            st.error(f"FAISS index file not found at {FAISS_INDEX_FILE}")
            return None

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = load_faiss_index()
            if faiss_index:
                llm = get_llama3_llm()
                response, metadata = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.write("### Metadata of Source Documents")
                st.write(metadata)
                st.success("Llama3 LLM Response Retrieved")

if __name__ == "__main__":
    main()
