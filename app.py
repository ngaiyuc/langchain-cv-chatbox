import streamlit as st
import os
import tempfile
import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from mistralai import Mistral
from dotenv import load_dotenv, dotenv_values
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from qdrant_client import QdrantClient
import smtplib
from email.mime.text import MIMEText
from email.header import Header
# res = classifier(query, candidate_labels=["technical_competency","work_experience","leadership_skills", "communication_skills","problem_solving","cultural_fit","career_goals","project_management","team_collaboration","conflict_resolution"])

load_dotenv()

st.markdown(
    '<meta name="robots" content="noindex, nofollow" />',
    unsafe_allow_html=True
)

st.markdown("""
    <style>
        .result-box {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        .result-title {
            font-weight: bold;
            color: #4CAF50;
        }
    
    @media (max-width: 768px) {
    /* Make select box text wrap to new line */
    .stSelectbox {
        white-space: normal !important;
        height: auto !important;
        min-height: 40px !important;
    }
    
    /* Make text input expand vertically */
    .stTextInput textarea {
        height: auto !important;
        min-height: 100px !important;
        white-space: pre-wrap !important;
    }
    
    /* Show full text without truncation */
    .stSelectbox div[role="listbox"] div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    
    /* Target the specific div elements that show selected/option text */
    div[class*="st-"][class*="st-ak"][class*="st-bn"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        height: auto !important;
        min-height: 40px !important;
        padding: 8px !important;
    }
    
    /* Ensure container expands */
    div[data-baseweb="select"] {
        height: auto !important;
        min-height: 40px !important;
    }
}
        
    </style>
""", unsafe_allow_html=True)


def sendEmailGmail(subject=None, message=None):
    try:
        sender = os.getenv('GMAIL_SENDER')
        recipient = os.getenv('GMAIL_RECIPIENT')
        password = os.getenv('GMAIL_PASSWORD')

        if message is None:
            message = "Default Message"

        msg = MIMEText(message, 'plain', 'utf-8')

        if subject is None:
            subject = "Subject in a Default (ENV)"
        
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = sender
        msg['To'] = recipient

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
            connection.login(user=sender, password=password)
            connection.send_message(msg)   
    except:
        pass

role = "Personal Branding Advisor"
custom_prompt_template = f"As a {role}, please provide responses to the interviewer's questions based on the query: {{query}}. Ensure that the response highlights my strengths and key qualifications. Use inclusive language, replacing 'he' with 'they', 'we', or 'I' as appropriate."


class MistralEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-embed"
        self.batch_size = 3  # Reduced batch size
        self.sleep_time = 0.5  # Sleep time between individual requests
        self.batch_sleep_time = 2  # Sleep time between batches

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            st.write(f"Processing embedding batch {i//self.batch_size + 1} of {(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            for text in batch:
                try:
                    response = self.client.embeddings.create(model=self.model, inputs=[text])
                    embeddings.append(response.data[0].embedding)
                    time.sleep(self.sleep_time)
                except Exception as e:
                    # st.error(f"Error in batch {i//self.batch_size + 1}: {str(e)}")
                    # st.write("Waiting for 30 seconds before retrying...")
                    time.sleep(30)
                    try:
                        response = self.client.embeddings.create(model=self.model, inputs=[text])
                        embeddings.append(response.data[0].embedding)
                    except Exception as retry_e:
                        st.error(f"Retry failed: {str(retry_e)}")
                        raise retry_e
            
            if i + self.batch_size < len(texts):
                # st.write(f"Sleeping for {self.batch_sleep_time} seconds before next batch...")
                time.sleep(self.batch_sleep_time)
                
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(model=self.model, inputs=[text])
                return response.data[0].embedding
            except Exception as e:
                if attempt < max_retries - 1:
                    # st.error(f"Error embedding query (attempt {attempt + 1}): {str(e)}")
                    # st.write("Waiting 30 seconds before retrying...")
                    time.sleep(30)
                else:
                    st.error(f"Final error embedding query: {str(e)}")
                    raise e

def ensure_collection_exists(client: QdrantClient, collection_name: str, embeddings: MistralEmbeddings) -> bool:
    try:
        collections = client.get_collections().collections
        exists = any(collection.name == collection_name for collection in collections)
        
        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": 1024,  # Mistral embedding dimension
                    "distance": "Cosine"
                }
            )
            return True
        return True
    except Exception as e:
        st.error(f"Error checking/creating collection: {str(e)}")
        return False

st.title("👨🏼‍💻 Welcome to know about me")

mistral_api_key = os.getenv("MISTRAL_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not (mistral_api_key and openrouter_api_key and qdrant_url and qdrant_api_key):
    st.warning("Please provide all API keys.")
else:
    embeddings = MistralEmbeddings(api_key=mistral_api_key)

    # Page selection
    page = st.sidebar.selectbox("Select Page", ["Know More about him", "Upload Document"])

    if page == "Upload Document":
        uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
        collection_name = None
        
        if uploaded_file:
            collection_name = st.text_input("Enter a custom collection name:", value=f"knowledge_base_{hash(uploaded_file.name)}")
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            try:
                st.write("Step 1: Loading document...")
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)

                documents = loader.load()
                st.write(f"Documents loaded. Number of pages/sections: {len(documents)}")

                st.write("Step 2: Converting to Document objects...")
                if isinstance(documents, list) and all(isinstance(doc, str) for doc in documents):
                    documents = [Document(page_content=text) for text in documents]

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                split_docs = text_splitter.split_documents(documents)
                st.write(f"Step 3: Number of chunks after splitting: {len(split_docs)}")

                texts = [doc.page_content for doc in split_docs]
                st.write(f"Step 4: Total number of text chunks to embed: {len(texts)}")
                
                embeddings_vectors = embeddings.embed_documents(texts)
                st.write(f"Embeddings generated successfully. Total vectors: {len(embeddings_vectors)}")

                st.write("Step 5: Storing vectors in Qdrant...")
                collection_name = f"knowledge_base_{hash(uploaded_file.name)}"
                
                # Initialize Qdrant client and ensure collection exists
                qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                ensure_collection_exists(qdrant_client, collection_name, embeddings)
                
                vectorstore = Qdrant.from_documents(
                    documents=split_docs,
                    embedding=embeddings,
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    collection_name=collection_name,
                    force_recreate=True
                )
                st.write("Vectors stored successfully")

                # st.write("Step 6: Setting up QA chain...")
                # qa_chain = RetrievalQA.from_chain_type(
                #     llm=llm,
                #     chain_type="stuff",
                #     retriever=vectorstore.as_retriever(),
                #     return_source_documents=True
                # )
                # st.write("QA chain ready")

                # st.subheader("Ask a question about your document")
                # query = st.text_input("Enter your question:")

                # if query:
                #     with st.spinner("Searching for answer..."):
                #         result = qa_chain({"query": query})
                #         st.write("Answer:", result["result"])

                #         st.subheader("Source Documents:")
                #         for i, doc in enumerate(result["source_documents"]):
                #             with st.expander(f"Source {i+1}"):
                #                 st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

            os.unlink(file_path)

    elif page == "Know More about him":      
        llm = ChatOpenAI(api_key=openrouter_api_key, model="google/gemma-2-9b-it:free", temperature=0.6, openai_api_base="https://openrouter.ai/api/v1")
  
        try:
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            collections = qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_names:
                selected_collection = st.selectbox("Choose a Collection", collection_names)
                
                if selected_collection:
                    embeddings = MistralEmbeddings(api_key=mistral_api_key)
                    
                    if ensure_collection_exists(qdrant_client, selected_collection, embeddings):
                        vectorstore = Qdrant(
                            client=qdrant_client,
                            collection_name=selected_collection,
                            embeddings=embeddings
                        )
                        
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vectorstore.as_retriever(),
                            return_source_documents=True
                        )

                        preset_questions = [
                            "What leadership experiences has he had in detail?",
                            "What communication skills does he possess that contribute to his effectiveness?",
                            "What is his approach to problem-solving?",
                            "What skills does he have related to data?",
                            "What IT applications does he have in marketing?",
                            "Does he have any government experience that demonstrates his integrity? Please list the years.",
                        ]

                        selected_question = st.selectbox("Choose a preset question:", [""] + preset_questions)
                        
                        query = st.text_input("or Enter your question:", value=selected_question if selected_question else "")

                        if query:
                            try:
                                with st.spinner("Searching for data..."):
                                    formatted_query = custom_prompt_template.format(query=query)
                                    result = qa_chain({"query": query})
                                    
                                    if result and "result" in result:
                                        st.markdown(f"""
                                            <div class="result-box">
                                                <div class="result-title">🤖:</div>
                                                {result["result"]}
                              
                                        """, unsafe_allow_html=True)
                                    

                                    try:
                                        subject = f"Query Result: {query[:50]}..." if len(query) > 50 else f"Query Result: {query}"
                                        email_message = f"""
                                        Query: {query}
                                        
                                        Result: {result["result"]}
                                        """
                                        
                                        
                                        sendEmailGmail(subject, email_message)
                                    except Exception:
                                        pass
                            

                                    st.subheader("Source Documents:")
                                    if result.get("source_documents"):
                                        for i, doc in enumerate(result["source_documents"]):
                                            with st.expander(f"Source Document {i + 1}"):
                                                st.write(doc.page_content)
                                    else:
                                        st.write("No source documents found for this query.")
                                
                                
                            except Exception as e:
                                st.error(f"Error during query processing: {str(e)}")
            else:
                st.write("No collections found. Please upload a document first.")
                
        except Exception as e:
            st.error(f"Error accessing Qdrant: {str(e)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Notice:
    1. This project is focused solely on the following aspects, RAG, NLP and API integration with Langchain.
    2. The project does not address user experience (UX), user interface (UI) design, search engine optimization (SEO), or other non-functional requirements.
    3. The adoption of Qdrant and Mistral embedding models is based on feasibility studies conducted prior to implementation.
                        
    """)