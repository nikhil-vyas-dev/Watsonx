import streamlit as st
import PyPDF2
import base64
from elasticsearch import Elasticsearch
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Elasticsearch connection details
ES_URL = "<elasticsearch-endpoint:port>"
ES_USERNAME = "<elasticsearch-username>"
ES_PASSWORD = "<elasticsearch-password>"

# IBM Watson credentials
WATSON_API_KEY = "<ibm-cloud-api-key>"
WATSON_URL = "https://us-south.ml.cloud.ibm.com"
WATSON_PROJECT_ID = "<watsonx-project-id>"

# Connect to Elasticsearch
@st.cache_resource
def get_es_client():
    return Elasticsearch(
        ES_URL,
        basic_auth=(ES_USERNAME, ES_PASSWORD),
        verify_certs=False,  # For development only
        request_timeout=10000
    )

def create_elser_pipeline(es_client):
    """Create the ELSER pipeline if it doesn't exist"""
    pipeline_body = {
        "description": "Create ELSER embedding",
        "processors": [
            {
                "inference": {
                    "model_id": ".elser_model_2_linux-x86_64",
                    "target_field": "elser_embedding",
                    "field_map": {
                        "content": "text_field"
                    },
                    "inference_config": {
                        "text_expansion": {
                            "results_field": "tokens"
                        }
                    }
                }
            }
        ]
    }
    
    try:
        # Check if pipeline exists
        es_client.ingest.get_pipeline(id="elser_pipeline")
        st.write("ELSER pipeline already exists")
    except Exception as e:
        # Create pipeline if it doesn't exist
        try:
            es_client.ingest.put_pipeline(id="elser_pipeline", body=pipeline_body)
            st.write("Created ELSER pipeline successfully")
        except Exception as e:
            st.error(f"Error creating ELSER pipeline: {str(e)}")
            # Check if ELSER model is deployed
            try:
                models = es_client.ml.get_trained_models()
                st.write("Available models:", models)
            except Exception as e:
                st.error(f"Error checking models: {str(e)}")

def create_index_if_not_exists(es_client, index_name="pdf_documents"):
    try:
        if not es_client.indices.exists(index=index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "page_number": {"type": "integer"},
                        "title": {"type": "text"},
                        "elser_embedding.tokens": {
                            "type": "rank_features"
                        }
                    }
                }
            }
            es_client.indices.create(index=index_name, body=mapping)
            st.write(f"Created index {index_name}")
        
        # Always try to create/verify ELSER pipeline
        create_elser_pipeline(es_client)
        
    except Exception as e:
        st.error(f"Error setting up Elasticsearch: {str(e)}")

# Set up IBM Watson Foundation model
@st.cache_resource
def get_watson_model():
    credentials = Credentials(
        url=WATSON_URL,
        api_key=WATSON_API_KEY,
    )
    client = APIClient(credentials)
    return ModelInference(
        model_id="ibm/granite-3-8b-instruct",
        api_client=client,
        project_id=WATSON_PROJECT_ID,
        params={
            "max_new_tokens": 1000,
            "min_new_tokens": 30,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "stop_sequences": ["\n\n"]
        }
    )

@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=512,  # Exact ELSER model limit
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

def read_pdf_chunks(file):
    text_splitter = get_text_splitter()
    tokenizer = get_tokenizer()
    chunks = []
    pdf_reader = PyPDF2.PdfReader(file)
    title = pdf_reader.metadata.title or file.name
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        text = page.extract_text()
        # Clean text to remove excessive whitespace
        text = ' '.join(text.split())
        
        # Split text into chunks and verify token count
        page_chunks = text_splitter.split_text(text)
        for chunk in page_chunks:
            # Verify chunk size in tokens
            tokens = tokenizer.encode(chunk)
            if len(tokens) <= 512:  # ELSER's limit
                chunks.append((page_num, title, chunk))
            else:
                # If chunk is too large, truncate it
                truncated_chunk = tokenizer.decode(tokens[:512])
                chunks.append((page_num, title, truncated_chunk))
    
    return chunks

def index_document(es_client, index_name, doc_id, page_num, title, content):
    # Initial document body without embeddings
    initial_body = {
        "content": content,
        "page_number": page_num,
        "title": title
    }
    
    try:
        # Index the document using the ELSER pipeline
        response = es_client.index(
            index=index_name,
            id=doc_id,
            body=initial_body,
            pipeline="elser_pipeline"
        )
        
        # Force refresh to make document immediately available
        es_client.indices.refresh(index=index_name)
        return response
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        return None

def retrieve_from_elasticsearch(es_client, query, top_k=3):
    search_body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "text_expansion": {
                            "elser_embedding.tokens": {
                                "model_id": ".elser_model_2_linux-x86_64",
                                "model_text": query
                            }
                        }
                    },
                    {
                        "match": {
                            "content": {
                                "query": query,
                                "boost": 0.3
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["content", "title", "page_number"],
        "highlight": {
            "fields": {
                "content": {
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "fragment_size": 200,
                    "number_of_fragments": 3,
                    "type": "unified"
                }
            }
        }
    }
    
    try:
        results = es_client.search(index="pdf_documents", body=search_body)
        retrieved_docs = []
        for hit in results['hits']['hits']:
            doc = {
                "content": hit['_source']['content'],
                "title": hit['_source']['title'],
                "page": hit['_source']['page_number'],
                "score": hit['_score'],
                "highlights": hit.get('highlight', {}).get('content', [])
            }
            retrieved_docs.append(doc)
        return retrieved_docs
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def generate_with_watson(model, input_text):
    response = model.generate(input_text)
    return response

def rag_pipeline(es_client, model, query):
    retrieved_docs = retrieve_from_elasticsearch(es_client, query)
    
    # Build context with LLM's capacity
    context_parts = []
    max_tokens_per_doc = 1000
    tokenizer = get_tokenizer()
    
    for doc in retrieved_docs:
        doc_text = f"Document: {doc['title']}, Page: {doc['page']}\nContent: {doc['content']}"
        # Use proper tokenization
        tokens = tokenizer.encode(doc_text)
        
        if len(tokens) > max_tokens_per_doc:
            # Properly truncate tokens and decode back to text
            doc_text = tokenizer.decode(tokens[:max_tokens_per_doc])
        
        context_parts.append(doc_text)
    
    context = "\n".join(context_parts)
    augmented_query = f"""Context:
{context}

Question: {query}

Please provide a detailed answer based on the context above. If the context doesn't contain relevant information, please indicate that. Answer:"""
    
    response = generate_with_watson(model, augmented_query)
    return response, retrieved_docs

def main():
    st.set_page_config(page_title="Watson RAG with PDF Ingestion", layout="wide")
    st.title("Watsonx RAG with PDF Ingestion")

    es_client = get_es_client()
    watson_model = get_watson_model()
    
    # Add initialization status section
    st.header("Initialization Status")
    with st.expander("View Elasticsearch Setup Details"):
        # Ensure index exists with proper mapping
        create_index_if_not_exists(es_client)
        
        # Check if ELSER model is available
        try:
            models = es_client.ml.get_trained_models(model_id=".elser_model_2_linux-x86_64")
            st.write("ELSER model status:", models)
        except Exception as e:
            st.error(f"ELSER model not found: {str(e)}")

    # PDF Ingestion Section
    st.header("PDF Ingestion")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Ingest PDF"):
            with st.spinner("Ingesting PDF..."):
                chunks = read_pdf_chunks(uploaded_file)
                progress_bar = st.progress(0)
                for i, (page_num, title, content) in enumerate(chunks):
                    doc_id = f"{base64.b64encode(uploaded_file.name.encode()).decode()}_chunk_{i}"
                    response = index_document(es_client, "pdf_documents", doc_id, page_num, title, content)
                    if response:
                        progress = (i + 1) / len(chunks)
                        progress_bar.progress(progress)
                        st.write(f"Indexed chunk {i+1} from {title}, page {page_num}: {response['result']}")
            st.success("PDF ingested successfully!")

    # RAG Question-Answering Section
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question:")
    if user_query:
        with st.spinner("Generating answer..."):
            rag_response, retrieved_docs = rag_pipeline(es_client, watson_model, user_query)
        
        st.subheader("Answer:")
        st.write(rag_response)
        
        st.subheader("Retrieved Context:")
        for doc in retrieved_docs:
            with st.expander(f"Document: {doc['title']}, Page: {doc['page']} (Score: {doc['score']:.2f})"):
                if doc['highlights']:
                    st.markdown("### Relevant Excerpts:")
                    for highlight in doc['highlights']:
                        st.markdown(highlight, unsafe_allow_html=True)
                st.markdown("### Full Content:")
                st.write(doc['content'])

if __name__ == "__main__":
    main()