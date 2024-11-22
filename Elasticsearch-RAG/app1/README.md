# Watsonx RAG with PDF Ingestion

A Streamlit application that combines IBM Watsonx.AI with Elasticsearch to provide intelligent document search and question-answering capabilities for PDF documents.

## Features

- PDF document ingestion and indexing
- Semantic search using Elasticsearch ELSER (Elastic Learned Sparse EncodeR)
- Question-answering using IBM Watson Foundation Models
- Interactive web interface built with Streamlit
- Highlighted search results with relevant context

## Prerequisites

- Python 3.8+
- Elasticsearch instance with ELSER model deployed
- IBM Cloud account with Watson Machine Learning service
- Required Python packages (see `requirements.txt`)

## Setup

1. Clone this repository: 

```bash
git clone https://github.com/nikhil-vyas-dev/Watsonx.git
cd Watsonx/Elasticsearch-RAG/app1
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your environment variables:

Update the following variables in `streamlit_watson_rag.py` with your credentials:

```python
# Elasticsearch connection details
ES_URL = "<elasticsearch-endpoint>:port"
ES_USERNAME = "<elasticsearch-username>"
ES_PASSWORD = "<elasticsearch-password>"

# IBM Watson credentials
WATSON_API_KEY = "<ibm-cloud-api-key>"
WATSON_URL = "https://us-south.ml.cloud.ibm.com"
WATSON_PROJECT_ID = "<watsonx-project-id>"
```

## Running the Application

1. Start the Streamlit application:
```bash
streamlit run streamlit_watson_rag.py
```

2. Open your web browser and navigate to `http://localhost:8501`

## Using the Application

### 1. Initialize and Check Status
- When the application starts, it will automatically check the Elasticsearch connection and ELSER model status
- View the initialization details by expanding the "View Elasticsearch Setup Details" section

### 2. Ingest PDF Documents
1. Click on "Choose a PDF file" under the "PDF Ingestion" section
2. Select a PDF file from your computer
3. Click "Ingest PDF" to process and index the document
4. Monitor the progress bar for indexing status

### 3. Ask Questions
1. Enter your question in the text input field under "Ask a Question"
2. Wait for the system to process your query
3. View the generated answer and supporting context
4. Explore retrieved documents by expanding the context sections, which show:
   - Document title and page number
   - Relevance score
   - Highlighted relevant excerpts
   - Full content of the retrieved sections

## Notes

- Ensure your Elasticsearch instance has the ELSER model properly deployed
- The application is configured to use IBM's Granite-3-8b-instruct model
- PDF chunks are limited to 512 tokens to match ELSER's requirements
- For development purposes, SSL certificate verification is disabled (`verify_certs=False`)

## Troubleshooting

If you encounter any issues:

1. Check the Elasticsearch connection details
2. Verify IBM Watson credentials
3. Ensure ELSER model is properly deployed in Elasticsearch
4. Check the console for any error messages
5. Verify all required packages are installed

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.