# Intelligent Chatbot Application

This Streamlit-based chatbot application utilizes OpenAI's GPT model for generating responses and Pinecone's vector database for storing and retrieving document embeddings. It is designed to provide users with intelligent responses based on the content from indexed documents loaded into Pinecone.

## Features

- **Web-Based Document Loading**: Load documents via URLs to index their content in Pinecone.
- **Intelligent Response Generation**: Uses OpenAI's powerful models to generate context-aware responses.
- **Embedding Storage and Retrieval**: Leverages Pinecone's serverless vector database for efficient storage and retrieval of text embeddings.

## How to Set Up and Run the Application

### Prerequisites

- Python 3.8 or higher
- Access to OpenAI API
- Access to Pinecone API
- An internet connection for accessing documents and APIs

### Installation

1. **Clone the repository:**
   ```bash
   git clone [repository URL]
   cd [repository name]
   
 ```bash
python -m venv venv
source venv/bin/activate  # For Unix or MacOS
venv\Scripts\activate  # For Windows

pip install -r requirements.txt


OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

streamlit run app.py

