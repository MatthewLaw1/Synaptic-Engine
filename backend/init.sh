#!/bin/bash

python3 -m venv venv
source venv/bin/activate

pip install langchain
pip install openai
pip install python-dotenv
pip install chromadb
pip install tiktoken
pip install faiss-cpu
pip install unstructured
pip install pdf2image
pip install pytesseract
pip install google-search-results

chmod +x init.sh

echo "Dependencies installed successfully!"