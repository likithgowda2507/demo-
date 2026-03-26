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     

To run this project on a different system, you'll need the source code, the required Python packages, and your .env file with your Groq API key.

Here are the step-by-step instructions and the code you need to set this up on any other computer:

1. File Structure
First, ensure you copy the entire 

llm
 folder to the new system. It should look like this:

text
llm/
├── .env                  # Contains your GROQ_API_KEY
├── requirements.txt      # List of Python dependencies
├── streamlit_app.py      # The main Streamlit UI
├── src/
│   └── rag_pipeline.py   # The core RAG logic using LangChain & FAISS
└── pdfs/                 # Folder containing all your SOP PDF files
2. The requirements.txt File
Make sure your requirements.txt has the following dependencies. If you don't have this file, create one in the 

llm
 folder:

text
streamlit
langchain
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
pypdf
requests
python-dotenv
3. The .env File
You must create a .env file in the root 

llm
 folder with your Groq API credentials. It should look exactly like this:

text
GROQ_API_KEY=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
GROQ_MODEL=llama-3.3-70b-versatile
LLM_PROVIDER=groq
GROQ_BASE_URL=https://api.groq.com/openai/v1
4. Setup Instructions (Terminal / Command Prompt)
Run these exact commands on the new system to get the app running:

Step A: Navigate to the folder Open a terminal and navigate to the project directory:

bash
cd path/to/your/llm/folder
Step B: Create a Virtual Environment (Optional but Recommended) It's best practice to use a virtual environment so you don't conflict with other Python projects:

bash
# Create the environment
python3 -m venv venv
# Activate it (Mac/Linux):
source venv/bin/activate
# Activate it (Windows):
venv\Scripts\activate
Step C: Install the Dependencies Run this command to install all the necessary libraries:

bash
pip install -r requirements.txt
Step D: Run the Application Finally, start the Streamlit server:

bash
streamlit run streamlit_app.py
Note on the First Run
When you run the app on the new system for the very first time, click the "Build / Refresh Index" button in the sidebar. This will tell FAISS to read all the PDFs in the pdfs/ folder and generate the vector embeddings database locally on that new computer.
