# Intelligent_Complaint_Analysis
This project involves developing an intelligent complaint-answering chatbot that empowers product, support, and compliance teams to understand customer pain points across five major product categories quickly: Credit Cards Personal Loans Buy Now, Pay Later (BNPL) Savings Accounts Money Transfers


  🧾 Intelligent Complaint Analysis for Financial Services  
  RAG-Powered Chatbot for CrediTrust Financial  

This project implements a   Retrieval-Augmented Generation (RAG)   system to help internal stakeholders at   CrediTrust Financial   quickly identify trends and insights from unstructured customer complaint data.

The chatbot allows non-technical users to ask questions like:
- "Why are people unhappy with BNPL services?"
- "Are there any fraud-related complaints in money transfers?"
- "How do customers feel about interest charges on credit cards?"

And receive synthesized, evidence-backed answers using semantic search over real customer feedback data.

   🚀 Features

✅ Cleaned and filtered dataset of customer complaints  
✅ Text chunking and embedding using `all-MiniLM-L6-v2`  
✅ FAISS vector store for fast similarity search  
✅ RAG pipeline using TinyLlama for grounded answer generation  
✅ Interactive Gradio chat interface with source visibility  
✅ Evaluation table showing quality scores and retrieved sources  






   📁 Folder Structure

project_root/
├── data/
│   └── filtered_complaints.csv
├── vector_store/
│   ├── faiss_index.bin
│   └── chunk_metadata.pkl
├── src/
│   ├── rag_pipeline.py
│   ├── preprocess.py
│   ├── chunking_embedding.py
│   └── evaluate_rag.py
├── notebooks/
│   └── eda_preprocessing.ipynb
├── app.py
├── requirements.txt
└── README.md

   🛠️ Setup Instructions

    1. Clone the Repository

bash
git clone https://github.com/yourusername/intelligent-complaint-analysis.git
cd intelligent-complaint-analysis



    2. Create Virtual Environment (Optional but Recommended)

bash
python -m venv .venv
source .venv/bin/activate     On Windows: .venv\Scripts\activate

    3. Install Dependencies
bash
pip install -r requirements.txt

   🧪 Run the App

Start the interactive chat interface:
bash
python app.py

Then open your browser at:

👉 http://127.0.0.1:7860

Ask natural-language questions about customer complaints and see the RAG system retrieve relevant narratives and generate concise answers.



   🧩 Key Components

    1. Data Preprocessing

- Loads and filters the [CFPB Consumer Complaint Dataset](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- Cleans text and prepares it for embedding
- Saved as `data/filtered_complaints.csv`

    2. Chunking & Embedding

- Splits long narratives into smaller chunks using `RecursiveCharacterTextSplitter`
- Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic embeddings
- Stores them in a FAISS index (`vector_store/faiss_index.bin`) with metadata

    3. RAG Pipeline

- Takes user input and encodes the query
- Retrieves top-k similar complaint chunks from FAISS
- Feeds context + question to an LLM (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- Returns a generated answer along with source documents used

    4. Web Interface

Built with   Gradio  , allowing users to interact with the system via a web dashboard:
- Input box for natural language queries
- Submit button
- Answer display
- Source document visibility
- Clear conversation button

   📊 Evaluation Results

| Question | Generated Answer | Retrieved Sources | Quality Score (1–5) | Comments |
|---------|------------------|-------------------|--------------------|----------|
| Why are people unhappy with BNPL services? | Some customers report unexpected fees, unclear repayment terms, and unauthorized charges... | 1. "I was charged recurring subscription..."<br>2. "The company resumed monthly charges..." | 4 | Good coverage but lacks depth |
| Are there any fraud-related complaints in money transfers? | Yes, several users reported unauthorized transactions and identity theft... | 1. "A payment was received from an unknown person..."<br>2. "Someone cashed my check but their system doesn’t have info..." | 5 | Accurate and well-supported |

> Full evaluation results are available in `Evaluation_table.md`
   🧠 Learning Outcomes

By completing this project, we achieved the following learning goals:

| Outcome | Description |
|--------|-------------|
| ✅ Semantic Search + LLM Integration | Combined FAISS with TinyLlama to generate grounded answers |
| ✅ Noisy Text Handling | Cleaned raw complaints for better embedding quality |
| ✅ Vector Store Indexing | Built and queried a FAISS index with metadata |
| ✅ Chatbot Development | Created a chat interface with sources shown |
| ✅ Real-Time Feedback System | Enabled non-technical teams to get instant insights |




   🧱 Technologies Used

| Component | Tool / Library |
|----------|----------------|
| RAG Core Logic | LangChain |
| Embeddings | Sentence Transformers |
| Vector Store | FAISS |
| LLM | TinyLlama / Mistral |
| UI Framework | Gradio |
| Data Format | CSV, Pickle, FAISS Binary |
| Tokenizer | Hugging Face Transformers |

   🧩 Challenges Faced

| Challenge | Solution |
|----------|----------|
| ❌ Mistral access denied | Switched to TinyLlama for development |
| ⏳ Long prompts exceeded token limits | Limited retrieved chunks and truncated inputs |
| 🔄 Gradio chatbot formatting errors | Fixed message structure and output format |
| 📁 Missing dependencies in virtual env | Installed Gradio, sentence-transformers, etc. |
| 🧠 Hallucination risk with smaller models | Ensured strict context-only generation |
| 🧪 Evaluation scoring | Developed a 5-point quality scale with analysis |

   📌 Future Enhancements

| Enhancement | Description |
|------------|-------------|
| 🧠 Upgrade to Mistral-7B or Llama3 | For higher-quality responses |
| 🧮 Add Streaming Response | Improve UX with token-by-token generation |
| 📥 Real-Time Ingestion | Automatically update the vector store with new complaints |
| 📋 Export Chat History | Allow download of conversations and sources |
| 📊 Dashboard View | Visualize complaint trends over time |
| 🧰 Integrate with Slack/Teams | Bring insights directly to business teams |


