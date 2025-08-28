# RAG API Server

A **Retrieval-Augmented Generation (RAG) system** for querying the U.S. Department of Energy FY 2025 budget using PDF documents. 
Built with **LangChain**, **Chroma**, and **FastAPI**, it allows asking questions and retrieving numerical or contextual answers from documents.


## 📄 Dataset DOE FY 2025 Budget in Brief

This project utilizes the U.S. Department of Energy's **Fiscal Year 2025 Budget in Brief** document as its primary data source. 
The dataset is a publicly available PDF that outlines the DOE's budgetary allocations, priorities, and justifications for the fiscal year 2025.

### 📘 Document Overview

*   **Title:** DOE FY 2025 Budget in Brief
    
*   **Publisher:** U.S. Department of Energy
    
*   **Access:** [https://www.energy.gov/sites/default/files/2024-03/doe-fy-2025-budget-in-brief-v2.pdf](https://www.energy.gov/sites/default/files/2024-03/doe-fy-2025-budget-in-brief-v2.pdf)
    
*   **Format:** PDF
    
*   **Size:** Approximately 100 pages
    
*   **Content:** Detailed breakdown of the DOE's budget, including appropriations by program, office, and state.

### 📊 Key Sections
    
*   The document is structured into several key sections, each providing insights into different aspects of the DOE's budget:
    
*   **Appropriation Summary:** An overview of budget allocations by major program areas.
    
*   **Comparative Organization Summary:** Year-over-year comparisons of funding levels across various offices and programs.
    
*   **State Tables:** Allocations and funding details specific to each U.S. state.
    
*   **Budget Justifications:** Narrative explanations supporting the proposed budget figures.


### 🔍 Usage in the RAG API

The PDF document serves as the foundational dataset for this Retrieval-Augmented Generation (RAG) system. 
The system processes the document to extract relevant information, enabling users to query specific budgetary details through the API. 
For example, users can inquire about the total budget for a particular department or the funding allocated to specific programs within the DOE.
## Features

- Build and store a vector database from PDF documents.
- Ask questions in natural language and get answers with sources and excerpts.
- Support for numerical, standard, and batch queries.
- FastAPI REST API for integration with Postman, front-end, or other services.

### Limitations

*   **Data Format:** The dataset is in PDF format, which may require preprocessing to extract structured data effectively.
    
*   **Coverage:** The system's responses are limited to the information contained within the provided document.
    
*   **Updates:** The dataset corresponds to the FY 2025 budget and does not include information from subsequent fiscal years.

## Project Structure

RAG/

├── dataset/

│ └── usa-2025-budget-brief-energy-dep-v2.pdf

├── chroma/ # Vector database

├── model.py # RAG logic and document processing

├── server.py # FastAPI server

├── requirements.txt # Python dependencies

└── README.md


---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/0quaaD/Fiscal-Year-2025-Budget-Justification-RAG-Model.git rag-api-model
cd rag-api-model
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate     # Windows
```
3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Add your PDF documents to the dataset/ folder.


Building the Database
---------------------

Before using the API, build the vector database from the PDFs:
```bash
python server.py --build # Or python3 depends on your python
```
Or via API:
```bash
POST /database/build
```
This will create the chroma/ folder for storing embeddings.

## Running the API Server
```bash
python server.py --build
```
**TODO** --> Access API docs (Swagger UI) at: [/docs](http://localhost:8000/docs)

## API Endpoints
| Endpoint           | Method | Description                                                 |
| ------------------ | ------ | ----------------------------------------------------------- |
| `/health`          | GET    | Check server health                                         |
| `/database/status` | GET    | Check database status                                       |
| `/database/build`  | POST   | Build vector database                                       |
| `/ask`             | POST   | Ask a single question (supports `standard` and `numerical`) |
| `/ask/batch`       | POST   | Ask multiple questions (max 10)                             |
| `/search`          | POST   | Search similar documents without LLM                        |
| `/docs`            | GET    | Documentation of API                                        |
