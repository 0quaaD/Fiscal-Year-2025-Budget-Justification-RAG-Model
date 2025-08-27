import asyncio
import os
import shutil
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
from pydantic import BaseModel, Field
from langchain.chains import RetrievalQAWithSourcesChain

DATA = 'dataset/usa-2025-budget-brief-energy-dep-v2.pdf'
CHROMA = 'chroma'
embeddings = HuggingFaceEmbeddings(model = 'sentence-transformers/all-MiniLM-L6-v2')

class BudgetProgram(BaseModel):
    name: str = Field(..., description = 'Program name')
    office: str = Field(..., description = 'Office under DOE')
    funding_request: Optional[float] = Field(None, description='Funding in billions USD')

async def load_data():
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"PDF file not found: {DATA}")

    loader = PyPDFLoader(DATA)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


def text_splitter(docs: list[Document]):
    if not docs:
        raise ValueError("No documents are provided for splitting.")

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, 
            chunk_overlap = 500,
            length_function = len,
            is_separator_regex = False,
            add_start_index = True
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    if len(chunks) > 10:
        document = chunks[10]
        print("\nSample chunk (10):")
        print(f"Content Review: {document.page_content[:200]}...")
        print(f"Metadata Review: {document.metadata}")
    else:
        print("Not enough chunks to print chunk 10.")
    
    return chunks

def save_at_chroma(chunks: list[Document]):
    if not chunks:
        raise ValueError("No chunks provided for saving at Chroma")

    if os.path.exists(CHROMA):
        shutil.rmtree(CHROMA)
    
    db = Chroma(
            embedding_function = embeddings,
            persist_directory = CHROMA
    )

    batch_size = 100
    for i in range(0,len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        db.add_documents(batch)
        print(f"Added batch: {i//batch_size + 1}: {len(batch)} chunks.")

    print(f"Successfully saved {len(chunks)} chunks at {CHROMA}")

def ask_questions(query_text):

    if not query_text.strip():
        return "Please provide a valid question."
    
    if not os.path.exists(CHROMA):
        return "Database not found! Please run with '--build' option first."

    try:
        # Initialize components
        db = Chroma(persist_directory=CHROMA, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={'k': 20})  # Match your k=20
        llm = ChatOllama(model='llama3.2:3b', temperature=0.1)

        # Create the chain - CORRECT way
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Execute the chain - CORRECT way
        response = qa_chain.invoke({"question": query_text})
        
        # Format the response nicely
        answer = response.get('answer', 'No answer found')
        sources = response.get('sources', 'No sources')
        source_docs = response.get('source_documents', [])
        
        formatted_response = f"""
ANSWER:
{answer}

SOURCES:
{sources}

RELEVANT EXCERPTS:
"""
        formatted_response_ = """

You are an assistant that extracts **specific funding amounts** from the context.  
Question: {question}  
If you find the funding request, respond ONLY with the number and its unit (e.g., "$12 million").  
If no number is present, say: "Not found in the provided context."  
Context: {context}
"""        
        # Add relevant excerpts for verification
        for i, doc in enumerate(source_docs[:3]):  # Show top 3 source documents
            formatted_response += f"\n--- Source {i+1} ({doc.metadata.get('source', 'Unknown')}) ---\n"
            formatted_response += doc.page_content[:400] + "...\n"
        
        return formatted_response
    
    except Exception as e:
        return f"Error occurred while processing question: {str(e)}"
def main():
    parser = argparse.ArgumentParser(description="RAG system for Fiscal year 2025 justification of Department of Energy Q&A")
    parser.add_argument('--build', action='store_true', help='Build the vector database')
    parser.add_argument('--ask', type=str, help='Ask a question (with LLM answer)')
    parser.add_argument('--query', type=str, help='Query database for similar documents (no LLM)')

    args = parser.parse_args()
    
    if args.build:
        asyncio.run(build_database())
    elif args.ask:
        response =  ask_questions(args.ask)
        print(response) 
    elif args.query:
        query_database(args.query)
    else:
        print("Please specify either --build to create database or --ask to ask a question")

def query_database(query_text):
    if not query_text.strip():
        print("Please provide a valid query.")
        return
    
    if not os.path.exists(CHROMA):
        print("Database not found! Please run with '--build' first.")
        return

    try:
        db = Chroma(persist_directory = CHROMA, embedding_function = embedding)

        res = db.similarity_search(query_text, k=3)
        if len(res) == 0:
            print('Unable to find any matches!')
            return
        
        print(f"Found {len(res)} relevant documents.")
        context_text = "\n\n---\n\n".join([docs.page_content for docs in res])
        print(context_text)

    except Exception as e:
        print(f"Error occured while querying the database: {str(e)}")

async def build_database():
    try:
        pages = await load_data()
        text_chunks = text_splitter(pages)
        save_at_chroma(text_chunks)
    except Exception as e:
        print(f'Error occured while building the database: {str(e)}')
if __name__ == '__main__':
    main()
