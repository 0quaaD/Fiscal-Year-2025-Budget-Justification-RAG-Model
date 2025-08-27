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

DATA = 'dataset/usa-2025-budget-brief-energy-dep-v2.pdf'
CHROMA = 'chroma'
embeddings = HuggingFaceEmbeddings(model = 'sentence-transformers/all-MiniLM-L6-v2')

async def load_data():
    loader = PyPDFLoader(DATA)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

#print(f"{pages[0].metadata}\n")
#print(pages[0].page_content)

def __text_splitter__(docs: list[Document]):
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
        print(document.page_content)
        print(document.metadata)
    else:
        print("Not enough chunks to print chunk 10.")
    
    return chunks

def save_at_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA):
        shutil.rmtree(CHROMA)
    
    db = Chroma(
            embedding_function = embeddings,
            persist_directory = CHROMA
    )

    db.add_documents(chunks)

    print(f"Saved {len(chunks)} chunks at {CHROMA}")

def ask_questions(query_text):
    if not os.path.exists(CHROMA):
        return "Database not found! Please run first '--build' option."

    db = Chroma(persist_directory = CHROMA, embedding_function = embeddings)
    retriever = db.as_retriever(search_kwargs={'k':3})
    llm = ChatOllama(model = 'llama3.2:3b', temperature = 0.1)

    prompt = ChatPromptTemplate.from_template(
            "Answer only based on the context below. If the answer is not in the context, say 'Not found in document.'"
    )

    chain = (
            {'context': retriever, 'question': RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    
    response = chain.invoke(query_text)
    print(response)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Build the vector database')
    parser.add_argument('--ask', type=str, help='Ask a question (with LLM answer)')
    args = parser.parse_args()
    question = args.ask
    if args.build:
        pages = asyncio.run(load_data())
        text_splitted = __text_splitter__(pages)
        save_at_chroma(text_splitted)
        print('Database created successfully!')
    elif args.ask:
        ask_questions(question)
    else:
        print("Please specify either --build to create database or --ask to ask a question")

def query_database(query_text):
    db = Chroma(persist_directory = CHROMA, embedding_function = embedding)

    res = db.similarity_search(query_text, k=3)
    if len(res) == 0:
        print('Unable to find any matches!')
        return
    context_text = "\n\n---\n\n".join([docs.page_content for docs in res])
    print(context_text)

if __name__ == '__main__':
    main()
