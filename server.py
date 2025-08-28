from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import os
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager

from model import (
        build_database,
        ask_questions,
        ask_numerical_questions,
        query_database,
        CHROMA
)

rag_system_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system_ready
    try:
        if os.path.exists(CHROMA):
            rag_system_ready = True
            print("RAG system is ready - database found.")
        else:
            print("RAG system starting - database not found, run /database/build first")
        yield
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
    finally:
        print("Shutting down server...")

app = FastAPI(
        title = "RAG API server",
        description = "REST API for Retrieval Augmented Generation System",
        version="1.0.0",
        lifespan = lifespan
)

class QuestionRequest(BaseModel):
    question: str
    type: Optional[str] = "standard"

class BatchQuestionRequest(BaseModel):
    questions: List[str]

class SearchRequest(BaseModel):
    query: str

class QuestionResponse(BaseModel):
    success: bool
    question: str
    type: str
    result: dict
    timestamp: str
    error: Optional[str] = None

class BatchResponse(BaseModel):
    success: bool
    total_questions: int
    results: List[dict]
    timestamp: str

class DatabaseStatus(BaseModel):
    database_exists: bool
    database_path: str
    stats: Optional[dict] = None


@app.get('/health')
async def health_check():
    return {
            "status":"OK",
            "timestamp": datetime.now().isoformat(),
            "service": "RAG API Server",
            "rag_system_ready": rag_system_ready
    }

@app.get('/database/status', response_model=DatabaseStatus)
async def get_database_status():
    try:
        exists = os.path.exists(CHROMA)
        stats = None

        if exists:
            stat = os.stat(CHROMA)
            stats = {
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "size": stat.st_size
            }

        return DatabaseStatus(
                database_exists = exists,
                database_path = CHROMA,
                stats = stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check database status: {str(e)}")

@app.post('/database/build')
async def build_vector_database():
    global rag_system_ready
    try:
        print("Building database...")
        build_database()
        rag_system_ready = True
        return {
                "success": True,
                "message":"Database built successfully",
                "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Failed to build database: {str(e)}")

@app.post('/ask', response_model = QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not rag_system_ready:
        raise HTTPException(status_code=503, detail='RAG system not ready. Build database first.')

    if not request.question.strip():
        raise HTTPException(status_code=400, detail = 'Question cannot be empty')

    try:
        print(f"Processing question: {request.question}")
        if request.type == 'numerical':
            result = ask_numerical_question(request.question)
            parsed_res = {"raw_output": result}
        elif request.type == 'query':
            res = query_database(request.question)
            parsed_res = {"raw_output": res}
        else:
            res = await ask_questions(request.question)
            try:
                if isinstance(res, str):
                    lines = res.split('\n')
                else:
                    lines = str(res).split('\n')

                answer = ''
                sources = ''
                excerpts = ''
                current_section = ''

                for line in lines:
                    if line.startswith("ANSWER:"):
                        current_section = 'answer'
                        continue
                    elif line.startswith("SOURCES:"):
                        current_section = 'sources'
                        continue
                    elif line.startswith("RELEVANT EXCERPTS:"):
                        current_section = 'excerpts'
                        continue

                    if current_section == 'answer':
                        answer += line + '\n'
                    elif current_section == 'sources':
                        sources += line + '\n'
                    elif current_section == 'excerpts':
                        excerpts += line + '\n'

                parsed_res = {
                        "answer": answer.strip(),
                        "sources": sources.strip(),
                        "excerpts": excerpts.strip()
                }
            except:
                parsed_res = {"raw_output": res}
        return QuestionResponse(
                success = True,
                question = request.question,
                type = request.type,
                result = parsed_res,
                timestamp = datetime.now().isoformat()
        )
    except Exception as e:
        return QuestionResponse(
                success = False,
                question = request.question,
                type = request.type,
                result = {},
                timestamp = datetime.now().isoformat(),
                error = str(e)
        )

@app.post('/ask/batch', response_model = BatchResponse)
async def ask_batch_questions(request: BatchQuestionRequest):
    if not rag_system_ready:
        raise HTTPException(status_code=500, detail = "RAG system not ready. Build database first.")

    if not request.questions or len(request.questions) == 0:
        raise HTTPException(status_code = 400, detail = "Questions array cannot be empty.")

    if len(request.questions) > 10:
        raise HTTPException(status_code = 400, detail = 'Max 10 questions allowed per batch')

    res = []
    for i, question in enumerate(request.questions):
        try:
            print(f"Processing batch question {i+1}/{len(request.questions)}: {question}")
            res.append({
                "question": question,
                "success": True,
                "result": res
            })
        except Exception as e:
            res.append({
                "question": {},
                "success": False,
                "error": str(e)
            })

    return BatchResponse(
            success = True,
            total_questions = len(request.questions),
            results = res,
            timestamp = datetime.now().isoformat()
    )

@app.post('/search')
async def search_documents(request: SearchRequest):
    if not rag_system_ready:
        raise HTTPException(status_code = 500, detail = "RAG system not ready. Build database first.")

    if not request.query.strip():
        raise HTTPException(status_code = 400, detail = "Query cannot be empty.")

    try:
        print(f"Searching for: {request.query}")
        res = query_database(request.query)

        return {
                "success": True,
                "query": request.query,
                "results": res,
                "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Search failed: {str(e)}")

@app.get('/')
async def root():
    return {
            "message": "RAG API Server",
            "version": "1.0.0",
            "endpoints": {
                "GET /health": "Check health",
                "GET /database/status": "Check database status",
                "POST /database/build": "Build vector database",
                "POST /ask": "Ask questions with sources",
                "POST /ask/batch": "Ask multiple questions",
                "POST /search" : "Search similar documents",
                "GET /docs": "API documentation (Swagger UI)"
            }
    }

if __name__ == "__main__":
    print("Starting RAG API server...")
    print("Access API docs at -> http://localhost:8000/docs")

    uvicorn.run(
            "server:app",
            host='0.0.0.0',
            port = 8000,
            reload = True,
            log_level = "info"
    )
