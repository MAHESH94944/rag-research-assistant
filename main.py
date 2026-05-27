"""
FastAPI Backend for RAG Research Assistant
Provides REST API endpoints for research queries
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import json
import uuid

# Import your existing system
from pipeline import run_research_pipeline
from advanced_system import quick_research, research_with_memory
from agents import get_model_info

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class ResearchRequest(BaseModel):
    """Request model for research endpoint"""
    topic: str = Field(..., description="Research topic/question", min_length=1, max_length=500)
    use_advanced: bool = Field(False, description="Use advanced system with all features")
    use_memory: bool = Field(False, description="Enable memory for follow-up questions")
    return_feedback: bool = Field(True, description="Include critic feedback in response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "How NVIDIA became one of the most valuable companies",
                "use_advanced": True,
                "use_memory": False,
                "return_feedback": True
            }
        }

class ResearchResponse(BaseModel):
    """Response model for research endpoint"""
    request_id: str
    topic: str
    report: str
    feedback: Optional[str] = None
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]
    generated_at: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_info: Dict[str, Any]
    timestamp: str

class MemoryQueryRequest(BaseModel):
    """Request for memory-based queries (follow-up questions)"""
    query: str = Field(..., description="Follow-up question")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Tell me more about CUDA technology"
            }
        }

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="RAG Research Assistant API",
    description="Multi-agent RAG system for automated research and report generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (for React/Frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for tracking requests (optional)
request_store: Dict[str, Dict] = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RAG Research Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "research": "/research (POST)",
            "research_stream": "/research/stream (POST)",
            "health": "/health (GET)",
            "memory_query": "/memory/query (POST)",
            "memory_stats": "/memory/stats (GET)",
            "model_info": "/model/info (GET)"
        },
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_info=get_model_info(),
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", tags=["Model"])
async def get_model_information():
    """Get current model configuration"""
    return {
        "model_info": get_model_info(),
        "available_endpoints": [
            "/research (POST)",
            "/research/stream (POST)",
            "/memory/query (POST)"
        ]
    }

@app.post("/research", response_model=ResearchResponse, tags=["Research"])
async def research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Execute research on a topic and return report
    
    - **topic**: Research question or topic
    - **use_advanced**: Use advanced system (LangGraph, streaming, evaluation)
    - **use_memory**: Enable memory for follow-up questions
    - **return_feedback**: Include critic feedback in response
    """
    request_id = generate_request_id()
    
    try:
        # Choose which system to use
        if request.use_advanced:
            # Use advanced system with all features
            result = quick_research(request.topic, full_pipeline=True)
            
            report = result.get("workflow_state", {}).get("report", "")
            sources = result.get("workflow_state", {}).get("sources", [])
            feedback = result.get("evaluation", None)
            
            # Format feedback if available
            feedback_text = None
            if request.return_feedback and feedback:
                feedback_text = f"""
Score: {feedback.overall_quality:.2%}
Faithfulness: {feedback.faithfulness:.2%}
Relevancy: {feedback.answer_relevancy:.2%}
Hallucination: {feedback.hallucination_score:.2%}
                """
            elif request.return_feedback and result.get("workflow_state", {}).get("feedback"):
                feedback_text = result["workflow_state"]["feedback"]
            
            metadata = {
                "system": "advanced",
                "has_memory": request.use_memory,
                "has_evaluation": feedback is not None,
                "request_id": request_id
            }
            
        else:
            # Use simple pipeline
            result = run_research_pipeline(request.topic)
            
            report = result.get("report", "")
            sources = result.get("sources", [])
            feedback = result.get("feedback", "") if request.return_feedback else None
            
            metadata = {
                "system": "simple",
                "has_memory": False,
                "has_evaluation": False,
                "search_results": len(result.get("search_results", [])),
                "scraped_sources": len(result.get("scraped_content", [])),
                "request_id": request_id
            }
        
        # Store request for tracking
        request_store[request_id] = {
            "topic": request.topic,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        return ResearchResponse(
            request_id=request_id,
            topic=request.topic,
            report=report,
            feedback=feedback,
            sources=sources[:10],  # Limit to 10 sources
            metadata=metadata,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/stream", tags=["Research"])
async def research_stream(request: ResearchRequest):
    """
    Stream research report as it's generated (Server-Sent Events)
    
    Returns streaming response with chunks as they're generated
    """
    async def generate_stream():
        try:
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'topic': request.topic})}\n\n"
            
            # Run the pipeline (simplified for streaming)
            # Note: Full streaming requires modification to pipeline
            result = run_research_pipeline(request.topic)
            
            # Send report in chunks
            report = result.get("report", "")
            chunk_size = 500
            
            for i in range(0, len(report), chunk_size):
                chunk = report[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for realism
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'sources': result.get('sources', [])})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/memory/query", tags=["Memory"])
async def memory_query(request: MemoryQueryRequest):
    """
    Query with memory enabled (for follow-up questions)
    Requires previous research in memory
    """
    try:
        result = research_with_memory(request.query)
        
        return {
            "query": request.query,
            "report": result.get("workflow_state", {}).get("report", ""),
            "sources": result.get("workflow_state", {}).get("sources", []),
            "memory_id": result.get("memory_id"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats", tags=["Memory"])
async def get_memory_statistics():
    """Get memory system statistics"""
    try:
        from memory_system import ResearchMemory
        memory = ResearchMemory()
        stats = memory.get_memory_stats()
        
        return {
            "status": "available",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/requests/{request_id}", tags=["Tracking"])
async def get_request_status(request_id: str):
    """Get status of a specific request"""
    if request_id in request_store:
        return request_store[request_id]
    raise HTTPException(status_code=404, detail="Request not found")

@app.get("/requests", tags=["Tracking"])
async def list_requests(limit: int = 50):
    """List recent requests"""
    requests = list(request_store.values())[-limit:]
    return {"total": len(request_store), "requests": requests}

# ============================================================================
# RUN SERVER (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 STARTING RAG RESEARCH API SERVER")
    print("="*70)
    print("\n📍 API Documentation: http://localhost:8000/docs")
    print("📍 ReDoc: http://localhost:8000/redoc")
    print("📍 Health Check: http://localhost:8000/health")
    print("\n📝 Available Endpoints:")
    print("   POST /research        - Research a topic")
    print("   POST /research/stream - Stream research report")
    print("   POST /memory/query    - Follow-up question with memory")
    print("   GET  /memory/stats    - Memory statistics")
    print("\n⏹️  Press Ctrl+C to stop server\n")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )