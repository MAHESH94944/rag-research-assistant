"""
FastAPI Backend for RAG Research Assistant
Provides REST API endpoints for research queries
"""

import os
import json
import tempfile

# ============================================================================
# CRITICAL: Handle Google Credentials for Render
# MUST be at the top before any other imports that use Vertex AI
# ============================================================================

def setup_google_credentials():
    """Setup Google credentials from environment variable for Render deployment"""
    
    # Check if we're on Render (has RENDER environment variable)
    is_render = os.getenv('RENDER', 'false').lower() == 'true'
    
    if is_render:
        print("[INFO] Running on Render - Setting up credentials from env var")
        
        # Try to get credentials from JSON env var
        creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        
        if creds_json:
            try:
                # Parse and validate JSON
                creds = json.loads(creds_json)
                
                # Write to temporary file
                temp_creds_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(creds, temp_creds_file)
                temp_creds_file.close()
                
                # Set environment variable to point to temp file
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_file.name
                print(f"[OK] Google credentials loaded from env var")
                
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
        else:
            # Fall back to file path
            creds_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'dev-google-credentials.json')
            if os.path.exists(creds_file):
                print(f"[OK] Using credentials file: {creds_file}")
            else:
                print(f"[WARN] No credentials found! Looking for: {creds_file}")
    else:
        print("[INFO] Running locally - using existing credentials")
    
    # Set project ID
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if project_id:
        print(f"[INFO] Using project: {project_id}")

# Run credential setup BEFORE any other imports
setup_google_credentials()

# Now import the rest of your application
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
# SECURITY (Add API Key Authentication)
# ============================================================================

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for protected endpoints"""
    api_key = credentials.credentials
    expected_key = os.getenv('API_SECRET_KEY')
    
    # Skip auth in development or if no key is set
    if not expected_key or os.getenv('ENVIRONMENT') != 'production':
        return api_key
    
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    return api_key

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

# Get CORS origins from environment
cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
if '*' in cors_origins and os.getenv('ENVIRONMENT') == 'production':
    # Don't allow * in production - override with specific origins
    cors_origins = ['https://your-frontend-domain.com']

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for tracking requests
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
        "environment": os.getenv('ENVIRONMENT', 'development'),
        "endpoints": {
            "research": "/research (POST) - Requires API Key",
            "research_stream": "/research/stream (POST) - Requires API Key",
            "health": "/health (GET) - Public",
            "memory_query": "/memory/query (POST) - Requires API Key",
            "memory_stats": "/memory/stats (GET)",
            "model_info": "/model/info (GET)"
        },
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint (public - no auth required)"""
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
            "/research (POST) - Requires API Key",
            "/research/stream (POST) - Requires API Key",
            "/memory/query (POST) - Requires API Key"
        ]
    }

@app.post("/research", response_model=ResearchResponse, tags=["Research"])
async def research(
    request: ResearchRequest, 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)  # Add API key auth
):
    """
    Execute research on a topic and return report
    
    - **topic**: Research question or topic
    - **use_advanced**: Use advanced system (LangGraph, streaming, evaluation)
    - **use_memory**: Enable memory for follow-up questions
    - **return_feedback**: Include critic feedback in response
    
    Requires API Key in Authorization header: Bearer <your-api-key>
    """
    request_id = generate_request_id()
    
    try:
        # Your existing research logic (same as before)
        if request.use_advanced:
            result = quick_research(request.topic, full_pipeline=True)
            report = result.get("workflow_state", {}).get("report", "")
            sources = result.get("workflow_state", {}).get("sources", [])
            feedback = result.get("evaluation", None)
            
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
        
        request_store[request_id] = {
            "topic": request.topic,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        return ResearchResponse(
            request_id=request_id,
            topic=request.topic,
            report=report,
            feedback=feedback_text if request.return_feedback else None,
            sources=sources[:10],
            metadata=metadata,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ... rest of your endpoints (research_stream, memory_query, etc.)
# Keep them the same but add Depends(verify_api_key) to protected ones

# ============================================================================
# RUN SERVER (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 STARTING RAG RESEARCH API SERVER")
    print("="*70)
    print(f"\n📍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("📍 Health Check: http://localhost:8000/health")
    print("\n📝 Protected Endpoints (require API Key):")
    print("   POST /research - Research a topic")
    print("   POST /research/stream - Stream research report")
    print("   POST /memory/query - Follow-up question with memory")
    print("\n⏹️  Press Ctrl+C to stop server\n")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv('ENVIRONMENT') != 'production',
        log_level=os.getenv('LOG_LEVEL', 'info')
    )