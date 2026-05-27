"""
AGENTS: LLM Chains for Research Writer and Critic
Uses Vertex AI Gemini models (non-deprecated)
"""

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Vertex AI Configuration
# Note: For Vertex AI, you need:
# 1. GOOGLE_APPLICATION_CREDENTIALS pointing to service account JSON
# 2. Proper authentication setup

# Available models in Vertex AI:
# - gemini-2.5-flash (fast, good for most tasks)
# - gemini-2.0-flash (balanced)
# - gemini-1.5-flash (older but stable)
# - gemini-1.5-pro (higher quality, slower)
# - gemini-2.0-pro-exp (experimental)

def get_llm(model_name: str = None, temperature: float = 0.5, location: str = None):
    """
    Create Vertex AI LLM instance
    
    Args:
        model_name: Model name (default from env or gemini-2.5-flash)
        temperature: Temperature for generation (0.0-1.0)
        location: GCP location (default from env or us-central1)
    
    Returns:
        ChatVertexAI instance
    """
    model = model_name or os.getenv("VERTEX_MODEL_NAME", "gemini-2.5-flash")
    temp = float(os.getenv("VERTEX_TEMPERATURE", temperature))
    loc = location or os.getenv("VERTEX_LOCATION", "us-central1")
    
    # Project ID - required for Vertex AI
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    if not project:
        print("[WARN] GOOGLE_CLOUD_PROJECT not set in .env")
        print("  Using default project from credentials")
    
    llm = ChatVertexAI(
        model_name=model,
        temperature=temp,
        location=loc,
        project=project,  # Optional if set in credentials
        max_output_tokens=8192,  # Gemini supports up to 8192
        top_p=0.95,
        top_k=40,
    )
    
    return llm

# ============================================================================
# DEFAULT LLM INSTANCE
# ============================================================================

# Create main LLM for writer and critic
llm = get_llm()

# Optional: Different models for different tasks
# For better quality reports, use Pro model
writer_llm = get_llm(model_name=os.getenv("WRITER_MODEL", "gemini-2.5-flash"), temperature=0.5)

# For critic, lower temperature for more consistent evaluation
critic_llm = get_llm(temperature=0.3)

# ============================================================================
# WRITER CHAIN
# ============================================================================

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research writer for a prestigious institution. 
Write clear, structured, and insightful reports with proper factual grounding.

Guidelines:
- Use professional, academic tone
- Cite information from research provided
- Be objective and balanced
- Avoid unsupported claims
- Write in markdown format for readability"""),

    ("human", """Write a detailed research report on the topic below.

TOPIC: {topic}

RESEARCH GATHERED:
{research}

STRUCTURE THE REPORT AS:
## Introduction
- Hook the reader
- State the importance of the topic
- Preview key findings

## Key Findings
(Minimum 3 well-explained points, each with evidence)

### Finding 1: [Title]
- Detailed explanation
- Supporting evidence from research
- Real-world implications

### Finding 2: [Title]
- Detailed explanation
- Supporting evidence from research
- Real-world implications

### Finding 3: [Title]
- Detailed explanation
- Supporting evidence from research
- Real-world implications

## Conclusion
- Summarize key insights
- Discuss limitations (if any)
- Suggest future research directions

IMPORTANT: 
- Do NOT include a sources section (will be added separately)
- Write detailed, factual, and professional content
- Each finding should be 2-3 paragraphs minimum"""),
])

# Using writer_llm (Pro model for better quality)
writer_chain = writer_prompt | writer_llm | StrOutputParser()

# ============================================================================
# CRITIC CHAIN
# ============================================================================

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a sharp, constructive research critic. Be honest, specific, and actionable.

Evaluation Criteria:
1. Factual Accuracy: Does the report align with provided research?
2. Structure: Is the report well-organized?
3. Completeness: Are key aspects covered?
4. Clarity: Is the writing clear and accessible?
5. Professionalism: Is the tone appropriate?

Be critical but fair. Focus on improvement."""),

    ("human", """Review the research report below and evaluate it strictly.

REPORT:
{report}

RESPOND IN THIS EXACT FORMAT:

## Score: X/10

## Strengths:
- [Specific strength 1 with example]
- [Specific strength 2 with example]
- [Specific strength 3 with example]

## Areas to Improve:
- [Specific issue 1 with suggestion]
- [Specific issue 2 with suggestion]
- [Specific issue 3 with suggestion]

## Hallucination Check:
- [Any unsupported claims? If yes, list them]
- [Any missing citations? If yes, note them]

## One Line Verdict:
[Summary sentence]

Be honest and specific. Provide actionable feedback."""),
])

# Using critic_llm (lower temperature for consistency)
critic_chain = critic_prompt | critic_llm | StrOutputParser()

# ============================================================================
# OPTIONAL: FAST VERSIONS FOR QUICK RESPONSES
# ============================================================================

# Fast writer (Flash model for speed)
fast_llm = get_llm(model_name="gemini-2.5-flash", temperature=0.5)

fast_writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research writer. Write concise, factual reports."),
    ("human", "Write a brief research report on: {topic}\n\nResearch: {research}"),
])

fast_writer_chain = fast_writer_prompt | fast_llm | StrOutputParser()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_report(topic: str, research: str, use_fast: bool = False) -> str:
    """
    Generate research report
    
    Args:
        topic: Research topic
        research: Research context
        use_fast: Use faster model (lower quality)
    
    Returns:
        Generated report
    """
    if use_fast:
        return fast_writer_chain.invoke({"topic": topic, "research": research})
    return writer_chain.invoke({"topic": topic, "research": research})

def generate_feedback(report: str) -> str:
    """
    Generate feedback for report
    
    Args:
        report: Report to review
    
    Returns:
        Feedback with score and suggestions
    """
    return critic_chain.invoke({"report": report})

def get_model_info() -> dict:
    """
    Get current model configuration
    
    Returns:
        Dictionary with model info
    """
    return {
        "writer_model": os.getenv("WRITER_MODEL", "gemini-2.5-flash"),
        "critic_model": os.getenv("VERTEX_MODEL_NAME", "gemini-2.5-flash"),
        "temperature": float(os.getenv("VERTEX_TEMPERATURE", 0.5)),
        "location": os.getenv("VERTEX_LOCATION", "us-central1"),
        "project": os.getenv("GOOGLE_CLOUD_PROJECT", "default"),
    }

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AGENTS: Vertex AI Configuration Test")
    print("="*70)
    
    # Show configuration
    print("\n[CONFIGURATION]")
    config = get_model_info()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test writer
    print("\n[TEST 1] Writer Chain")
    print("-" * 70)
    
    test_topic = "Impact of AI on Software Development"
    test_research = """
    AI tools like GitHub Copilot and Cursor are transforming software development.
    Studies show 40-60% productivity gains for routine coding tasks.
    Developers report spending less time on boilerplate code and more on architecture.
    Challenges include code quality concerns and security considerations.
    """
    
    print(f"Topic: {test_topic}")
    print("Generating report...\n")
    
    try:
        report = generate_report(test_topic, test_research)
        print(report[:500] + "...\n" if len(report) > 500 else report)
        print(f"\n[OK] Report generated: {len(report)} characters")
    except Exception as e:
        print(f"[ERROR] Writer failed: {e}")
    
    # Test critic
    print("\n[TEST 2] Critic Chain")
    print("-" * 70)
    
    test_report = """
    ## Introduction
    AI is changing software development significantly.
    
    ## Key Findings
    AI tools improve developer productivity by 40-60%.
    Developers can focus on higher-level architecture.
    """
    
    print("Generating feedback...\n")
    
    try:
        feedback = generate_feedback(test_report)
        print(feedback[:500] + "..." if len(feedback) > 500 else feedback)
        print(f"\n[OK] Feedback generated: {len(feedback)} characters")
    except Exception as e:
        print(f"[ERROR] Critic failed: {e}")
    
    # Test fast writer
    print("\n[TEST 3] Fast Writer (Flash Model)")
    print("-" * 70)
    
    try:
        fast_report = generate_report(test_topic, test_research, use_fast=True)
        print(f"[OK] Fast report: {len(fast_report)} characters")
    except Exception as e:
        print(f"[ERROR] Fast writer failed: {e}")
    
    print("\n" + "="*70)
    print("[OK] Vertex AI Agents configured successfully!")
    print("="*70)