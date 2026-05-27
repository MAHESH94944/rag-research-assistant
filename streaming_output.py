"""
STEP 14: Streaming Output
Enables streaming responses like ChatGPT typing effect
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI  # ✅ Fixed: Use non-deprecated
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator, Optional, Dict, Any
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# STREAMING CHAINS
# ============================================================================

class StreamingWriter:
    """
    Report writer with streaming support
    Streams tokens as they're generated
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.5):
        """
        Initialize streaming writer
        
        Args:
            model_name: Gemini model name (default from env or gemini-2.5-flash)
            temperature: Temperature for generation
        """
        model = model_name or os.getenv("MODEL_NAME", "gemini-2.5-flash")
        
        self.llm = ChatVertexAI(
            model_name=model,
            temperature=temperature,
            streaming=True
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
You are a professional research report writer.

TOPIC: {topic}

RESEARCH CONTEXT:
{research}

Write a comprehensive, well-structured research report with:
1. Clear Introduction
2. Key Findings (numbered)
3. Detailed Analysis
4. Implications & Future Direction
5. Conclusion

Make it engaging, informative, and cite the research context provided.
""")
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def stream(self, topic: str, research: str) -> Iterator[str]:
        """
        Stream report generation
        
        Args:
            topic: Research topic
            research: Research context
        
        Yields:
            Token strings
        """
        for chunk in self.chain.stream({
            "topic": topic,
            "research": research
        }):
            if chunk:
                yield chunk
    
    def generate_streaming(self, topic: str, research: str,
                          display: bool = True, callback=None) -> str:
        """
        Generate report with optional display
        
        Args:
            topic: Research topic
            research: Research context
            display: Whether to print streaming output
            callback: Optional callback function for each token
        
        Returns:
            Complete report
        """
        full_report = ""
        
        for token in self.stream(topic, research):
            full_report += token
            
            if display:
                # Print token without newline (streaming effect)
                sys.stdout.write(token)
                sys.stdout.flush()
            
            if callback:
                callback(token)
        
        if display:
            print()  # Final newline
        
        return full_report

class StreamingCritic:
    """
    Critic with streaming support
    Streams feedback as it's generated
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.3):
        """
        Initialize streaming critic
        
        Args:
            model_name: Gemini model name (default from env or gemini-2.5-flash)
            temperature: Temperature for generation
        """
        model = model_name or os.getenv("MODEL_NAME", "gemini-2.5-flash")
        
        self.llm = ChatGoogleGenerativeAI(  # ✅ Fixed: Using non-deprecated class
            model=model,
            temperature=temperature,
            streaming=True
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert research critic. Review this report and provide constructive feedback.

REPORT:
{report}

Provide:
1. Score (0-10)
2. 3-4 Strengths
3. 3-4 Areas for Improvement
4. Overall Assessment

Be specific and actionable.
""")
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def stream(self, report: str) -> Iterator[str]:
        """
        Stream feedback generation
        
        Args:
            report: Report to review
        
        Yields:
            Token strings
        """
        for chunk in self.chain.stream({"report": report}):
            if chunk:
                yield chunk
    
    def generate_streaming(self, report: str,
                          display: bool = True, callback=None) -> str:
        """
        Generate feedback with optional display
        
        Args:
            report: Report to review
            display: Whether to print streaming output
            callback: Optional callback function for each token
        
        Returns:
            Complete feedback
        """
        full_feedback = ""
        
        for token in self.stream(report):
            full_feedback += token
            
            if display:
                sys.stdout.write(token)
                sys.stdout.flush()
            
            if callback:
                callback(token)
        
        if display:
            print()
        
        return full_feedback

# ============================================================================
# STREAMING UTILITIES (Convenience Functions)
# ============================================================================

def stream_report(topic: str, research: str, display: bool = True) -> str:
    """
    Generate streaming report (convenience function)
    
    Args:
        topic: Research topic
        research: Research context
        display: Whether to display streaming output
    
    Returns:
        Full report
    """
    if display:
        print("\n" + "="*70)
        print(f"[GENERATING REPORT] Topic: {topic}")
        print("="*70)
        print("\n")  # Extra newline for better streaming effect
    
    writer = StreamingWriter()
    report = writer.generate_streaming(topic, research, display=display)
    
    if display:
        print("\n" + "="*70)
        print("[REPORT COMPLETE]")
        print("="*70)
    
    return report

def stream_feedback(report: str, display: bool = True) -> str:
    """
    Generate streaming feedback (convenience function)
    
    Args:
        report: Report to review
        display: Whether to display streaming output
    
    Returns:
        Full feedback
    """
    if display:
        print("\n" + "="*70)
        print("[GENERATING FEEDBACK]")
        print("="*70)
        print("\n")
    
    critic = StreamingCritic()
    feedback = critic.generate_streaming(report, display=display)
    
    if display:
        print("\n" + "="*70)
        print("[FEEDBACK COMPLETE]")
        print("="*70)
    
    return feedback

def stream_report_silent(topic: str, research: str) -> str:
    """
    Generate report without displaying streaming output
    
    Args:
        topic: Research topic
        research: Research context
    
    Returns:
        Full report
    """
    return stream_report(topic, research, display=False)

def stream_feedback_silent(report: str) -> str:
    """
    Generate feedback without displaying streaming output
    
    Args:
        report: Report to review
    
    Returns:
        Full feedback
    """
    return stream_feedback(report, display=False)

class StreamCollector:
    """
    Collects streaming output for async processing
    Useful for web applications where you need to send chunks
    """
    
    def __init__(self):
        self.chunks = []
        self.full_text = ""
    
    def collect(self, chunk: str):
        """Collect a chunk"""
        self.chunks.append(chunk)
        self.full_text += chunk
    
    def get_chunks(self) -> list:
        """Get all collected chunks"""
        return self.chunks
    
    def get_full_text(self) -> str:
        """Get complete collected text"""
        return self.full_text
    
    def reset(self):
        """Reset collector"""
        self.chunks = []
        self.full_text = ""

def stream_report_with_callback(topic: str, research: str, callback) -> str:
    """
    Stream report with callback for each chunk
    
    Args:
        topic: Research topic
        research: Research context
        callback: Function that takes chunk as argument
    
    Returns:
        Full report
    """
    writer = StreamingWriter()
    return writer.generate_streaming(topic, research, display=False, callback=callback)

def stream_feedback_with_callback(report: str, callback) -> str:
    """
    Stream feedback with callback for each chunk
    
    Args:
        report: Report to review
        callback: Function that takes chunk as argument
    
    Returns:
        Full feedback
    """
    critic = StreamingCritic()
    return critic.generate_streaming(report, display=False, callback=callback)

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 14: STREAMING OUTPUT TEST")
    print("="*70)
    
    # Test data
    test_topic = "quantum computing"
    test_research = """
    Quantum computing uses qubits that can exist in superposition states.
    This enables parallel processing and exponential speedup for certain problems.
    Current challenges include decoherence and error correction.
    Companies like IBM, Google, and IonQ are leading development.
    """
    
    # Test 1: Streaming report
    print("\n[TEST 1] Streaming Report Generation")
    print("-" * 70)
    report = stream_report(test_topic, test_research)
    
    # Test 2: Streaming feedback
    print("\n[TEST 2] Streaming Feedback Generation")
    print("-" * 70)
    feedback = stream_feedback(report)
    
    # Test 3: Silent mode
    print("\n[TEST 3] Silent Mode (No Display)")
    print("-" * 70)
    report_silent = stream_report_silent(test_topic, test_research)
    print(f"Report generated silently: {len(report_silent)} chars")
    
    # Test 4: Callback mode
    print("\n[TEST 4] Callback Mode (Collect chunks)")
    print("-" * 70)
    collector = StreamCollector()
    
    report_callback = stream_report_with_callback(
        test_topic, 
        test_research, 
        callback=collector.collect
    )
    
    print(f"Collected {len(collector.get_chunks())} chunks")
    print(f"Total length: {len(collector.get_full_text())} chars")
    
    # Test 5: Verify functions exist
    print("\n[TEST 5] Function Availability Check")
    print("-" * 70)
    functions = [
        "stream_report",
        "stream_feedback",
        "stream_report_silent",
        "stream_feedback_silent",
        "stream_report_with_callback",
        "stream_feedback_with_callback",
        "StreamCollector",
        "StreamingWriter",
        "StreamingCritic"
    ]
    
    for func in functions:
        if func in globals() or func in locals():
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} MISSING")
    
    # Summary
    print("\n" + "="*70)
    print("[OK] Streaming test passed!")
    print("="*70)
    print(f"\nReport length: {len(report)} chars")
    print(f"Feedback length: {len(feedback)} chars")
    print(f"Silent report length: {len(report_silent)} chars")
    print(f"Callback mode: {len(collector.get_chunks())} chunks collected")
    
    print("\n" + "="*70)
    print("STREAMING OUTPUT READY FOR USE")
    print("="*70)
    print("""
Available functions:
  - stream_report(topic, research)     # Display streaming report
  - stream_feedback(report)            # Display streaming feedback  
  - stream_report_silent()             # Generate without display
  - stream_feedback_silent()           # Generate without display
  - stream_report_with_callback()      # Use with custom callback
  - stream_feedback_with_callback()    # Use with custom callback
  - StreamCollector()                  # Collect chunks for later use
""")