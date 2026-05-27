"""
PHASE 3: Advanced AI Engineering - Integrated System
Combines all 6 steps into production-ready system
"""

from langgraph_workflow import run_graph_pipeline, PipelineState
from planner_agent import create_planner
from memory_system import ResearchMemory
from async_scraper import scrape_urls_async
from streaming_output import stream_report, stream_feedback
from evaluation_system import create_evaluator, print_evaluation
from typing import Optional
import traceback

# ============================================================================
# ADVANCED RAG SYSTEM
# ============================================================================

class AdvancedRAGSystem:
    """
    Production-grade RAG system with all PHASE 3 features:
    - LangGraph workflows
    - Strategic planning
    - Memory management
    - Async scraping
    - Streaming output
    - Quality evaluation
    """
    
    def __init__(self, enable_memory: bool = True, enable_streaming: bool = True):
        """
        Initialize advanced RAG system
        
        Args:
            enable_memory: Enable memory system
            enable_streaming: Enable streaming output
        """
        print("\n" + "="*70)
        print("INITIALIZING ADVANCED RAG SYSTEM")
        print("="*70)
        
        try:
            self.planner = create_planner()
            print("[OK] Planner Agent initialized")
        except Exception as e:
            print(f"[WARN] Planner Agent failed: {e}")
            self.planner = None
        
        try:
            self.memory = ResearchMemory() if enable_memory else None
            if enable_memory:
                print("[OK] Memory System initialized")
        except Exception as e:
            print(f"[WARN] Memory System failed: {e}")
            self.memory = None
        
        try:
            self.evaluator = create_evaluator()
            print("[OK] Evaluator initialized")
        except Exception as e:
            print(f"[WARN] Evaluator failed: {e}")
            self.evaluator = None
        
        self.enable_streaming = enable_streaming
        if enable_streaming:
            print("[OK] Streaming enabled")
        
        print("\n[READY] Advanced RAG System initialized")
    
    def execute(self, query: str, use_planner: bool = True,
               use_async: bool = True, evaluate: bool = True) -> dict:
        """
        Execute complete research workflow
        
        Args:
            query: Research query
            use_planner: Use planner agent for strategy
            use_async: Use async scraping
            evaluate: Evaluate results
        
        Returns:
            Complete results dict
        """
        results = {
            "query": query,
            "plan": None,
            "workflow_state": None,
            "evaluation": None,
            "memory_id": None,
            "error": None
        }
        
        print("\n" + "="*70)
        print("ADVANCED RESEARCH PIPELINE")
        print("="*70)
        
        # STEP 11: Planning
        if use_planner and self.planner:
            try:
                print("\n[STEP 11] Planning research strategy...")
                plan = self.planner.plan(query)
                results["plan"] = plan
                
                print(f"  - Search needed: {plan.needs_search}")
                print(f"  - URLs to scrape: {plan.num_urls}")
                print(f"  - Deeper research: {plan.deeper_research}")
                print(f"  - Focus: {plan.research_focus}")
                print(f"  - Confidence: {plan.confidence:.2f}")
            except Exception as e:
                print(f"  [ERROR] Planning failed: {e}")
                results["error"] = str(e)
        
        # STEP 10: LangGraph Workflow
        print("\n[STEP 10] Executing workflow graph...")
        try:
            workflow_state = run_graph_pipeline(query)
            results["workflow_state"] = workflow_state
            print(f"  [OK] Workflow completed")
        except Exception as e:
            print(f"  [ERROR] Workflow failed: {e}")
            traceback.print_exc()
            results["error"] = f"Workflow failed: {e}"
            return results
        
        # STEP 12: Store in Memory
        if self.memory and results["workflow_state"]:
            try:
                print("\n[STEP 12] Storing in memory...")
                self.memory.store_query(query)
                
                # Get report and sources safely
                report = workflow_state.get("report", "")
                sources = workflow_state.get("sources", [])
                
                if report:
                    memory_id = self.memory.store_report(query, report, sources)
                    results["memory_id"] = memory_id
                    print(f"  [OK] Report stored with ID: {memory_id}")
                
                # Store sources with actual content if available
                scraped_content = workflow_state.get("scraped_content", [])
                content_map = {item.get("url"): item.get("content", "") for item in scraped_content}
                
                for source in sources:
                    url = source.get("url", "")
                    title = source.get("title", "Unknown")
                    content = content_map.get(url, "")
                    
                    if url and title:
                        self.memory.store_source(url, title, content)
                
                print(f"  [OK] {len(sources)} sources stored")
            except Exception as e:
                print(f"  [WARN] Memory storage failed: {e}")
        
        # STEP 14: Streaming (optional - show preview if enabled)
        if self.enable_streaming and results["workflow_state"]:
            report = results["workflow_state"].get("report", "")
            if report:
                print(f"\n[STEP 14] Report preview (first 500 chars):")
                print("="*70)
                print(report[:500])
                if len(report) > 500:
                    print("...")
                print("="*70)
        
        # STEP 15: Evaluation
        if evaluate and self.evaluator and results["workflow_state"]:
            try:
                print("\n[STEP 15] Evaluating quality...")
                
                report = workflow_state.get("report", "")
                retrieved_context = workflow_state.get("retrieved_context", "")
                sources = workflow_state.get("sources", [])
                
                if report and retrieved_context:
                    metrics = self.evaluator.evaluate_all(
                        query,
                        report,
                        retrieved_context,
                        sources
                    )
                    
                    results["evaluation"] = metrics
                    print_evaluation(metrics)
                else:
                    print("  [WARN] Cannot evaluate - missing report or context")
            except Exception as e:
                print(f"  [WARN] Evaluation failed: {e}")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        
        return results
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics if memory is enabled"""
        if self.memory:
            return self.memory.get_memory_stats()
        return {"error": "Memory not enabled"}
    
    def clear_memory(self):
        """Clear all memory if enabled"""
        if self.memory:
            self.memory.clear_all()
            print("[OK] Memory cleared")
        else:
            print("[WARN] Memory not enabled")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_research(query: str, full_pipeline: bool = True) -> dict:
    """
    Quick research with default settings
    
    Args:
        query: Research query
        full_pipeline: Use all features
    
    Returns:
        Results dict
    """
    system = AdvancedRAGSystem(
        enable_memory=full_pipeline,
        enable_streaming=full_pipeline
    )
    
    return system.execute(
        query,
        use_planner=full_pipeline,
        use_async=full_pipeline,
        evaluate=full_pipeline
    )

def research_with_memory(query: str) -> dict:
    """
    Research with memory enabled (for follow-up questions)
    
    Args:
        query: Research query
    
    Returns:
        Results dict with memory
    """
    system = AdvancedRAGSystem(enable_memory=True, enable_streaming=True)
    return system.execute(query, use_planner=True, use_async=True, evaluate=True)

def research_quick(query: str) -> dict:
    """
    Fast research without evaluation (for quick answers)
    
    Args:
        query: Research query
    
    Returns:
        Results dict
    """
    system = AdvancedRAGSystem(enable_memory=False, enable_streaming=False)
    return system.execute(query, use_planner=False, use_async=False, evaluate=False)

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 3: ADVANCED AI ENGINEERING - SYSTEM TEST")
    print("="*70)
    
    # Test query
    test_query = "latest breakthroughs in artificial intelligence 2024"
    
    # Run complete system
    print(f"\n[INFO] Testing with query: '{test_query}'")
    print("[INFO] This may take 30-60 seconds...")
    
    results = quick_research(test_query, full_pipeline=True)
    
    # Display summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if results.get("error"):
        print(f"\n[ERROR] {results['error']}")
    
    if results.get("plan"):
        print(f"\n📋 Planned Strategy:")
        print(f"  - Search: {results['plan'].needs_search}")
        print(f"  - URLs: {results['plan'].num_urls}")
        print(f"  - Focus: {results['plan'].research_focus}")
    
    if results.get("workflow_state"):
        state = results["workflow_state"]
        print(f"\n📄 Workflow Results:")
        print(f"  - Report Length: {len(state.get('report', ''))} characters")
        print(f"  - Sources Found: {len(state.get('sources', []))} sources")
        print(f"  - Feedback Length: {len(state.get('feedback', ''))} characters")
        
        # Show first few sources
        if state.get("sources"):
            print(f"\n🔗 Top Sources:")
            for i, source in enumerate(state["sources"][:3], 1):
                print(f"  {i}. {source.get('title', 'Unknown')[:60]}")
    
    if results.get("evaluation"):
        metrics = results["evaluation"]
        print(f"\n⭐ Quality Scores:")
        print(f"  - Overall Quality: {metrics.overall_quality:.2%}")
        print(f"  - Faithfulness: {metrics.faithfulness:.2%}")
        print(f"  - Answer Relevancy: {metrics.answer_relevancy:.2%}")
        print(f"  - No Hallucination: {metrics.hallucination_score:.2%}")
    
    if results.get("memory_id"):
        print(f"\n💾 Memory ID: {results['memory_id']}")
    
    print("\n" + "="*70)
    print("[OK] Advanced system test complete!")
    print("="*70)
    
    # Optional: Show memory stats
    if results.get("workflow_state") and results["workflow_state"].get("sources"):
        print("\n[INFO] System ready for follow-up questions.")
        print("       Example: research_with_memory('Tell me more about the first finding')")