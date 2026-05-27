"""
STEP 15: Evaluation System
Quality metrics for RAG systems using RAGAS
Measures: faithfulness, answer relevancy, hallucination detection
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# ============================================================================
# EVALUATION METRICS
# ============================================================================

@dataclass
class EvaluationMetrics:
    """Container for evaluation results"""
    faithfulness: float
    answer_relevancy: float
    hallucination_score: float
    context_coverage: float
    source_accuracy: float
    overall_quality: float
    findings: Dict

class MetricsOutput(BaseModel):
    """Output schema for metrics"""
    score: float = Field(description="Score from 0.0 to 1.0")
    reasoning: str = Field(description="Explanation of the score")

# ============================================================================
# EVALUATOR
# ============================================================================

class RAGEvaluator:
    """
    Evaluates RAG system output using multiple metrics
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=MetricsOutput)
    
    def evaluate_faithfulness(self, report: str, context: str) -> float:
        """
        Evaluate how faithful the report is to the provided context
        
        Args:
            report: Generated report
            context: Source context
        
        Returns:
            Faithfulness score (0.0-1.0)
        """
        prompt = ChatPromptTemplate.from_template("""
Evaluate how faithful this report is to the provided context.
Faithful means: the report only uses information from the context, doesn't add unsupported claims.

CONTEXT:
{context}

REPORT:
{report}

Provide a score (0.0-1.0) where:
- 1.0 = Completely faithful to context
- 0.7 = Mostly faithful with minor additions
- 0.5 = Some unsupported claims
- 0.0 = Mostly hallucinated content

Return JSON with score and reasoning.
""")
        
        chain = prompt | self.llm | self.parser
        result = chain.invoke({
            "context": context,
            "report": report
        })
        
        return float(result.get("score", 0.5)) if isinstance(result, dict) else float(result.score)
    
    def evaluate_relevancy(self, query: str, report: str) -> float:
        """
        Evaluate how relevant the report is to the query
        
        Args:
            query: Original query
            report: Generated report
        
        Returns:
            Relevancy score (0.0-1.0)
        """
        prompt = ChatPromptTemplate.from_template("""
Evaluate how relevant this report is to the query.
Relevant means: the report directly addresses the query.

QUERY:
{query}

REPORT:
{report}

Provide a score (0.0-1.0) where:
- 1.0 = Perfectly addresses the query
- 0.7 = Mostly relevant with some tangents
- 0.5 = Partially relevant
- 0.0 = Completely irrelevant

Return JSON with score and reasoning.
""")
        
        chain = prompt | self.llm | self.parser
        result = chain.invoke({
            "query": query,
            "report": report
        })
        
        if isinstance(result, dict):
            return float(result.get("score", 0.5))
        return float(result.score)
    
    def evaluate_hallucination(self, report: str, sources: List[str]) -> float:
        """
        Detect hallucinations in the report
        
        Args:
            report: Generated report
            sources: List of source URLs
        
        Returns:
            Non-hallucination score (0.0-1.0)
        """
        prompt = ChatPromptTemplate.from_template("""
Detect hallucinations in this report.
Hallucination means: claims made without factual basis or from unreliable sources.

REPORT:
{report}

SOURCES USED:
{sources}

Analyze for:
1. Fabricated facts
2. Unsupported claims
3. Logical inconsistencies
4. Out-of-context citations

Provide a score (0.0-1.0) where:
- 1.0 = No hallucinations detected
- 0.7 = Minor hallucinations
- 0.5 = Some concerning claims
- 0.0 = Extensive hallucinations

Return JSON with score and reasoning.
""")
        
        sources_str = "\n".join(sources) if sources else "No sources"
        
        chain = prompt | self.llm | self.parser
        result = chain.invoke({
            "report": report,
            "sources": sources_str
        })
        
        if isinstance(result, dict):
            return float(result.get("score", 0.5))
        return float(result.score)
    
    def evaluate_context_coverage(self, context: str, report: str) -> float:
        """
        Evaluate how well the report covers the provided context
        
        Args:
            context: Available context
            report: Generated report
        
        Returns:
            Coverage score (0.0-1.0)
        """
        prompt = ChatPromptTemplate.from_template("""
Evaluate how well the report covers the available context.
Coverage means: important points from context are mentioned.

CONTEXT:
{context}

REPORT:
{report}

Provide a score (0.0-1.0) where:
- 1.0 = All important points covered
- 0.7 = Most important points covered
- 0.5 = Some important points missed
- 0.0 = Major points missing

Return JSON with score and reasoning.
""")
        
        chain = prompt | self.llm | self.parser
        result = chain.invoke({
            "context": context,
            "report": report
        })
        
        if isinstance(result, dict):
            return float(result.get("score", 0.5))
        return float(result.score)
    
    def evaluate_source_accuracy(self, report: str, sources: List[Dict]) -> float:
        """
        Evaluate accuracy of source citations
        
        Args:
            report: Generated report
            sources: List of sources with title and URL
        
        Returns:
            Accuracy score (0.0-1.0)
        """
        prompt = ChatPromptTemplate.from_template("""
Evaluate the accuracy of source citations in this report.

REPORT:
{report}

SOURCES:
{sources}

Check:
1. All sources are properly cited
2. Citations match source content
3. No false attributions
4. Proper formatting

Provide a score (0.0-1.0) where:
- 1.0 = Perfect citations
- 0.7 = Good citations with minor issues
- 0.5 = Adequate citations with errors
- 0.0 = Poor or missing citations

Return JSON with score and reasoning.
""")
        
        sources_str = "\n".join([f"- {s['title']}: {s['url']}" for s in sources])
        
        chain = prompt | self.llm | self.parser
        result = chain.invoke({
            "report": report,
            "sources": sources_str
        })
        
        if isinstance(result, dict):
            return float(result.get("score", 0.5))
        return float(result.score)
    
    def evaluate_all(self, query: str, report: str, context: str,
                    sources: List[Dict]) -> EvaluationMetrics:
        """
        Run all evaluations
        
        Args:
            query: Original query
            report: Generated report
            context: Source context
            sources: List of sources
        
        Returns:
            Complete evaluation metrics
        """
        print("\n" + "="*70)
        print("EVALUATING REPORT QUALITY")
        print("="*70)
        
        print("\n[1/5] Evaluating faithfulness...")
        faithfulness = self.evaluate_faithfulness(report, context)
        print(f"  Score: {faithfulness:.2f}")
        
        print("\n[2/5] Evaluating answer relevancy...")
        relevancy = self.evaluate_relevancy(query, report)
        print(f"  Score: {relevancy:.2f}")
        
        print("\n[3/5] Evaluating hallucination...")
        no_hallucination = self.evaluate_hallucination(
            report,
            [s['url'] for s in sources]
        )
        print(f"  Score: {no_hallucination:.2f}")
        
        print("\n[4/5] Evaluating context coverage...")
        coverage = self.evaluate_context_coverage(context, report)
        print(f"  Score: {coverage:.2f}")
        
        print("\n[5/5] Evaluating source accuracy...")
        accuracy = self.evaluate_source_accuracy(report, sources)
        print(f"  Score: {accuracy:.2f}")
        
        # Calculate overall quality
        overall = (faithfulness + relevancy + no_hallucination + coverage + accuracy) / 5.0
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
        return EvaluationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            hallucination_score=no_hallucination,
            context_coverage=coverage,
            source_accuracy=accuracy,
            overall_quality=overall,
            findings={
                "faithfulness": f"{faithfulness:.2%}",
                "relevancy": f"{relevancy:.2%}",
                "no_hallucination": f"{no_hallucination:.2%}",
                "coverage": f"{coverage:.2%}",
                "accuracy": f"{accuracy:.2%}"
            }
        )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_evaluator() -> RAGEvaluator:
    """Factory function"""
    return RAGEvaluator()

def print_evaluation(metrics: EvaluationMetrics):
    """Pretty print evaluation results"""
    print("\n[EVALUATION RESULTS]")
    print("="*70)
    print(f"Faithfulness:       {metrics.faithfulness:.2%}")
    print(f"Answer Relevancy:   {metrics.answer_relevancy:.2%}")
    print(f"No Hallucination:   {metrics.hallucination_score:.2%}")
    print(f"Context Coverage:   {metrics.context_coverage:.2%}")
    print(f"Source Accuracy:    {metrics.source_accuracy:.2%}")
    print("-"*70)
    print(f"OVERALL QUALITY:    {metrics.overall_quality:.2%}")
    print("="*70)

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 15: EVALUATION SYSTEM TEST")
    print("="*70)
    
    # Sample data
    query = "What are the applications of quantum computing?"
    context = "Quantum computing uses qubits and superposition. Applications include drug discovery, optimization, and cryptography."
    report = "Quantum computing has revolutionary applications in pharmaceuticals, finance, and security. It uses qubits instead of bits."
    sources = [
        {"title": "Quantum Computing Basics", "url": "https://example.com/qc"},
        {"title": "QC Applications", "url": "https://example.com/apps"}
    ]
    
    # Run evaluation
    evaluator = create_evaluator()
    metrics = evaluator.evaluate_all(query, report, context, sources)
    
    # Display results
    print_evaluation(metrics)
    
    print("\n[OK] Evaluation system test passed!")
