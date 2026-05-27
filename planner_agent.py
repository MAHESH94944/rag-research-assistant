"""
STEP 11: Planner Agent
Decides research strategy based on query
- Whether web search is needed
- How many URLs to scrape
- Whether deeper research is needed
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from typing import Optional

# ============================================================================
# PLANNER OUTPUT SCHEMA
# ============================================================================

class ResearchPlan(BaseModel):
    """Output schema for the planner agent"""
    needs_search: bool = Field(
        description="Whether web search is needed (True if external info required)"
    )
    search_query: Optional[str] = Field(
        description="Modified/optimized search query if needed"
    )
    num_urls: int = Field(
        description="Number of URLs to scrape (1-5)"
    )
    deeper_research: bool = Field(
        description="Whether deeper research is needed (multiple sources, cross-referencing)"
    )
    research_focus: str = Field(
        description="Primary focus area for research"
    )
    strategy: str = Field(
        description="Brief explanation of the chosen strategy"
    )
    confidence: float = Field(
        description="Confidence level (0.0-1.0)"
    )

# ============================================================================
# PLANNER AGENT
# ============================================================================

class PlannerAgent:
    """
    Strategic planning agent for RAG research
    Analyzes queries and creates research plans
    """
    
    def __init__(self):
        """Initialize planner with LLM"""
        self.llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0.3)
        self.parser = JsonOutputParser(pydantic_object=ResearchPlan)
        
        # Build prompt
        self.prompt = ChatPromptTemplate.from_template("""
You are a research planning agent. Analyze the user query and create a strategic research plan.

USER QUERY: {query}

Consider:
1. Does the query need current/external information (search needed)?
2. Is this a simple query (1-2 URLs) or complex (3-5 URLs)?
3. Does it need cross-referencing and multiple sources (deeper research)?

Return JSON with your plan:
{format_instructions}

IMPORTANT:
- For factual/current topics: needs_search=true
- For general knowledge: needs_search=false OR minimal search
- For controversial topics: needs_search=true (multiple sources)
- For technical topics: needs_search=true (recent developments)

Be strategic. Recruiters will see this.
""")
        
        self.chain = self.prompt | self.llm | self.parser
    
    def plan(self, query: str) -> ResearchPlan:
        """
        Create a research plan for the given query
        
        Args:
            query: User's research question/topic
        
        Returns:
            ResearchPlan with strategy
        """
        result = self.chain.invoke({
            "query": query,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # Convert dict to ResearchPlan if needed
        if isinstance(result, dict):
            return ResearchPlan(**result)
        return result

def create_planner() -> PlannerAgent:
    """Factory function to create planner agent"""
    return PlannerAgent()

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 11: PLANNER AGENT TEST")
    print("="*70)
    
    planner = create_planner()
    
    test_queries = [
        "What are the latest developments in quantum computing?",
        "Explain the theory of relativity",
        "Is AI dangerous?",
        "Python best practices for 2024"
    ]
    
    for query in test_queries:
        print(f"\n[QUERY] {query}")
        print("-" * 70)
        
        plan = planner.plan(query)
        
        print(f"Search Needed: {plan.needs_search}")
        print(f"Search Query: {plan.search_query or 'N/A'}")
        print(f"URLs to Scrape: {plan.num_urls}")
        print(f"Deeper Research: {plan.deeper_research}")
        print(f"Research Focus: {plan.research_focus}")
        print(f"Strategy: {plan.strategy}")
        print(f"Confidence: {plan.confidence:.2f}")
