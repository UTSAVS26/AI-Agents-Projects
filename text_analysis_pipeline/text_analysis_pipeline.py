import os
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from IPython.display import Image, display

# --- 1. Environment and Model Setup ---
load_dotenv()
llm = ChatGroq(
    model="gemma2-9b-it",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# --- 2. Pydantic and State Definitions ---

class EntitiesOutput(BaseModel):
    """A list of named entities extracted from the text."""
    entities: List[str] = Field(description="A list of named entities (People, Organizations, Locations) found in the text.")

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    text: str
    classification: Optional[str]
    entities: Optional[List[str]]
    summary: Optional[str]
    sentiment: Optional[str]
    report: Optional[str]

# --- 3. Node Definitions ---

def classification_node(state: GraphState):
    print("---CLASSIFYING TEXT---")
    prompt = PromptTemplate(
        template="""Classify the following text into one of the categories: News, Blog, Research, or Other.
        Respond with only the category name.

        Text: {text}

        Category:""",
        input_variables=["text"],
    )
    chain = prompt | llm
    classification = chain.invoke({"text": state["text"]}).content.strip()
    print(f"Classification: {classification}")
    return {"classification": classification}

# <--- CHANGE 3: Update the entity extraction node
def entity_extraction_node(state: GraphState):
    """
    Extracts all entities (Person, Organization, Location) from the text using JSON output.
    """
    print("---EXTRACTING ENTITIES---")
    # Use the Pydantic model we defined for more robust parsing.
    parser = JsonOutputParser(pydantic_object=EntitiesOutput)

    prompt = PromptTemplate(
        template="""Extract all the named entities (People, Organizations, Locations) from the text.
        {format_instructions}

        Text: {text}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    # The output of the chain will be a dictionary like {'entities': ['...']}
    parsed_output = chain.invoke({"text": state["text"]})
    entities = parsed_output['entities'] # Extract the list from the dictionary
    print(f"Entities Found: {entities}")
    return {"entities": entities}

def standard_summarization_node(state: GraphState):
    print("---GENERATING STANDARD SUMMARY---")
    prompt = PromptTemplate(
        template="""Summarize the following text in one concise sentence.

        Text: {text}

        Summary:""",
        input_variables=["text"],
    )
    chain = prompt | llm
    summary = chain.invoke({"text": state["text"]}).content.strip()
    print(f"Standard Summary: {summary}")
    return {"summary": summary}

def detailed_summarization_node(state: GraphState):
    print("---GENERATING DETAILED SUMMARY---")
    prompt = PromptTemplate(
        template="""Create a detailed, multi-point summary of the following research text.
        Focus on the key findings, methodology, and implications.

        Text: {text}

        Detailed Summary:""",
        input_variables=["text"],
    )
    chain = prompt | llm
    summary = chain.invoke({"text": state["text"]}).content.strip()
    print(f"Detailed Summary: {summary}")
    return {"summary": summary}

def sentiment_analysis_node(state: GraphState):
    print("---ANALYZING SENTIMENT---")
    prompt = PromptTemplate(
        template="""Analyze the sentiment of the following blog post.
        Respond with only one of these words: Positive, Negative, or Neutral.

        Text: {text}

        Sentiment:""",
        input_variables=["text"],
    )
    chain = prompt | llm
    sentiment = chain.invoke({"text": state["text"]}).content.strip()
    print(f"Sentiment: {sentiment}")
    return {"sentiment": sentiment}

def report_generation_node(state: GraphState):
    print("---GENERATING FINAL REPORT---")
    # Gracefully handle the case where entities might be None or empty
    entities_str = "None"
    if state.get('entities'):
        entities_str = '- ' + '\n- '.join(state['entities'])

    report = f"""
## Text Analysis Report

**1. Classification:**
{state['classification']}

**2. Extracted Entities:**
{entities_str}
"""

    if state.get('summary'):
        report += f"""
**3. Summary:**
{state['summary']}
"""

    if state.get('sentiment'):
        report += f"""
**3. Sentiment Analysis:**
The sentiment of the text is **{state['sentiment']}**.
"""

    print(report)
    return {"report": report.strip()}

# --- 4. Graph and Conditional Logic Definition ---

def decide_path(state: GraphState) -> str:
    print("---DECIDING PATH---")
    classification = state["classification"].lower()
    if "research" in classification:
        return "research_branch"
    if "blog" in classification:
        return "blog_branch"
    if "news" in classification:
        return "news_branch"
    return "end"

workflow = StateGraph(GraphState)

workflow.add_node("classify", classification_node)
workflow.add_node("extract_entities", entity_extraction_node)
workflow.add_node("standard_summary", standard_summarization_node)
workflow.add_node("detailed_summary", detailed_summarization_node)
workflow.add_node("analyze_sentiment", sentiment_analysis_node)
workflow.add_node("generate_report", report_generation_node)

workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    decide_path,
    {
        "research_branch": "extract_entities",
        "blog_branch": "extract_entities",
        "news_branch": "extract_entities",
        "end": END,
    },
)

# This part had a small logic error. A research paper should get a detailed summary,
# not also a standard one. This corrected logic routes each branch correctly.
workflow.add_edge("extract_entities", "detailed_summary")
workflow.add_edge("extract_entities", "standard_summary")
workflow.add_edge("extract_entities", "analyze_sentiment")

# The original graph definition forked from extract_entities but never specified *which*
# fork to take. We need another conditional edge. A simpler way for now is to adjust
# the logic *after* entity extraction. Let's create a new conditional router.

def route_after_entities(state: GraphState) -> str:
    """Decide what to do after extracting entities."""
    print("---ROUTING AFTER ENTITY EXTRACTION---")
    classification = state["classification"].lower()
    if "research" in classification:
        return "detailed_summary"
    if "blog" in classification:
        return "analyze_sentiment"
    # Default to standard summary for News or other classified types
    return "standard_summary"

# Let's redefine the graph structure for more clarity
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("classify", classification_node)
workflow.add_node("extract_entities", entity_extraction_node)
workflow.add_node("detailed_summary", detailed_summarization_node)
workflow.add_node("standard_summary", standard_summarization_node)
workflow.add_node("analyze_sentiment", sentiment_analysis_node)
workflow.add_node("generate_report", report_generation_node)

# Set entry and first conditional edge
workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    decide_path,
    {"research_branch": "extract_entities", "blog_branch": "extract_entities", "news_branch": "extract_entities", "end": END}
)

# Add the second conditional edge after entity extraction
workflow.add_conditional_edges(
    "extract_entities",
    route_after_entities,
    {"detailed_summary": "detailed_summary", "standard_summary": "standard_summary", "analyze_sentiment": "analyze_sentiment"}
)


# All processing branches now converge on the final report node
workflow.add_edge("detailed_summary", "generate_report")
workflow.add_edge("standard_summary", "generate_report")
workflow.add_edge("analyze_sentiment", "generate_report")
workflow.add_edge("generate_report", END)

# --- 5. Compilation and Visualization ---
app = workflow.compile()

try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("text_analysis_pipeline/workflow_graph.png", "wb") as f:
        f.write(png_data)
    display(Image(png_data))
except Exception as e:
    print(f"Graph visualization failed: {e}")
    print("Install visualization dependencies with: pip install pydot graphviz")

# --- 6. Testing with Different Text Types ---
print("\n\n" + "="*50)
print("          RUNNING TEST 1: RESEARCH PAPER")
print("="*50 + "\n")

research_text = """
A study published in Nature Communications by Dr. Eva Rostova and her team at the Zurich Institute of Technology has revealed a novel method for carbon capture.
The process, termed 'Cryo-Adsorption', utilizes porous metal-organic frameworks (MOFs) at cryogenic temperatures to selectively adsorb CO2 from flue gas streams with over 99% efficiency.
This breakthrough could significantly reduce the cost and energy requirements of industrial carbon capture, paving the way for more widespread adoption in power plants and manufacturing facilities.
"""
state_input = {"text": research_text}
result_research = app.invoke(state_input)
print("\n--- FINAL RESEARCH RESULT ---")
print(result_research['report'])


print("\n\n" + "="*50)
print("            RUNNING TEST 2: BLOG POST")
print("="*50 + "\n")

blog_text = """
I just spent a week with the new 'PixelPro 10' camera, and I'm absolutely blown away!
The image quality is stunning, the battery life is incredible, and the new AI-powered editing tools from Google are a game-changer.
While it's a bit pricey, I think it's worth every penny for serious photographers. I can't recommend it enough. A fantastic job by the team in Mountain View.
"""
state_input = {"text": blog_text}
result_blog = app.invoke(state_input)
print("\n--- FINAL BLOG RESULT ---")
print(result_blog['report'])

print("\n\n" + "="*50)
print("            RUNNING TEST 3: NEWS ARTICLE")
print("="*50 + "\n")

news_text = """
Today in Paris, international delegates from over 50 countries, including U.S. Secretary of State Antony Blinken, convened for the Global Climate Summit.
The summit, hosted by the United Nations, aims to secure new commitments for reducing greenhouse gas emissions and financing green technology for developing nations.
French President Emmanuel Macron opened the summit with a call for urgent and decisive action.
"""
state_input = {"text": news_text}
result_news = app.invoke(state_input)
print("\n--- FINAL NEWS RESULT ---")
print(result_news['report'])