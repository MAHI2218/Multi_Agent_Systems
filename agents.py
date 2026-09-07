from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
grok_api_key = os.getenv("GROK_API_KEY")
if not grok_api_key:
    raise RuntimeError("Set GROK_API_KEY in .env before running the pipeline.")

llm = ChatOpenAI(
    model=os.getenv("GROK_MODEL", "grok-3-mini"),
    temperature=0,
    api_key=grok_api_key,
    base_url="https://api.x.ai/v1",
)

# first agent 
def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search]

    )

# 2nd agent 
def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url]

    )

# writer chain

writer_prompt = ChatPromptTemplate.from_messages([

    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic_chain
critic_prompt = ChatPromptTemplate.from_messages([

       ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

