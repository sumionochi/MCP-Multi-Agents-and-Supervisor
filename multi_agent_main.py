import os
import asyncio
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()   

async def run_agent1(query):
    client = MultiServerMCPClient(
        {
            "Bright Data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                },
                "transport": "stdio", #stdio as local server only
            },
        }
    )
    tools = await client.get_tools()
    model = init_chat_model(model="openai:gpt-4o", api_key = os.getenv("OPENAI_API_KEY"))
    
    stock_research_agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a stock research analyst specializing in German equities. "
            "Research individual DAX-listed companies using Bright Data tools. "
            "Analyze revenue, earnings, profitability, growth, debt, valuation, "
            "and important recent company developments. "
            "Clearly separate reported facts from your own analysis. "
            "Mention the date of important financial data. "
            "Do not invent financial figures. "
            "Provide concise research that another financial analyst can use."
        ),
        name="stock_research_agent",
    )

    market_data_agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a market data analyst specializing in German stock markets. "
            "Retrieve the latest available market data for DAX-listed companies. "
            "Find stock prices, daily price changes, market capitalization, "
            "trading volume, and relevant historical performance. "
            "Always include the date and time of market data when available. "
            "Clearly distinguish live, delayed, and historical data. "
            "Do not invent prices or percentages. "
            "Provide concise market data that another analyst can use."
        ),
        name="market_data_agent",
    )

    news_analyst_agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a financial news analyst specializing in German companies "
            "and markets. Find and analyze the latest relevant news about "
            "DAX-listed companies. "
            "Search for company announcements, earnings news, economic developments, "
            "and major events that may affect stock prices. "
            "Prioritize recent and reliable sources. "
            "Explain what happened, why it matters, and whether the impact appears "
            "positive, negative, or uncertain. "
            "Clearly separate facts from interpretation. "
            "Mention publication dates. "
            "Do not exaggerate or invent news."
        ),
        name="news_analyst_agent",
    )

    price_recommendation_agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a stock comparison and valuation analyst specializing in DAX-listed companies. "
            "Your job is to evaluate stocks using fundamental research, market data, and recent news. "
            "Compare companies based on valuation, growth, profitability, financial strength, "
            "recent performance, and important risks. "
            "Identify which companies appear relatively attractive based on the available evidence. "
            "Explain the reasoning behind every conclusion. "
            "Do not guarantee returns or claim that a stock will rise. "
            "Clearly state that your output is research, not personalized financial advice. "
            "Present the final comparison in a clean, structured format."
        ),
        name="price_recommendation_agent",
    )
    
    agent_response = await agent.ainvoke({"messages": "what is the weather in hannover?"})
    print(agent_response["messages"][-1].pretty_repr()) #take final answer only

if __name__ == "__main__":
    asyncio.run(run_multi_agent1("Give me good stock recommendation from DAX"))