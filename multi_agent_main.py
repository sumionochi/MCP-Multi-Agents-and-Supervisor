import os
import asyncio
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor

load_dotenv()


def pretty_print_message(message):
    print(message.pretty_repr())


async def workspace(query):
    client = MultiServerMCPClient(
        {
            "Bright Data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                },
                "transport": "stdio",
            },
        }
    )

    tools = await client.get_tools()

    model = init_chat_model(
        model="openai:gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000,
    )

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
            "Provide concise research that another financial analyst can use. "
            "Do not copy or return long scraped content."
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
            "Provide concise market data that another analyst can use. "
            "Do not copy or return long scraped content."
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
            "Prioritize recent and reliable sources. "
            "Explain what happened, why it matters, and whether the impact appears "
            "positive, negative, or uncertain. "
            "Clearly separate facts from interpretation. "
            "Mention publication dates. "
            "Do not exaggerate or invent news. "
            "Return a maximum of 3 short bullet points. "
            "Never copy or return full article text."
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
            "Present the final comparison in a clean, structured format. "
            "Keep the final response concise."
        ),
        name="price_recommendation_agent",
    )

    supervisor = create_supervisor(
        model=model,
        agents=[
            stock_research_agent,
            market_data_agent,
            news_analyst_agent,
            price_recommendation_agent,
        ],
        prompt=(
            "You are a supervisor managing four specialized financial agents.\n\n"

            "Available agents:\n"
            "- stock_research_agent: Research company fundamentals, "
            "earnings, revenue, profitability, debt, and valuation.\n"
            "- market_data_agent: Retrieve stock prices, market "
            "capitalization, trading volume, and historical performance.\n"
            "- news_analyst_agent: Find and analyze recent company "
            "news and important market developments.\n"
            "- price_recommendation_agent: Compare companies and "
            "evaluate their relative attractiveness.\n\n"

            "Instructions:\n"
            "- Assign each task to the most suitable agent.\n"
            "- Use one agent at a time. Do not call agents in parallel.\n"
            "- Do not do the research yourself.\n"
            "- Keep every agent response concise.\n"
            "- Do not include scraped article text.\n"
            "- After receiving the necessary research, ask the "
            "price_recommendation_agent to prepare the final comparison.\n"
            "- Return the final answer to the user.\n"
            "- Do not invent financial figures or guarantee returns.\n"
        ),
        add_handoff_back_messages=True,
        output_mode="last_message",
    )

    app = supervisor.compile()

    return app


async def run_multi_agent1(query):
    app = await workspace(query)

    result = await app.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    for message in result["messages"]:
        pretty_print_message(message)

    return result


if __name__ == "__main__":
    asyncio.run(
        run_multi_agent1(
            "Give me good stock recommendations from DAX"
        )
    )