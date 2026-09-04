import os
import asyncio
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()   

async def run_agent1():
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
    agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a web search agent with access to Bright Data tools. "
            "Answer clearly and concisely. "
            "For weather questions, use this exact format:\n\n"
            "🌤️ Weather Report\n"
            "📍 Location: ...\n"
            "🌡️ Temperature: ...\n"
            "☁️ Conditions: ...\n"
            "💨 Wind: ...\n"
            "🌧️ Forecast: ...\n\n"
            "Do not use Markdown links or bullet points."
        )
    )
    agent_response = await agent.ainvoke({"messages": "what is the weather in hannover?"})
    print(agent_response["messages"][-1].content) #take final answer only

if __name__ == "__main__":
    asyncio.run(run_agent1())