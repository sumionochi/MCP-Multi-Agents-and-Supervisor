import os
import asyncio
import textwrap
from dotenv import load_dotenv

from langchain_core.tools import StructuredTool
from langchain_core.messages import convert_to_messages
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor

load_dotenv()

# Bright Data exposes around 50 tools, the agents only need these two.
ALLOWED_TOOLS = {"search_engine", "scrape_as_markdown"}

# A single scraped page can be larger than the whole per minute token limit.
MAX_TOOL_CHARS = 3000

AGENT_COLORS = {
    "supervisor": "\033[95m",
    "stock_research_agent": "\033[96m",
    "market_data_agent": "\033[94m",
    "news_analyst_agent": "\033[93m",
    "price_recommendation_agent": "\033[92m",
}
GREY = "\033[90m"
RESET = "\033[0m"

def limit_tool_output(tool):
    """Wrap a tool so a huge page cannot flood the model context."""

    async def call_tool(**kwargs):
        result = await tool.ainvoke(kwargs)
        text = str(result)

        if len(text) > MAX_TOOL_CHARS:
            text = text[:MAX_TOOL_CHARS] + "\n[truncated]"

        return text

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=call_tool,
    )


def message_text(message):
    content = message.content

    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )

    return (content or "").strip()


def short_args(args):
    parts = []

    for key, value in (args or {}).items():
        value = str(value).replace("\n", " ")
        if len(value) > 50:
            value = value[:50] + "..."
        parts.append(f"{key}={value}")

    return ", ".join(parts)


def print_text(text):
    for paragraph in text.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(paragraph, width=90,
                                initial_indent="  ", subsequent_indent="  "))


class Transcript:
    """Prints the handoffs, tool calls and agent answers while the graph runs."""

    def __init__(self):
        self.seen = set()
        self.speaker = None
        self.final_answer = ""

    def header(self, speaker):
        if speaker == self.speaker:
            return

        self.speaker = speaker
        color = AGENT_COLORS.get(speaker, "")
        print(f"\n{color}[{speaker}]{RESET}")

    def step(self, text):
        print(f"{GREY}  {text}{RESET}")

    def handle(self, namespace, update):
        if not isinstance(update, dict):
            return

        for node_name, node_update in update.items():
            if not isinstance(node_update, dict) or "messages" not in node_update:
                continue

            # Inside a sub agent the namespace carries its name, at the top
            # level the node name is the speaker.
            speaker = namespace[-1].split(":")[0] if namespace else node_name

            for message in convert_to_messages(node_update["messages"]):
                self.show(speaker, message)

    def show(self, speaker, message):
        # The same message arrives once from the sub graph and once from the parent.
        if message.id in self.seen:
            return
        self.seen.add(message.id)

        if message.type == "tool":
            if not (message.name or "").startswith("transfer_"):
                self.header(speaker)
                self.step(f"got {len(message_text(message))} chars back")
            return

        if message.type != "ai":
            return

        for call in message.tool_calls:
            name = call["name"]
            self.header(speaker)

            if name.startswith("transfer_to_"):
                self.step(f"handing off to {name.replace('transfer_to_', '')}")
            elif name.startswith("transfer_back"):
                self.step("handing back to supervisor")
            else:
                self.step(f"calling {name}({short_args(call['args'])})")

        text = message_text(message)

        if not text or text.startswith("Transferring back to"):
            return

        # The supervisor's last plain message is the answer, shown at the end.
        if speaker == "supervisor" and not message.tool_calls:
            self.final_answer = text
            return

        self.header(speaker)
        print_text(text)

    def print_final(self):
        print()
        print("=" * 90)
        print("FINAL ANSWER")
        print("=" * 90)
        print_text(self.final_answer or "No final answer was produced.")
        print()


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

    tools = [
        limit_tool_output(tool)
        for tool in await client.get_tools()
        if tool.name in ALLOWED_TOOLS
    ]

    model = init_chat_model(
        model="openai:gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        max_retries=5,
        # Spaces the calls out so the run stays inside the tokens per minute limit.
        rate_limiter=InMemoryRateLimiter(requests_per_second=0.33),
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
            "Use at most 3 tool calls and summarize instead of quoting pages. "
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
            "Use at most 3 tool calls and summarize instead of quoting pages. "
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
            "Do not exaggerate or invent news. "
            "Use at most 3 tool calls and return at most 5 short bullet points."
        ),
        name="news_analyst_agent",
    )

    price_recommendation_agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a stock comparison and valuation analyst specializing in DAX-listed companies. "
            "Your job is to evaluate stocks using fundamental research, market data, and recent news. "
            "Use the research already present in the conversation and avoid new searches "
            "unless something essential is missing. "
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
            "- Pick three concrete DAX companies first and name them with their "
            "Xetra tickers in every handoff, for example Siemens AG (SIE.DE).\n"
            "- Assign each task to the most suitable agent.\n"
            "- Use one agent at a time. Do not call agents in parallel.\n"
            "- Call each research agent at most once.\n"
            "- Do not do the research yourself.\n"
            "- After receiving the necessary research, ask the "
            "price_recommendation_agent to prepare the final comparison.\n"
            "- Return the final answer to the user.\n"
            "- Do not invent financial figures or guarantee returns.\n"
        ),
        add_handoff_back_messages=True,
        # Was "full_history", which pushed every scraped page from every agent
        # back into the supervisor and then into the next agent.
        output_mode="last_message",
    )

    app = supervisor.compile()

    return app


async def run_multi_agent1(query):
    app = await workspace(query)
    transcript = Transcript()

    print("=" * 90)
    print(f"DAX MULTI-AGENT RESEARCH: {query}")
    print("=" * 90)

    async for namespace, update in app.astream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
        subgraphs=True,
    ):
        transcript.handle(namespace, update)

    transcript.print_final()

    return transcript.final_answer


if __name__ == "__main__":

    asyncio.run(
        run_multi_agent1(
            "Give me good stock recommendations from DAX"
        )
    )