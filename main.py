#https://jpmc.udemy.com/course/langchain/learn/lecture/53365485#overview
# answers in projets/search agent branch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
"""
Notes, alternatively, you can use langhcain-tavily package to integrate with Tavily, 
which provides a more seamless experience for using Tavily's capabilities within LangChain agents.
 Here's how you can set it up:
"""
load_dotenv()
tavily = TavilyClient()

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    print (f"Getting weather for {location}...")
    return f"The weather in {location} is sunny."

@tool
def search_web(query: str) -> str:
    """Search the web for a given query and return the results."""
    print (f"Searching the web for '{query}'...")
    #return f"Search results for '{query}'"
    return tavily.search(query)

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7)
tools = [search_web, get_weather]

#agent = create_agent(llm=llm, tools=tools, agent_type="zero-shot-react-description")
agent = create_agent(model=llm, tools=tools)

def main():
    user_input = "What's the price of a Tesla Model S?"
    #response = agent.run(HumanMessage(content=user_input))
    ## or
    response =agent.invoke({'messages':HumanMessage(content=user_input)})
    ### diff between run and invoke
    print(response)

if __name__ == "__main__":
    main()
    print("Hello, LangChain!")