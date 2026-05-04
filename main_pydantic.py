#section 3.22
"""
the goal of pyadantic and agent response is
to specify the field of the output of the agent, 
and to make sure that the output is in a specific format,
 which can be easily parsed and used by other parts of the code.

To Do:
reate the AgentResponse model to include an answer field and a sources field,
, where the sources field is a list of Source models.
and then in create_agent(), add the response_format as AgentResponse
"""


# answers in projets/search agent branch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv()
tavily = TavilyClient()


class Source(BaseModel):
    """
    Represents a source of information used to generate an answer."""
    name: str = Field(description="The name of the source")
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Represents the response from the agent, including the answer and the sources used."""
    answer: str = Field( description="The answer to the user's query")
    sources: list[Source] = Field(default_factory=list, description="A list of sources used to generate the answer")


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
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

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