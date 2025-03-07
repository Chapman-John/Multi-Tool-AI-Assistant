from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
import os
from typing import Optional, List, Dict, Any

# MCP client 
class MultiServerMCPClient:
    """Mock client to simulate connection to different tool servers"""
    def __init__(self, server_name: str):
        self.server_name = server_name
        print(f"Initializing connection to {server_name}")
    
    def run_query(self, query: str) -> str:
        """Simulate sending a query to the tool server"""
        print(f"Sending query to {self.server_name}: {query}")
        # API call and return the response
        return f"Response from {self.server_name} for query: {query}"

# LangChain adapter to convert MCP clients to LangChain tools
class LangChainAdapter:
    """Adapter to convert MCP clients to LangChain-compatible tools"""
    def __init__(self, mcp_client: MultiServerMCPClient, name: Optional[str] = None, description: Optional[str] = None):
        self.mcp_client = mcp_client
        self.name = name or mcp_client.server_name
        self.description = description or f"Use this tool to access {self.name} functionality"
    
    def run(self, query: str) -> str:
        """Method that LangChain will call when the tool is used"""
        return self.mcp_client.run_query(query)

# Set up OpenAI API key and load from environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def create_multi_tool_agent():
    # Initialize MCP clients for each tool
    web_search_mcp = MultiServerMCPClient("web_search_server")
    story_writer_mcp = MultiServerMCPClient("story_writer_server")
    image_generator_mcp = MultiServerMCPClient("image_generator_server")
    
    # Create LangChain tools with proper descriptions
    tools = [
        Tool(
            name="WebSearch",
            func=LangChainAdapter(
                web_search_mcp,
                name="WebSearch",
                description="Useful for searching the web for current information and facts"
            ).run,
            description="Search the web for information and recent facts"
        ),
        Tool(
            name="StoryWriter",
            func=LangChainAdapter(
                story_writer_mcp,
                name="StoryWriter",
                description="Creates creative stories based on provided themes and topics"
            ).run,
            description="Write creative stories on any topic"
        ),
        Tool(
            name="ImageGenerator",
            func=LangChainAdapter(
                image_generator_mcp,
                name="ImageGenerator",
                description="Generates images based on text descriptions"
            ).run,
            description="Create images based on descriptions"
        )
    ]
    
    # Initialize the LLM
    llm = OpenAI(temperature=0.7)
    
    # Initialize the agent with the tools
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

# Main function to handle user requests
def handle_request(user_input: str) -> Dict[str, Any]:
    agent = create_multi_tool_agent()
    try:
        # Process the user input with the agent
        agent_response = agent.run(user_input)
        
        # structure the response to include metadata, status codes, etc.
        result = {
            "status": "success",
            "response": agent_response,
            "error": None
        }
    except Exception as e:
        result = {
            "status": "error",
            "response": None,
            "error": str(e)
        }
    
    return result

# Example usage
if __name__ == "__main__":
    user_request = "Create a short story about the philosopher Spinoza, with an accompanying image, and include some relevant/recent facts."
    result = handle_request(user_request)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {result['error']}")
