import os
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ✅ Define State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ✅ Tools (FIXED: Added docstrings)
@tool
def get_weather(location: str):
    """Get the current weather for a given location."""
    return f"The weather in {location} is currently 72°F and sunny."


@tool
def calculate_sum(a: int, b: int):
    """Calculate the sum of two numbers."""
    return str(a + b)


# Tool list
tools = [get_weather, calculate_sum]
tool_node = ToolNode(tools)


# ✅ Hardcoded Groq API Key
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.0,
    groq_api_key="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"   # 🔴 Replace with your actual key
)

# Bind tools
llm_with_tools = llm.bind_tools(tools)


# ✅ Chatbot Node
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ✅ Main Function
def run_langgraph_tools_demo():
    print("Building LangGraph Pipeline with Tools...")

    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    # Flow
    graph_builder.add_edge(START, "chatbot")

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    graph_builder.add_edge("tools", "chatbot")

    # Compile
    app = graph_builder.compile()

    print("Pipeline built successfully!\n")
    print("Welcome to the LangGraph Groq Chatbot with Tools!")
    print("Try asking for the weather or doing some math.")
    print("Type 'quit' or 'exit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")

            # Exit condition
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            # Skip empty input
            if not user_input.strip():
                continue

            # Run graph
            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="updates"
            )

            # Process output
            for event in events:
                for node_name, state_update in event.items():

                    latest_message = state_update["messages"][-1]

                    # Tool call
                    if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                        for tc in latest_message.tool_calls:
                            print(f"-> [Agent is calling tool '{tc['name']}' with args {tc['args']}]")

                    # Tool output
                    elif isinstance(latest_message, ToolMessage):
                        print(f"<- [Tool returned: {latest_message.content}]")

                    # Final response
                    elif latest_message.type == "ai" and latest_message.content:
                        print(f"Groq Bot: {latest_message.content}")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# ✅ Run
if __name__ == "__main__":
    run_langgraph_tools_demo()