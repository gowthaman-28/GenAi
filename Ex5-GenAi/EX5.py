import os
from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler


# ✅ (Optional) Hardcode Langfuse keys here
os.environ["LANGFUSE_PUBLIC_KEY"] = "your_public_key"
os.environ["LANGFUSE_SECRET_KEY"] = "your_secret_key"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"   # or your host


# ✅ Initialize Langfuse
langfuse_client = Langfuse()
langfuse_handler = CallbackHandler()


# ✅ Define State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ✅ Hardcoded Groq API Key
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key="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"   # 🔴 Replace with your real key
)


# ✅ Chatbot Node
def chatbot(state: State, config: RunnableConfig):
    response = llm.invoke(state["messages"], config=config)
    return {"messages": [response]}


# ✅ Main Function
def run_langfuse_demo():
    print("Building LangGraph Pipeline with Langfuse Tracing...")

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    app = graph_builder.compile()

    print("Pipeline built successfully!\n")
    print("Welcome to the LangGraph + Langfuse Chatbot!")
    print("Type 'quit' or 'exit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")

            if user_input.lower() in ["quit", "exit"]:
                break

            if not user_input.strip():
                continue

            # ✅ Attach Langfuse callback
            config = {"callbacks": [langfuse_handler]}

            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="updates",
                config=config
            )

            for event in events:
                for node_name, state_update in event.items():
                    latest_message = state_update["messages"][-1]

                    if latest_message.type == "ai":
                        print(f"Groq Bot: {latest_message.content}")

        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    # ✅ Flush Langfuse logs
    print("\nFlushing traces to Langfuse...")
    try:
        langfuse_client.flush()
    except Exception as e:
        print(f"Error flushing (check Langfuse keys): {e}")

    print("Goodbye!")


# ✅ Run
if __name__ == "__main__":
    run_langfuse_demo()