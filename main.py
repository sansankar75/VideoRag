from agent.rag_agent import build_agent

if __name__ == "__main__":
    agent = build_agent()

    result = agent.run("How many chairs are present in the video?")
    print("\nFINAL ANSWER:", result)
