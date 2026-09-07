from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

def run_reasearch_pipeline(topic: str) -> dict:

    state = {}
    # search agent working 

    print("\n"+"="*50)
    print("step-1 - search agent is working...")
    print("="*50)

    search_agent = build_search_agent()
    search_result =search_agent.invoke({
        "messages" : [("user" , f"Search for recent and reliable information on the topic: {topic}")]

    })
    state["search_results"] = search_result["messages"][-1].content

    print("\n search_result",state["search_results"])


    # step-2 reader agent 

    print("\n"+"="*50)
    print("step-2 - reader agent is scraping top resources...")
    print("="*50)


    reader_agent = build_reader_agent()
    reader_result =reader_agent.invoke({
        "messages": [("user", 
          f"Based on the following search results about '{topic}', "
          f"pick the most relevant URL and scrape it for deeper content.\n\n"
          f"Search Results:\n{state['search_results'][:800]}"
       )]

    })

    state['scraped_content'] = reader_result["messages"][-1].content

    print("scraped_content: \n", state['scraped_content'])

    # step-3 writer agent 

    print("\n"+"="*50)
    print("step-3 - writer agent is generating the final report...")
    print("="*50)
    
    reasearch_combined = (
        f"SEARCH_RESULTS :\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED_CONTENT :\n{state['scraped_content']}"
    )

    
    state["report"] =writer_chain.invoke({
        "topic":topic,
        "research": reasearch_combined
    })

    print("\n Final Report\n", state["report"])


    # critic report 
    print("\n"+"="*50)
    print("step-4 - critic agent is reviewing the report...")
    print("="*50)

    state['feedback'] =critic_chain.invoke({
        "report" : state["report"],
        
    })


    print("\n Feedback from critic agent\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\n Enter the research topic : ")
    run_reasearch_pipeline(topic)


