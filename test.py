from app.agent import graph 


config = {"configurable": {"thread_id": "user_123"}}

result = graph.invoke(
    {
        "question": "What causes headaches?",
        "history": [],
        "image_description": "",
        "kb": [],
        "web": [],
        "answer": "",
        "route": ""
    },
    config=config
)

print("Final Result:", result)