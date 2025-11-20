def get_openai_tools():
    tools=[{
        "type": "function",
        "name": "query_vector_store",
        "description": "Search transcript based on a query and optional speaker filter.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "speaker": {
                    "type": "string",
                    "enum": ["Agent", "Client", "Any"],
                    "description": "Filter results by speaker"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max number of matching segments to return",
                    "default": 3,
                },
            },
            "required": ["query"]
        }
    }]
    return tools