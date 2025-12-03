def get_openai_tools():
    tools=[{
        "type": "function",
        "name": "query_vector_store",
        "description": "Search transcript and return client name",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
    }]
    return tools