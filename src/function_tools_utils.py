import json
from function_tools import get_openai_tools, query_vector_store

def getResponseByFunctionCalling(openAiclient,vector_store):
    user_question = "Return the client's name in JSON format."

# ----------------------------------------
# define openAi function calling
# ----------------------------------------
    tools = get_openai_tools()

# ----------------------------------------
# get dynamic function call
# ----------------------------------------
    response = openAiclient.responses.create(
        model="gpt-4.1",
        input=user_question,
        tools=tools
    )

#print("Raw output items from first call:")
#print (response)
    for item in response.output:
        #print the function calling name
        print("-", item.type,":",getattr(item, "name", None))

#Try to find a function_call
    tool_call = next(
        (item for item in response.output if item.type == "function_call"),
        None,
    )


    if tool_call is None:
        # Model answered directly, so now output_text is valid
        print("Final answer (no tool used):", response.output_text)
    else:
        # 2) Run your local Python function using the arguments from the tool call
        args = json.loads(tool_call.arguments or "{}")
        print ('- function args:',args)
        
        tool_result = query_vector_store(
            vector_store=vector_store,
            query=args.get("query", ""),
            speaker_target=args.get("speaker", "Any"),
            top_k=int(args.get("top_k", 3)),
        )

        # 3) Second call: send tool call + its output back to the model
        followup_input = [
            {"role": "user", "content": user_question},
            tool_call,  # the function_call item we got from the model
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": [{
                    "type": "input_text",   # REQUIRED
                    "text": json.dumps(tool_result)
            }], 
            },
        ]

        print('tool_result',tool_result)

        response2 = openAiclient.responses.create(
            model="gpt-4.1",
            input=followup_input,
        )

        # Now the model has the tool result and will normally respond with text → output_text is populated
        print("Final answer (after tool):", response2.output_text)