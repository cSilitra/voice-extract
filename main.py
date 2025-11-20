import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from src.utils import trascribe_audio_by_openAI,create_chroma_vector_store, delete_vector_store,add_in_vector_store,query_vector_store,print_vectore_store_info,create_chunks,generate_chunks_documents
from src.function_tools import get_openai_tools

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
file_path = "call_center_audio_1.mp3"
chroma_db_path="./chroma_db"

# 0. Init
openAiclient = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# ----------------------------------------
# get transcription from audio using OpenAI
# ----------------------------------------
#transcript_list = trascribe_audio_by_openAI(file_path,openAiclient)
transcript_list = [{'speaker': 'Agent', 'start': 8.052, 'end': 10.951999999999998, 'text': 'Thank you for calling Nissan. My name is Lauren. Can I have your name?'}, {'speaker': 'Client', 'start': 11.152, 'end': 12.602, 'text': 'Yeah, my name is John Smith.'}, {'speaker': 'Agent', 'start': 14.001999999999999, 'end': 15.251999999999999, 'text': 'Thank you, John. How can I help you?'}, {'speaker': 'Client', 'start': 15.902000000000001, 'end': 19.902, 'text': 'I was just calling about to see how much it would cost to update the map in my car.'}, {'speaker': 'Agent', 'start': 20.402, 'end': 23.902, 'text': "I'd be happy to help you with that today. Did you receive a mailer from us?"}, {'speaker': 'Client', 'start': 23.902, 'end': 26.102, 'text': 'I did. Do you need the customer number?'}, {'speaker': 'Agent', 'start': 26.252, 'end': 26.901999999999997, 'text': 'Yes, please.'}, {'speaker': 'Client', 'start': 27.151999999999997, 'end': 29.901999999999997, 'text': "Okay, it's 15243."}, {'speaker': 'Agent', 'start': 30.551999999999996, 'end': 33.152, 'text': 'Thank you and the year, make and model of your vehicle.'}, {'speaker': 'Client', 'start': 33.30199999999999, 'end': 36.501999999999995, 'text': 'Yeah, I have a 2009 Nissan Altima.'}, {'speaker': 'Agent', 'start': 37.3, 'end': 38.05, 'text': 'Oh, nice car.'}, {'speaker': 'Client', 'start': 38.349999999999994, 'end': 40.25, 'text': 'Yeah, thank you. We really enjoy it.'}, {'speaker': 'Agent', 'start': 40.699999999999996, 'end': 46.05, 'text': 'Okay, I think I found your profile here. Can I have you verify your address and phone number, please?'}, {'speaker': 'Client', 'start': 46.3, 'end': 57.3, 'text': "Yes, it's 1255 North Research Way. That's in Orem, Utah, 84097. And my phone number is 801-431-1000."}, {'speaker': 'Agent', 'start': 58.05, 'end': 73.95400000000001, 'text': 'Thanks, John. I located your information. The newest version we have available for your vehicle is version 7.7, which was released in March of... 2012. The price of the new map is $99 plus shipping and tax. Let me go ahead and set up this order for you.'}, {'speaker': 'Client', 'start': 74.70400000000001, 'end': 78.404, 'text': "Well, can we wait just a second? I'm not really sure if I can afford it right now."}, {'speaker': 'Agent', 'start': 79.104, 'end': 90.952, 'text': "Alright, well here are a few reasons to consider purchasing today. It looks as though you haven't updated your vehicle for three years, so that would be the equivalent of getting three years worth of updates for the price of one."}, {'speaker': 'Client', 'start': 90.952, 'end': 91.702, 'text': 'Oh, okay.'}, {'speaker': 'Agent', 'start': 92.102, 'end': 101.902, 'text': "In addition, special offers like the current promotion don't come around too often. I would definitely recommend taking advantage of the extra $50 off before it expires."}, {'speaker': 'Client', 'start': 102.30199999999999, 'end': 103.852, 'text': 'Yeah, that does sound pretty good.'}, {'speaker': 'Agent', 'start': 104.502, 'end': 112.968, 'text': "If I set this order up for you now, it'll ship out today and for $50 less. Do you have your credit card handy and I can place this order for you now?"}, {'speaker': 'Client', 'start': 113.41799999999999, 'end': 118.268, 'text': "Yeah, let's go ahead and use your visa. My number is..."}]
#print ('transcript',transcript_list)

delete_vector_store(chroma_db_path)

# ----------------------------------------
#create/get vector store
# ----------------------------------------
vector_store = create_chroma_vector_store(embeddings,chroma_db_path)

#define retriver - not used
#retriever = vector_store.as_retriever()

# ----------------------------------------
#create documents for vector store from list
# ----------------------------------------
chunks = create_chunks(transcript_list)
#chunks = generate_chunks_documents(transcript_list)
#print (chunks)

# ----------------------------------------
#add chunks in vector store
# ----------------------------------------
add_in_vector_store(vector_store,chunks)
#print_vectore_store_info(vector_store)

# ----------------------------------------
# define openAi function calling
# ----------------------------------------
tools = get_openai_tools()

# ----------------------------------------
# define query for creating
# ----------------------------------------
user_question = "Return the customer's car modal in JSON."

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
    print(" -", item.type,":",getattr(item, "name", None))

# Try to find a function_call
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
        }],  # can be dict; SDK will JSON-serialize
        },
    ]

    print('tool_result',tool_result)

    response2 = openAiclient.responses.create(
        model="gpt-4.1",
        input=followup_input,
    )

    # Now the model has the tool result and will normally respond with text → output_text is populated
    print("Final answer (after tool):", response2.output_text)