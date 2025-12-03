"""
Utility functions for audio transcription.
"""
import base64
import os
import shutil
import json
from pydub import AudioSegment
import speech_recognition as sr
from faster_whisper import WhisperModel
import whisper
from pyannote.audio import Pipeline
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap


def trascribe_audio_by_openAI(audio_file: str, openAIClient) -> str:
    # this may be done also with  Whisper + Pyannote
    with open(audio_file, "rb") as audio_file:
        transcript = openAIClient.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=audio_file,
            response_format="diarized_json",
            chunking_strategy="auto", #'VadConfig.'
            # if we have know speaker we can use a sample for matching
            #extra_body={ "known_speaker_names": ["agent"], 
           #             "known_speaker_references": [to_data_url("call_center_voice_sample1.wav")] },
        )

    #print(transcript.segments)
    print ('SERGMENTS:')
    for segment in transcript.segments:
        print(segment.speaker, segment.text, segment.start, segment.end)

    merged = format_transcript(transcript.segments)
    return merged
    #print ('MERGED:')
    #for item in merged:
    #    print(item['speaker'], item['text'], item['start'], item['end'])
 
def format_transcript(segments):
    # Mapping of speaker labels
    merged = []
    current = None
    
    agent_speaker = None
    client_speaker = None

    # the key_word  for identify which speaker is the agent 
    KEYWORD = "nissan"
    # segment format {text:String,start:number,end:number,speaker:string }
    for seg in segments:
        text = seg.text.strip()
        text_lower = text.lower()

        # Detect which speaker mentions Nissan first → assign Agent
        if agent_speaker is None and KEYWORD in text_lower:
            agent_speaker = seg.speaker
            # Assign the other speaker immediately if known
            # (We may not know yet if only one speaker has appeared so far)

        # Once agent is known, assign client when second speaker appears
        if agent_speaker is not None and client_speaker is None and seg.speaker != agent_speaker:
            client_speaker = seg.speaker

        # ---- Assign role for this segment ----
        if seg.speaker == agent_speaker:
            speaker_label = "Agent"
        elif seg.speaker == client_speaker:
            speaker_label = "Client"
        else:
            # Before roles determined, fallback to original
            speaker_label = seg.speaker

        # ---- Merge consecutive segments ----
        if current is None:
            current = {
                "speaker": speaker_label,
                "start": seg.start,
                "end": seg.end,
                "text": text
            }
            continue

        if current["speaker"] == speaker_label:
            current["text"] += " " + text
            current["end"] = seg.end
        else:
            merged.append(current)
            current = {
                "speaker": speaker_label,
                "start": seg.start,
                "end": seg.end,
                "text": text
            }

    # Add last segment
    if current:
        merged.append(current)

    return merged


def to_data_url(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:audio/wav;base64," + base64.b64encode(fh.read()).decode("utf-8")


def query_openai(USER_QUERY,tools, openAIClient):
    GPT_MODEL = "gpt-5"
    input_list = [
    {"role": "user", "content": USER_QUERY}
]        
    response = openAIClient.responses.create(
            model=GPT_MODEL,
            tools=tools,
            input=input_list)
    return response


def trascribe_audio_by_whisper(audio_file: str) -> str:
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio_file, beam_size=5)

    full_text=''
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    
    return full_text


def transcribe_audio_speech_recognistion(audio_file: str) -> str:
    """
    Transcribe audio file and format as support/client dialogue.
    
    Args:
        audio_file: Path to the audio file (e.g., 'call_center_audio_1.mp3')
    
    Returns:
        Formatted transcript with support and client labels
    """
    try:
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Load the audio file
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        
        # Perform transcription using Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        print('whisper',text)

        # Format the transcript
        # This is a basic template - you may need to enhance based on actual audio structure
        
        return text
    
    except FileNotFoundError:
        return f"Error: Audio file '{audio_file}' not found."
    except sr.UnknownValueError:
        return "Error: Could not understand audio."
    except sr.RequestError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error during transcription: {str(e)}"


#---------------------------------------------------------------
# Vector store functions
#---------------------------------------------------------------
def adjust_chunks(documents: list[Document]) -> list[Document]:
    """Create chunks from documents using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
        length_function=len
    )
    all_splits  = text_splitter.split_documents(documents)
    return all_splits

def create_chroma_vector_store(embeddings,chroma_db_path) -> Chroma:
    """Create a Chroma vector store and add transcript segments.

    Args:
        documents: List of documents to add to the vector store
        embeddings: Embeddings model to use for vectorization
        
    Returns:
        Chroma: The vector store instance
    """
    # Check if vector store already exists
    if os.path.exists(chroma_db_path):
        # Load existing vector store
        vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings
        )
        print("Loaded existing vector store from")
    else:
        # Create new vector store if none exists
        vector_store = Chroma(
            collection_name="call_segments",
            embedding_function=embeddings,
            persist_directory=chroma_db_path
        )
        print("Created new vector store")

    return vector_store

def generate_chunks_documents(segments, chunk_size=1000, chunk_overlap=150):
    """
    Convert diarized segments into longer document chunks suitable for embeddings.
    """

    # ---- Step 1: Build full transcript text ----
    full_text = ""
    metadata_blocks = []

    for seg in segments:
        line = f"{seg['speaker']}: {seg['text']}\n"
        full_text += line

        metadata_blocks.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

    # ---- Step 2: Wrap into a single Document ----
    master_doc = Document(
        page_content=full_text.strip(),
        metadata={
            "segments": json.dumps(metadata_blocks) 
        }
    )

    # ---- Step 3: Split into longer chunks ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents([master_doc])

    return chunks


# @todo to be refactored. the documents needs to be longer
def create_chunks(segments):
    docs = [
            Document(
                page_content=item["speaker"] +': '+item["text"],   # text content
                metadata={
                    "speaker": item["speaker"],
                    "start": item["start"],
                    "end": item["end"]
                }
            )
            for item in segments
            ]
    chunks = adjust_chunks(docs)
    return chunks

def add_in_vector_store(vector_store, chunks):
    vector_store.add_documents(chunks)

def print_vectore_store_info(vector_store):
    all_docs = vector_store.get()
    for key, value in all_docs.items():
        print(f'{key}: {value}')


def delete_vector_store(chroma_db_path):
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)

        

#@todo , refactor it- > needs to work with a new set of matches
def query_vector_store(vector_store,query, speaker_target:str = "Any", top_k: int = 3):
    # speaker_target: "Client" | "Agent"
    filt = {"speaker": speaker_target} if speaker_target != "Any" else None
    print('similarity_search query:',query)
    results = vector_store.similarity_search(
        query,
        k=3,
        #k=top_k,
        #filter=filt
    )

    return {
        "query": query,
        "speaker": speaker_target,
        "matches": [
            {
                "text": r.page_content,
                "speaker": r.metadata.get("speaker"),
                "start": r.metadata.get("start"),
                "end": r.metadata.get("end")
            }
            for r in results
        ]
    }

#@todo , refactor it- > needs to work with a new set of matches
def query_vector_store_V2(vector_store,query, speaker_target:str = "Any", top_k: int = 3):
    # speaker_target: "Client" | "Agent"
    filt = {"speaker": speaker_target} if speaker_target != "Any" else None
    print('similarity_search query:',query)
    results = vector_store.similarity_search(
        query,
        k=3,
        #k=top_k,
        #filter=filt
    )

    return results

def create_rag_chain(llm, retriever):
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful analyst. 
        Use the following context to answer the question.
    
        Respond in JSON format.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
                                                     
        Context:
        {context}

        Question:
        {question}
        """)
    
    rag_chain = (
        RunnableMap({
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
    )
    return rag_chain