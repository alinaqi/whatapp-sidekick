from fastapi import FastAPI, Request, Response, Form, HTTPException, File, UploadFile
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import openai
import os
from dotenv import load_dotenv
from typing import Optional
import redis
import json
import logging
from datetime import datetime
import numpy as np
import io
from pydub import AudioSegment
import tempfile
import uuid
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField,
    TagField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.query import Query

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Twilio credentials
ACCOUNT_SID = '<YOUR_TWILIO_ACCOUNT_SID>'
AUTH_TOKEN = '<YOUR_TWILIO_AUTH_TOKEN>'
TWILIO_NUMBER = '<YOUR_TWILIO_NUMBER>'
API_KEY_SID = '<YOUR_TWILIO_API_KEY_SID>'
API_KEY_SECRET = '<YOUR_TWILIO_API_KEY_SECRET>'
TWIML_APP_SID = '<YOUR_TWILIO_TWIML_APP_SID>'

openai.api_key = "<YOUR_OPENAI_API_KEY>"

# Redis configuration
REDIS_HOST = '<YOUR_REDIS_HOST>'
REDIS_PORT = <YOUR_REDIS_PORT>
REDIS_PASSWORD = '<YOUR_REDIS_PASSWORD>'

# Constants for Redis search
INDEX_NAME = "sidekick_index"
PREFIX = "doc:"
VECTOR_DIM = 1536  # Dimension of text-embedding-3-small

# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# Initialize Twilio client
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

def create_index(client):

    # Define the index fields
    schema = (
        TextField("content"),
        TextField("summary"),
        TextField("phone_number"),  # Store phone number as TextField
        VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"})
    )

    # Create the index
    client.ft(INDEX_NAME).create_index(
        fields=schema,
        definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )
    logger.info(f"Index {INDEX_NAME} created successfully")

# Make sure to call this function to recreate the index
create_index(redis_client)


def check_and_register_user(phone_number):
    user_key = f"sidekick_user:{phone_number}"
    if not redis_client.exists(user_key):
        user_data = {
            "phone_number": phone_number,
            "registration_date": str(datetime.now())
        }
        redis_client.set(user_key, json.dumps(user_data))
        logger.info(f"New user registered: {phone_number}")
        return True
    return False

def get_openai_response(message):
    try:
        logger.info(f"Getting OpenAI response for message: {message}")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a SideKickler. Summarize the information requested by the user."},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting OpenAI response: {str(e)}")
        return "I'm sorry, I'm having trouble processing your request right now."

def summarize_information(information):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "Summarize the data below and return in json. If it's a url, summarize the information in the url."},
                {"role": "user", "content": f"input: {information}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return None

def get_embedding(text):
    response = openai.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def save_to_redis(phone_number, content):
    try:
        print(f"Saving information for phone number: {phone_number}")
        # Generate a unique key
        key = f"{PREFIX}{redis_client.incr('doc:id')}"

        summary = summarize_information(content)

        # Get the embedding
        embedding = get_embedding(summary)

        # Prepare the document
        doc = {
            "content": content,
            "summary": summary,
            "phone_number": phone_number,  # Save the phone number directly without the "phone:" prefix
            "embedding": np.array(embedding).astype(np.float32).tobytes()
        }

        # Save to Redis
        redis_client.hset(key, mapping=doc)
        logger.info(f"Document saved with key: {key}")
        return True
    except Exception as e:
        logger.error(f"Error saving information to Redis: {str(e)}")
        return False

def search_documents(query_text, phone_number):
    # Get the embedding for the query
    query_embedding = get_embedding(query_text)


    # Prepare the query, filtering by phone number and using KNN search
    base_query = f"(@phone_number:{phone_number})=>[KNN 5 @embedding $embedding AS score]"

    print(f"Base query: {base_query}")
    query = (
        Query(base_query)
        .return_fields("content", "summary", "phone_number", "score")
        .sort_by("score")  # Sort by score
        .dialect(2)
    )

    logger.info(f"Executing query: {query.query_string()}")

    # Search
    try:
        results = redis_client.ft(INDEX_NAME).search(
            query,
            query_params={"embedding": np.array(query_embedding).astype(np.float32).tobytes()}
        )
        logger.info(f"Search results: {results}")
        return [(doc.content, doc.summary, doc.phone_number, 1 - float(doc.score)) for doc in results.docs]
    except redis.exceptions.ResponseError as e:
        logger.error(f"Search error: {e}")
        return []
  

def summarize_results(results):
    logger.info("Summarizing results")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Review the results below from our vector database and return the best possible result based on the highest similarity score. Return result in a user friendly way. Also add your own summary. Show links separately as Links: <link>. This will be sent as whatsapp message"},
            {"role": "user", "content": f"Results: {results}"}
        ]
    )
    return response.choices[0].message.content.strip()

def categorize_query(message):
    try:
        logger.info(f"Categorize incoming query: {message}")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", 
                 "content": 
                 """
                 Categorize incoming query or a link into either save_information or retrieve_information. 
                 Usually, retrieve_information will be a query like "what did I share with you..", "what was the link ..." or something similar. 
                 Information save would be a link or text or image or video etc. 
                 Return as json format: 
                 { 
                    'category': 'save_information', 
                    'query': 'the query', 
                    'link': 'any link in the query'
                }
                 """
                 },
                {"role": "user", "content": f"user query: {message}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting OpenAI response: {str(e)}")
        return "I'm sorry, I'm having trouble processing your request right now."

@app.post("/test_webhook_text")
async def test_webhook_text(
    query: str = Form(...),
    from_number: Optional[str] = Form(None)
):
    return process_message(query, from_number)

@app.post("/test_webhook_audio")
async def test_webhook_audio(
    audio_file: UploadFile = File(...),
    from_number: Optional[str] = Form(None)
):
    try:
        content = await audio_file.read()
        
        # Convert to MP3
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            audio = AudioSegment.from_file(io.BytesIO(content), format=audio_file.filename.split('.')[-1])
            audio.export(tmp_file.name, format="mp3")
            
        with open(tmp_file.name, "rb") as audio_file_to_transcribe:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file_to_transcribe
            )
        
        incoming_msg = transcript.text
        logger.info(f"Transcribed audio: {incoming_msg}")
        return process_message(incoming_msg, from_number)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

def process_message(incoming_msg: str, from_number: Optional[str]):
    if incoming_msg.lower() == 'ping':
        response_text = "PONG"
    else:
        # categorize the query
        query_category = json.loads(categorize_query(incoming_msg))
        logger.info(f"Query category: {query_category}")
        category = query_category.get('category', '')

        if category == 'save_information':
            if save_to_redis(from_number, incoming_msg):
                response_text = f"Information saved successfully."
            else:
                response_text = "Sorry, there was an error saving your information. Please try again."
        elif category == 'retrieve_information':
            results = search_documents(incoming_msg, from_number)
            if results:
                response_text = summarize_results(results)
            else:
                response_text = "I couldn't find any information matching your query. Can you try rephrasing or provide more details?"
        else:
            response_text = get_openai_response(incoming_msg)

    return {"response": response_text}

@app.post("/webhook")
async def webhook(request: Request):
    form_data = await request.form()
    incoming_msg = form_data.get('Body', '')
    conversation_sid = form_data.get('ConversationSid')
    
    # Extract sender information
    author = form_data.get('Author', '')
    from_number = author.replace('whatsapp:', '') if author.startswith('whatsapp:') else author
    
    logger.info(f"Received message: {incoming_msg} from {from_number} in conversation: {conversation_sid}")
    logger.info(f"Full form data: {form_data}")
    
    # Check and register user
    is_new_user = check_and_register_user(from_number)
    
    if is_new_user:
        logger.info("New user registered")
        response_text = "Welcome! You've been registered. If you need to save a link or information just send it to me. If you have a question, just ask! It's really that easy!"
    else:
        response_text = process_message(incoming_msg, from_number)["response"]

    try:
        message = twilio_client.conversations \
                        .v1 \
                        .conversations(conversation_sid) \
                        .messages \
                        .create(author='system', body=response_text)
        logger.info(f"Message sent: {message.sid}")
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")

    return Response(content="OK", media_type="text/plain")

@app.get("/")
async def root():
    return {"message": "WhatsApp Bot is running!"}