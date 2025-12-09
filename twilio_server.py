"""
Twilio WebSocket Server for Pipecat Voice Bot

This FastAPI server handles Twilio WebSocket connections for both inbound and outbound calls.
Run this server and expose it via ngrok or deploy it to a cloud service.
"""

import os
import sys
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

# Import Pipecat components
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GeminiTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.server import WebsocketServerParams, WebsocketServerTransport
from pipecat.transcriptions.language import Language

load_dotenv(override=True)

app = FastAPI()

# Read Google Cloud credentials
credentials_path = os.getenv("GOOGLE_TEST_CREDENTIALS")
with open(credentials_path, "r") as f:
    credentials_json = f.read()


@app.get("/")
async def root():
    return {"message": "Pipecat Twilio Voice Bot Server is running!"}


@app.post("/incoming")
async def incoming_call(request: Request):
    """Handle incoming Twilio calls"""
    logger.info("Incoming call received")

    # Get the WebSocket URL (you'll need to replace this with your actual server URL)
    host = request.headers.get("host")
    protocol = "wss" if request.url.scheme == "https" else "ws"
    websocket_url = f"{protocol}://{host}/ws"

    # Create TwiML response
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=websocket_url)
    stream.parameter(name="call_type", value="inbound")
    stream.parameter(name="language", value="bn-BD")
    connect.append(stream)
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle Twilio WebSocket connections"""
    await websocket.accept()

    logger.info("WebSocket connection accepted from Twilio")

    # Twilio uses 8kHz audio
    transport = WebsocketServerTransport(
        websocket=websocket,
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,
                    start_secs=0.1,
                    confidence=0.6,
                )
            ),
            serializer=TwilioFrameSerializer(),
        ),
    )

    # Initialize services
    stt = GoogleSTTService(
        params=GoogleSTTService.InputParams(
            languages=Language.BN,
            model="chirp_3",
            enable_automatic_punctuation=True,
            enable_interim_results=True,
            enable_voice_activity_events=True,
        ),
        credentials=credentials_json,
        location="us",
    )

    tts = GeminiTTSService(
        model="gemini-2.5-flash-tts",
        voice_id="Kore",
        language_code="bn-BD",
        credentials=credentials_json,
        prompt="Speak naturally and conversationally in Bangladeshi Bangla with a friendly tone.",
        sample_rate=8000,  # Twilio uses 8kHz
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant on a phone call. Respond naturally and keep your answers conversational. Always respond in Bangla (Bengali) language.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,  # Twilio audio rate
            audio_out_sample_rate=8000,  # Twilio audio rate
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected to voice bot")
        # Greet the caller
        messages.append(
            {
                "role": "system",
                "content": "Say hello and briefly introduce yourself in Bangla.",
            }
        )
        await task.queue_frame(transport.output())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Twilio WebSocket server...")
    logger.info("To expose locally, use ngrok:")
    logger.info("  ngrok http 8765")
    logger.info("")
    logger.info("Then update your Twilio webhook URL to:")
    logger.info("  https://your-ngrok-url.ngrok.io/incoming")

    uvicorn.run(app, host="0.0.0.0", port=8765)
