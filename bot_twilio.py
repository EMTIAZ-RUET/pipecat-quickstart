"""
Pipecat Twilio Bot with Bangla Language Support

This bot handles Twilio outbound calls with Google Cloud STT, Gemini TTS,
and OpenAI LLM, configured for Bangladeshi Bangla (bn-BD).
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame, LLMFullResponseEndFrame, OutputAudioRawFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GeminiTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transcriptions.language import Language
from google.cloud import texttospeech
import json
import asyncio

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Read Google Cloud credentials
credentials_path = os.getenv("GOOGLE_TEST_CREDENTIALS")
with open(credentials_path, "r") as f:
    credentials_json = f.read()


def precompute_greeting_audio(text: str, credentials_json: str) -> bytes:
    """Precompute greeting audio using Google Gemini TTS synchronously.

    Returns raw PCM audio bytes at 8kHz mono for Twilio.
    """
    logger.info(f"Precomputing greeting audio for: {text}")

    # Parse credentials
    credentials_dict = json.loads(credentials_json)

    # Create TTS client
    client = texttospeech.TextToSpeechClient.from_service_account_info(credentials_dict)

    # Configure synthesis input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Configure voice (Bangladeshi Bangla)
    # Let Google select the best available voice for bn-BD
    voice = texttospeech.VoiceSelectionParams(
        language_code="bn-BD"
        # Don't specify name - let Google pick available voice
    )

    # Configure audio (8kHz for Twilio, LINEAR16 PCM)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,  # Twilio's sample rate
        speaking_rate=1.0
    )

    # Perform TTS synthesis
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    logger.info(f"Precomputed audio size: {len(response.audio_content)} bytes")
    return response.audio_content


async def run_bot(transport: BaseTransport, handle_sigint: bool):
    """Run the bot pipeline with Bangla language support."""

    # Google STT for Bangla speech recognition
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

    # Gemini TTS for Bangla speech synthesis
    # Note: Gemini natively outputs at 24kHz, pipeline will resample to 8kHz
    tts = GeminiTTSService(
        model="gemini-2.5-flash-tts",
        voice_id="Kore",
        language_code="bn-BD",
        credentials=credentials_json,
        prompt="Speak naturally and conversationally in Bangladeshi Bangla with a friendly, warm tone.",
        sample_rate=24000,  # Gemini's native sample rate
    )

    # Precompute greeting audio to avoid delay
    greeting_text = "আসসালামু আলাইকুম! আমি একটি এআই সহকারী। আপনাকে কীভাবে সাহায্য করতে পারি?"
    precomputed_greeting = precompute_greeting_audio(greeting_text, credentials_json)

    # OpenAI LLM for conversation
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly AI assistant making an outbound phone call in Bangladeshi Bangla language. "
                "Your responses will be read aloud, so keep them concise and conversational. "
                "Always respond in Bangla (Bengali). Begin by politely greeting the person and introducing yourself."
            ),
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
            audio_in_sample_rate=8000,   # Twilio input is 8kHz
            audio_out_sample_rate=8000,  # Twilio output is 8kHz (will auto-resample from 24kHz)
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
            enable_transcription_timing_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Starting outbound call conversation in Bangla")
        # Play precomputed greeting audio immediately
        # Signal bot started speaking to prevent interruptions
        await task.queue_frames([
            BotStartedSpeakingFrame(),
            OutputAudioRawFrame(audio=precomputed_greeting, sample_rate=8000, num_channels=1),
            BotStoppedSpeakingFrame(),
            LLMFullResponseEndFrame()
        ])
        logger.info("Precomputed greeting sent")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Outbound call ended")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for Twilio WebSocket connections."""

    # Parse Twilio WebSocket connection
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    # Access custom stream parameters passed from TwiML
    body_data = call_data.get("body", {})
    to_number = body_data.get("to_number")
    from_number = body_data.get("from_number")
    language = body_data.get("language", "bn-BD")

    logger.info(f"Call metadata - To: {to_number}, From: {from_number}, Language: {language}")

    # Create Twilio serializer
    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    # Create FastAPI WebSocket transport with Twilio configuration
    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,
                    start_secs=0.1,
                    confidence=0.6,
                )
            ),
            serializer=serializer,
        ),
    )

    handle_sigint = runner_args.handle_sigint

    await run_bot(transport, handle_sigint)
