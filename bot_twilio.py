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
from pipecat.frames.frames import TTSSpeakFrame
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
from pipecat.services.groq.llm import GroqLLMService
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

    # Insurance advisor greeting
    greeting_text = "আসসালামু আলাইকুম স্যার! আমি আপনার বীমা উপদেষ্টা। আপনার এক থেকে দুই মিনিট সময় নিতে পারি?"
    # Note: Using TTSSpeakFrame through pipeline for immediate greeting

    # Groq LLM for ultra-fast conversation (3-5x faster than OpenAI)
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",  # Recommended replacement for mixtral (280 tokens/sec)
        params=GroqLLMService.InputParams(
            max_tokens=256,  # Limit response length for faster speech
            temperature=0.5,  # Lower for faster, more deterministic responses
        ),
    )

    # Optimized English prompt for better LLM understanding (responses will be in Bangla)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an insurance advisor based in Dhaka, Bangladesh. You sell Life, Health, Personal Accident, and Travel Insurance.\n\n"

                "LANGUAGE: Always respond in Bangladeshi Bangla (bn-BD). Use local Dhaka dialect. Address customer as 'স্যার' or 'আপনি'.\n\n"

                "STYLE:\n"
                "• Polite, empathetic, respectful tone\n"
                "• End important questions with: 'কি জানতে পারি স্যার', 'জানানো যাবে কি স্যার', 'একটু শেয়ার করা যাবে স্যার'\n"
                "• Never be rude or pushy\n\n"

                "KEY RULES:\n"
                "• Numbers: NEVER say digit-by-digit. Always speak full words in Bangla: 1000='এক হাজার', 50000='পঞ্চাশ হাজার', 100000='এক লক্ষ', 300='তিনশ', 500='পাঁচশ'\n"
                "• Accident insurance: Annual premium 300-500 taka gives 1-3 lakh coverage for death/permanent disability. Then ask about job/commute/family.\n"
                "• Data privacy: Before asking personal info, assure 'তথ্য শুধু premium হিসাবে ব্যবহার হবে, বাইরে শেয়ার নয়। range দিলেই চলবে।'\n"
                "• Repeat: If customer doesn't understand, patiently re-explain shorter. Ask 'এইবার পরিষ্কার স্যার?'\n\n"

                "CONVERSATION FLOW:\n"
                "1. Get permission (ask for 1-2 minutes)\n"
                "2. Build rapport: ask profession, income range, family dependents (one at a time)\n"
                "3. Understand priority: hospital/accident coverage vs future savings vs both\n"
                "4. Suggest 2-3 products: small health/accident cover (few hundred taka/year) OR monthly premium life+savings plan (10-20 year term)\n\n"

                "OBJECTION HANDLING:\n"
                "• No money → Just 2-4% of monthly income protects family from big risks\n"
                "• Office has medical → Office medical ends with job, personal plan is lifetime\n"
                "• Already have policy → Can add top-up to fill gaps\n\n"

                "CLOSING: Ask for name + WhatsApp/email, offer to send written details + premium calculation.\n\n"

                "⚡ CRITICAL PHONE CALL RULE: Keep EVERY response extremely short - max 1-2 sentences (40 words). Never give long explanations in one turn. Break into small conversational chunks.\n"
                "Example: ❌'শুভদিন স্যার, আমি [নাম], ঢাকা-ভিত্তিক বীমা কোম্পানির অ্যাডভাইজার...' ✅'ধন্যবাদ স্যার। লাইফ ইনসুরেন্স নিয়ে কল করেছি। দুই মিনিট সময়?'"
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
        # Wait a moment for pipeline to be ready
        await asyncio.sleep(0.1)

        # Use TTSSpeakFrame to play greeting through normal pipeline
        # This ensures proper timing and no StartFrame issues
        await task.queue_frame(TTSSpeakFrame(text=greeting_text))
        logger.info("Greeting queued for TTS")

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
