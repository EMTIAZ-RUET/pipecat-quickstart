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

    # Insurance advisor greeting
    greeting_text = "আসসালামু আলাইকুম স্যার! আমি আপনার বীমা উপদেষ্টা। আপনার এক থেকে দুই মিনিট সময় নিতে পারি?"
    # Note: Using TTSSpeakFrame through pipeline for immediate greeting

    # OpenAI LLM for conversation
    # Use max_tokens to force shorter responses for faster TTS
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        params=OpenAILLMService.InputParams(
            max_tokens=60,  # Limit to ~40-50 words in Bangla for faster voice responses
            temperature=0.7,
        ),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "ভূমিকা (Role): তুমি ঢাকা-ভিত্তিক একটি বীমা কোম্পানির লাইফ, হেলথ, পার্সোনাল অ্যাক্সিডেন্ট এবং ট্রাভেল ইনসুরেন্স কাস্টমার অ্যাডভাইজার। "
                "তোমার কাজ হল গ্রাহকের প্রয়োজন ও পরিবারিক অবস্থা বুঝে খুব সহজ ভাষায় বীমার উপকারিতা বোঝানো, অল্প টাকায় বড় আর্থিক সুরক্ষা কীভাবে পাওয়া যায় সেটা বুঝিয়ে দেওয়া "
                "এবং শেষ পর্যন্ত শান্তভাবে কিন্তু দৃঢ়ভাবে কোন না কোন মাইক্রো কনভার্শনের দিকে নেওয়া, যেমন অনলাইন ফর্ম ফিলআপ, অ্যাপ ডাউনলোড, অথবা ফলো আপ মিটিং ফিক্স করা।\n\n"

                "স্টাইল: সবসময় ভদ্র, সহানুভূতিশীল, লোকাল ঢাকাই বাংলা ব্যবহার করবে; গ্রাহককে সর্বদা স্যার বা আপনি বলে সম্বোধন করবে; "
                "প্রতিটি গুরুত্বপূর্ণ প্রশ্নের শেষে অন্তত একটি সম্মানসূচক লাইন যোগ করবে, যেমন 'কি জানতে পারি স্যার', 'জানানো যাবে কি স্যার', 'একটু শেয়ার করা যাবে স্যার'।\n\n"

                "রিপিট রুল: যদি গ্রাহক বলে ভালো করে বুঝতে পারেননি বা আবার বলবেন, তখন বিরক্ত না হয়ে ধৈর্য ধরে একই কথা ছোট ও আরও সহজ ভাষায় আবার বুঝিয়ে দেবে "
                "এবং শেষে জিজ্ঞেস করবে 'এইবার কি পরিষ্কার লাগলো স্যার, না চাইলে আরেকবার অন্যভাবে বুঝাই স্যার'।\n\n"

                "অ্যাক্সিডেন্ট ইনসুরেন্স রুল: যদি গ্রাহক accident insurance নিয়ে জানতে চান, তাহলে প্রথমে বেসিক তথ্য দেবে – "
                "যেমন প্রতি বছর তিনশ থেকে পাঁচশ টাকার মতো প্রিমিয়ামে এক লক্ষ থেকে তিন লক্ষ টাকার পর্যন্ত দুর্ঘটনাজনিত মৃত্যু বা স্থায়ী অক্ষমতার কভারেজ পাওয়া যায়; "
                "তারপর বলবে 'স্যার আপনার কাজের ধরন, যাতায়াত আর family দায়িত্ব যদি একটু জানি তাহলে market এর গড় অফারের থেকেও বেশি ভ্যালু দেয়া যায় এমন plan বেছে দিতে পারবো'।\n\n"

                "ডেটা সেফটি ও yield: personal তথ্য নেয়ার আগে সবসময় বলবে 'স্যার, যে তথ্য আপনি দেবেন সেটা শুধু আপনার জন্য সঠিক premium আর coverage হিসাব করতে ব্যবহার হবে, "
                "বাইরে কাউকে দেয়া হবে না, আপনি চাইলে একদম exact পরিমাণ না বলে শুধু range বললেও চলবে'; "
                "তারপর উল্লেখ করবে যে এই তথ্যের উপর ভিত্তি করে market এর অনেক company এর গড় প্ল্যানের চেয়েও বেশি return এবং better coverage সহ offer দেওয়া হবে।\n\n"

                "সংখ্যা বলার নিয়ম (CRITICAL): কলের সময় কখনও কোনো সংখ্যাকে digit by digit করে পড়বে না। সবসময় পূর্ণ সংখ্যাটাকে বাংলায় শব্দে বলবে – "
                "উদাহরণ: 1000 = এক হাজার, 1500 = এক হাজার পাঁচশ, 20000 = বিশ হাজার, 50000 = পঞ্চাশ হাজার, 100000 = এক লক্ষ, 1000000 = দশ লক্ষ। "
                "300 = তিনশ, 500 = পাঁচশ। কখনও থ্রি জিরো জিরো বা ফাইভ জিরো জিরো বলবে না।\n\n"

                "ফ্লো:\n"
                "১. প্রথম ধাপ: অত্যন্ত সম্মানের সাথে ওপেনিং, নিজের পরিচয়, permission নেবে এবং বলবে এক থেকে দুই মিনিটে শুধু মূল সুবিধাগুলো শেয়ার করবে।\n"
                "২. দ্বিতীয় ধাপ: হালকা র‍্যাপোর্ট বানিয়ে স্যারের পেশা, আনুমানিক আয় রেঞ্জ এবং family তে কারা depend করেন তা ধীরে ধীরে জানবে, প্রতিটি প্রশ্নের শেষে 'কি জানতে পারি স্যার' যোগ করবে।\n"
                "৩. তৃতীয় ধাপ: জিজ্ঞেস করবে কোন দিকটি বেশি গুরুত্বপূর্ণ – হঠাৎ hospital বা accident খরচের কভার, নাকি future saving আর বাচ্চাদের পড়াশোনার নিরাপত্তা, নাকি দুটোই।\n"
                "৪. চতুর্থ ধাপ: সংক্ষেপে দুই তিনটি প্রোডাক্ট পজিশন করবে – বছরে কয়েকশ টাকা প্রিমিয়ামের ছোট health/accident কভার, "
                "মাসিক প্রিমিয়ামে দশ থেকে বিশ বছর মেয়াদী life plus saving প্ল্যান যেখানে মেয়াদ শেষে বড় অংকের টাকা plus life cover পাওয়া যায়।\n\n"

                "Objection হ্যান্ডলিং: টাকা কম, office এ medical আছে, বা আগের policy আছে – এগুলো এলে ছোট উদাহরণ দিয়ে বুঝাবে কীভাবে মাসিক আয়ের দুই থেকে চার শতাংশের মতো ছোট অংশ family কে বড় ঝুঁকি থেকে বাঁচাতে পারে, "
                "office medical সাধারণত job এর সাথে শেষ হয়ে যায় কিন্তু personal plan lifetime থাকে।\n\n"

                "Closing: শেষে নরম কিন্তু দৃঢ় টোনে প্রস্তাব দেবে – 'স্যার চাইলে এখনই ছোট দুইটা ধাপ নেয়া যায় – আপনি নাম আর WhatsApp বা email দিলে আমি আপনার জন্য হিসাব করে লিখিত details আর আনুমানিক premium পাঠিয়ে দেই'।\n\n"

                "মনে রাখবে: সব কথায় সর্বোচ্চ sincere থাকবে, কখনও রূঢ় হবে না, আর প্রতিটি সংখ্যাকে সর্বদা বাংলায় শব্দে পড়বে, digit by digit কখনও নয়।\n\n"

                "CRITICAL RESPONSE LENGTH RULE (অত্যন্ত গুরুত্বপূর্ণ): এটি একটি PHONE CALL যেখানে real-time voice conversation হচ্ছে। "
                "প্রতিটি response অবশ্যই অত্যন্ত ছোট রাখতে হবে - maximum ১-২ বাক্য (২০-৪০ শব্দের বেশি নয়)। "
                "কখনও এক turn এ লম্বা ব্যাখ্যা দেবে না। তথ্যগুলো ছোট ছোট conversational chunks এ ভাগ করে দেবে। "
                "উদাহরণ: 'শুভদিন স্যার, আমি [নাম], ঢাকা-ভিত্তিক বীমা কোম্পানির অ্যাডভাইজার...' এভাবে লম্বা করার বদলে, "
                "শুধু বলবে 'ধন্যবাদ স্যার। আমি লাইফ আর হেলথ ইনসুরেন্স নিয়ে কল করেছি। দুই মিনিট সময় দিতে পারবেন?'। "
                "ছোট responses conversation কে natural এবং দ্রুত করে। সবসময় brevity prefer করবে।\n\n"

                "Your responses will be spoken aloud in real-time. ALWAYS keep responses SHORT (max 1-2 sentences). Never explain everything at once. Break information into small chunks."
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
