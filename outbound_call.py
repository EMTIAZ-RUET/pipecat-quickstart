"""
Pipecat Twilio Outbound Call Example

This script makes an outbound phone call using Twilio and connects it to your Bangla voice bot.
"""

import os
from dotenv import load_dotenv
from twilio.rest import Client
from loguru import logger

load_dotenv(override=True)


def make_outbound_call(to_number: str, server_url: str):
    """
    Make an outbound call using Twilio.

    Args:
        to_number: The phone number to call (E.164 format, e.g., +8801765786548)
        server_url: Your server's WebSocket URL (e.g., wss://your-server.com/ws or https://your-ngrok-url.ngrok.io)
    """
    # Get Twilio credentials from environment
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not all([account_sid, auth_token, from_number]):
        logger.error("Missing Twilio credentials. Please check your .env file.")
        return None

    # Initialize Twilio client
    client = Client(account_sid, auth_token)

    # Create TwiML for WebSocket connection
    # Note: Replace 'ws' with 'wss' and update the path based on your server setup
    websocket_url = f"{server_url}/ws" if not server_url.endswith('/ws') else server_url

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{websocket_url}">
            <Parameter name="call_type" value="outbound" />
            <Parameter name="language" value="bn-BD" />
        </Stream>
    </Connect>
</Response>"""

    try:
        logger.info(f"Initiating outbound call to {to_number} from {from_number}")
        logger.info(f"WebSocket URL: {websocket_url}")

        # Create the call
        call = client.calls.create(
            to=to_number,
            from_=from_number,
            twiml=twiml
        )

        logger.success(f"Call initiated successfully!")
        logger.info(f"Call SID: {call.sid}")
        logger.info(f"Status: {call.status}")

        return call

    except Exception as e:
        logger.error(f"Failed to create call: {e}")
        return None


if __name__ == "__main__":
    import sys

    # Get target phone number from environment or use default
    target_number = os.getenv("TARGET_PHONE_NUMBER", "+8801765786548")

    # Check if server URL is provided as command line argument
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        # Auto-detect ngrok URL if available
        try:
            import requests
            response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
            tunnels = response.json()
            if tunnels.get("tunnels"):
                server_url = tunnels["tunnels"][0]["public_url"]
                logger.info(f"Auto-detected ngrok URL: {server_url}")
            else:
                server_url = input("Enter your server WebSocket URL (e.g., https://your-server.ngrok.io): ").strip()
        except Exception:
            server_url = input("Enter your server WebSocket URL (e.g., https://your-server.ngrok.io): ").strip()

    if not server_url:
        logger.error("Server URL is required!")
        exit(1)

    # Make the call
    call = make_outbound_call(target_number, server_url)

    if call:
        logger.info("Call in progress. Check your phone!")
        logger.info(f"Target number: {target_number}")
        logger.info("The call should ring shortly. If answered, you'll be connected to the Bangla voice bot.")
    else:
        logger.error("Failed to initiate call.")
