"""
Make Outbound Call Script

Simple script to initiate an outbound call via the /dialout endpoint.
"""

import os
import sys

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


def make_outbound_call(to_number: str, from_number: str, server_url: str = "http://localhost:8765"):
    """
    Make an outbound call by calling the /dialout endpoint.

    Args:
        to_number: The phone number to call (E.164 format)
        from_number: The Twilio phone number to call from
        server_url: The server URL (default: http://localhost:8765)
    """
    dialout_url = f"{server_url}/dialout"

    payload = {"to_number": to_number, "from_number": from_number}

    try:
        logger.info(f"Initiating outbound call to {to_number} from {from_number}")
        logger.info(f"Dialout endpoint: {dialout_url}")

        response = requests.post(dialout_url, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            logger.success("Call initiated successfully!")
            logger.info(f"Call SID: {result['call_sid']}")
            logger.info(f"Status: {result['status']}")
            logger.info(f"To Number: {result['to_number']}")
            return result
        else:
            logger.error(f"Failed to initiate call: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error making call: {e}")
        return None


if __name__ == "__main__":
    # Get target phone number from environment or command line
    target_number = os.getenv("TARGET_PHONE_NUMBER", "+8801765786548")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not from_number:
        logger.error("TWILIO_PHONE_NUMBER not set in .env file")
        sys.exit(1)

    # Allow server URL override
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"

    logger.info(f"Making outbound call to {target_number}")
    logger.info(f"Using Twilio number: {from_number}")
    logger.info(f"Server URL: {server_url}")

    result = make_outbound_call(target_number, from_number, server_url)

    if result:
        logger.info("\\nCall is being initiated!")
        logger.info("The phone should ring shortly. Answer to talk to the Bangla voice bot.")
    else:
        logger.error("Failed to initiate call")
        sys.exit(1)
