import json
import asyncio
import websockets
import boto3
import os
import time
import csv
import argparse
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from models.data_models import (
    Behavior,
    Persona,
    KnowledgeBase,
    Conversation,
    Grade,
    ConversationTurn,
)
import logging

logger = logging.getLogger(__name__)

load_dotenv()


# Function to get parameters from AWS SSM
def get_parameter(name, with_decryption=True):
    ssm = boto3.client("ssm", region_name="us-east-2")
    try:
        response = ssm.get_parameter(Name=name, WithDecryption=with_decryption)
        return response["Parameter"]["Value"]
    except Exception as e:
        logger.info(f"Could not get parameter {name} from SSM: {e}")
        return None


# Get OpenAI API key (use environment variable or SSM)
OPENAI_API_KEY = get_parameter("/sendero_ai/openai_key")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing the OpenAI API key. Set OPENAI_API_KEY environment variable."
        )

# Load system instructions
try:
    with open("data/instructions.txt", "r") as file:
        SYSTEM_MESSAGE = file.read()
except FileNotFoundError:
    logger.warning("Warning: instructions.txt not found. Using default instructions.")
    SYSTEM_MESSAGE = """You are a helpful virtual assistant for Sendero Health Plans."""

# Configuration
MODEL = "gpt-4o-realtime-preview-2024-12-17"  # OpenAI model to use

# CSV field names
FIELDNAMES = [
    "conversation_id",
    "persona_name",
    "persona_traits",
    "behavior_name",
    "behavior_characteristics",
    "question",
    "expected_outputs",
    "actual_outputs",
    "turns_count",
    "empathy",
    "accuracy",
    "response_time",
]


def export_conversations_to_csv(conversations, output_file):
    """Export conversations to CSV with multi-turn data"""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for conv in conversations:
            writer.writerow(
                {
                    "conversation_id": conv.id,
                    "persona_name": conv.persona.name,
                    "persona_traits": ", ".join(conv.persona.traits),
                    "behavior_name": conv.behavior.name,
                    "behavior_characteristics": ", ".join(
                        conv.behavior.characteristics
                    ),
                    "question": conv.question,
                    "expected_outputs": conv.expected_outputs,
                    "actual_outputs": conv.actual_outputs,
                    "turns_count": len(conv.conversation_turns),
                    "empathy": conv.grade.empathy if conv.grade else -1,
                    "accuracy": conv.grade.accuracy if conv.grade else -1,
                    "response_time": conv.grade.response_time if conv.grade else -1,
                }
            )


async def process_test_file(input_file, output_file, max_turns=3):
    """Process the test file and generate multi-turn conversation outputs"""
    """Process the test file and generate the outputs for multi-turn conversations"""
    # Read test cases
    test_cases = Conversation.load_from_csv(input_file)

    # Process each test case
    results = []
    for test_case in tqdm(test_cases, desc="Processing test cases"):
        # Create persona context from the test case
        persona_context = f"Persona: {test_case.persona.name}\nTraits: {', '.join(test_case.persona.traits)}\n"
        persona_context += (
            f"Behavior: {test_case.behavior.name}\n"
            f"Characteristics: {', '.join(test_case.behavior.characteristics)}"
        )

        # Run the conversation
        start_time = time.time()

        # Initialize with the first question
        initial_question = test_case.question

        # Create a copy of the test case to store results
        result = Conversation(
            id=test_case.id,
            persona=test_case.persona,
            behavior=test_case.behavior,
            question=initial_question,
            expected_outputs=test_case.expected_outputs,
            conversation_turns=[],
            grade=Grade(),
        )

        # Connect to the AI service
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={MODEL}",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:
            # Initialize session
            await initialize_session(openai_ws, persona_context)

            # Start with the initial question
            current_question = initial_question

            # Conduct the conversation for multiple turns
            for turn_index in range(max_turns):
                # Send the current question
                await send_user_message(openai_ws, current_question)

                # Get AI response
                ai_response = await collect_assistant_response(openai_ws)

                # Add the turn to our conversation result
                result.add_turn(current_question, ai_response, time.time())

                # If we've reached the last turn, break
                if turn_index == max_turns - 1:
                    break

                # Otherwise, generate a follow-up question based on the AI's response
                follow_up_question = await generate_follow_up_question(
                    openai_ws,
                    persona_context,
                    initial_question,
                    current_question,
                    ai_response,
                    turn_index,
                )

                # Update the current question for the next turn
                current_question = follow_up_question

        # Calculate total response time
        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        # Update the grade
        result.grade.response_time = response_time

        # Add result to results list
        results.append(result)

    # Write results to output file
    export_conversations_to_csv(results, output_file)

    logger.info(f"Results written to {output_file}")
    return results


async def run_single_test(persona_context, question):
    """Run a single test conversation with the AI"""
    try:
        # Connect to OpenAI's realtime API
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={MODEL}",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:
            # Initialize session
            await initialize_session(openai_ws, persona_context)

            # Send the test question
            await send_user_message(openai_ws, question)

            # Collect the response
            assistant_response = await collect_assistant_response(openai_ws)

            return assistant_response

    except Exception as e:
        logger.error(f"Error in test run: {e}")
        return f"ERROR: {str(e)}"


async def initialize_session(openai_ws, persona_context):
    """Initialize the session with the OpenAI API"""
    # Combine the standard system message with the persona context
    combined_instructions = f"{SYSTEM_MESSAGE}\n\n{persona_context}"

    session_update = {
        "type": "session.update",
        "session": {
            "tools": [
                {
                    "type": "function",
                    "name": "schedule_callback",
                    "description": "Call this function when a user confirms that they want to schedule a call-back.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "language": {
                                "type": "string",
                                "description": "The language the user is speaking in (in lowercase).",
                            }
                        },
                        "required": ["language"],
                    },
                },
                {
                    "type": "function",
                    "name": "pay_by_phone",
                    "description": "Call this function when a user indicates that they want to make a payment over the phone.",
                },
                {
                    "type": "function",
                    "name": "provider",
                    "description": "Call this function when a user indicates that they are a provider.",
                },
                {
                    "type": "function",
                    "name": "member",
                    "description": "Call this function when a user indicates that they are a MEMBER.",
                },
            ],
            "tool_choice": "auto",
            "input_audio_format": None,  # Text-only mode
            "output_audio_format": None,  # Text-only mode
            "instructions": combined_instructions,
            "modalities": ["text"],  # Text-only modality
            "temperature": 0.8,
        },
    }

    await openai_ws.send(json.dumps(session_update))


async def send_user_message(openai_ws, text):
    """Send a user message to the AI"""
    message = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }
    await openai_ws.send(json.dumps(message))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def collect_assistant_response(openai_ws):
    """Collect the assistant's response"""
    full_response = ""
    try:
        async for message in openai_ws:
            data = json.loads(message)

            # Handle text responses
            if data.get("type") == "response.text.delta":
                if "delta" in data:
                    full_response += data["delta"]

            # Handle function calls
            if data.get("type") == "response.done" and any(
                item.get("type") == "function_call"
                for item in data.get("response", {}).get("output", [])
            ):

                output = data["response"]["output"]
                function_item = next(
                    (item for item in output if item.get("type") == "function_call"),
                    None,
                )

                if function_item:
                    function_name = function_item.get("name")

                    # Handle different functions
                    if function_name == "schedule_callback":
                        func_args = json.loads(function_item.get("arguments", "{}"))
                        language = func_args.get("language", "english")
                        if language.lower() == "spanish":
                            full_response += (
                                "\n[FUNCTION CALLED: schedule_callback - SPANISH]"
                            )
                            full_response += "\nMuy bien, he registrado una solicitud de devolución de llamada."
                        else:
                            full_response += (
                                "\n[FUNCTION CALLED: schedule_callback - ENGLISH]"
                            )
                            full_response += (
                                "\nAlright, I have created a call-back request."
                            )

                    elif function_name == "pay_by_phone":
                        full_response += "\n[FUNCTION CALLED: pay_by_phone]"
                        full_response += (
                            "\nPlease hold while I connect you to our payment system."
                        )

                    elif function_name == "provider":
                        full_response += "\n[FUNCTION CALLED: provider]"
                        full_response += "\nPlease hold while I connect you to our provider services."

                    elif function_name == "member":
                        full_response += "\n[FUNCTION CALLED: member]"
                        full_response += "\nThank you. I can help answer most questions about your Id, Member Portal and more."

                    break  # Exit after handling the function

            # Check if response is done
            if data.get("type") == "response.done":
                break

    except Exception as e:
        full_response += f"\n[ERROR: {str(e)}]"

    return full_response


async def main():
    parser = argparse.ArgumentParser(
        description="Test Sendero AI with conversation test cases"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input CSV file with test cases"
    )
    parser.add_argument("--output", type=str, help="Output CSV file for results")

    args = parser.parse_args()
    input_file = args.input

    # Generate default output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{timestamp}.csv"
    else:
        output_file = args.output

    await process_test_file(input_file, output_file)


async def generate_follow_up_question(
    openai_ws,
    persona_context,
    initial_question,
    current_question,
    ai_response,
    turn_index,
):
    """Generate a follow-up question based on the persona and the conversation so far"""

    # Create a prompt for generating a follow-up question
    follow_up_prompt = f"""
    Based on the persona information provided and the conversation so far, generate a natural follow-up question that the user would ask.
    
    {persona_context}
    
    Initial question: {initial_question}
    
    Current conversation:
    User: {current_question}
    Assistant: {ai_response}
    
    Turn number: {turn_index + 1}
    
    Generate a single follow-up question that:
    1. Feels natural based on the persona
    2. Follows from the assistant's response
    3. Could seek clarification, additional information, or express concerns based on the persona's behavior
    
    Return ONLY the follow-up question, with no additional text or explanation.
    """

    # Send the prompt to get a follow-up question
    await send_user_message(openai_ws, follow_up_prompt)

    # Collect the response with the follow-up question
    response = await collect_assistant_response(openai_ws)

    # Clean up the response to get just the question
    # Remove any quotation marks, prefixes like "Follow-up:", etc.
    follow_up = response.strip('" \n')

    # Remove common prefixes if present
    for prefix in ["Follow-up:", "User:", "Follow-up question:", "Question:"]:
        if follow_up.startswith(prefix):
            follow_up = follow_up[len(prefix) :].strip()

    return follow_up
