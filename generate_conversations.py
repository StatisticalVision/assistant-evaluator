from pathlib import Path
from core.conversation_generator import ConversationGenerator
from core.persona_manager import PersonaBehaviorManager
from dotenv import load_dotenv
import os
import json
import argparse
import logging
from utils.logging_config import setup_logging
from models.data_models import KnowledgeBase


setup_logging()

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Example FAQ data


# Initialize Argument Parser, Allowing for Custom KB File Path
# parser = argparse.ArgumentParser(description="AI conversation generator")
# parser.add_argument(
#     "-k",
#     "--knowledgebase",
#     type=str,
#     help="Path to the knowledge base JSON file (default: kb/sendero.json)",
# )
# parser.add_argument(
#     "-p",
#     "--personas",
#     type=str,
#     help="Path to the CSV file containing AI responses (default: data/ai_responses.csv) columns expected are ['Customer_Query', 'AI_Response']",
# )
# args = parser.parse_args()
# kb_file = args.knowledgebase  # Use provided KB file or fallback to default
# personas_file = args.personas

kb_file = "data/kb.json"
personas_file = "data/behaviorPersona.json"
knowledge_base = KnowledgeBase(kb_file)

# Initialize generator
generator = ConversationGenerator(api_key=OPENAI_API_KEY)
manager = PersonaBehaviorManager(config_file=Path(personas_file))

# List all personas and behaviors
logger.info("Personas:", manager.list_personas())
logger.info("Behaviors:", manager.list_behaviors())

# Generate questions
generator.generate_questions(
    knowledge_base=knowledge_base,
    personas=manager.personas,
    behaviors=manager.behaviors,
    questions_per_faq=3,
    out_of_scope_questions_per_persona=2,
    output_file=Path("generated_conversations2.csv"),
)
