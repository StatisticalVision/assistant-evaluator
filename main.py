from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import yaml
from pathlib import Path
import logging
import os
from datetime import datetime
from models.data_models import Conversation, Persona, KnowledgeBase
from core.persona_manager import PersonaManager
from core.assistant_interface import AssistantInterface
from core.conversation_generator import ConversationGenerator
from core.report_generator import ReportGenerator
from core.conversation_evaluator import ConversationEvaluator
from core.empathy_evaluator import EmpathyEvaluator
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
# TODO read this from external file or config

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# In main.py
empathy_evaluator = EmpathyEvaluator(OPENAI_API_KEY)


def main():
    # Initialize logging

    # Load configuration
    config = yaml.safe_load(open("config.yaml"))

    # Initialize components
    persona_mgr = PersonaManager(Path(config["persona_file"]))
    kb = KnowledgeBase(Path(config["knowledge_base_file"]))
    assistant = AssistantInterface(config["api_key"])
    evaluator = ConversationEvaluator(config["grading_criteria"])

    # Load data
    personas = persona_mgr.load_personas()
    knowledge_base = kb.load_knowledge()

    # Generate conversations
    generator = ConversationGenerator(knowledge_base, personas)
    conversations = generator.generate_conversations(config["num_conversations"])

    # Record responses
    for conv in conversations:
        for i, input_text in enumerate(conv.inputs):
            response = assistant.record_response(input_text)
            conv.actual_outputs[i] = response

    # Grade conversations
    for conv in conversations:
        conv.grades = evaluator.grade_conversation(conv)

    # Generate report
    report_gen = ReportGenerator()
    final_report = report_gen.generate_report(conversations)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"evaluation_report_{timestamp}.txt", "w") as f:
        f.write(final_report)


if __name__ == "__main__":
    main()

# # Example usage
# if __name__ == "__main__":
#     personas_and_behaviors = read_in_personas_and_behaviors()
#     knowledge_base = read_in_knowledge_base()
#     input_conversations = generate_conversations(knowledge_base, personas_and_behaviors)
#     conversations = record_outputs(input_conversations)
#     graded_output = grade_conversations(conversations)
#     generate_report(graded_output)
