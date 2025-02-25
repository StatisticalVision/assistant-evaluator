from typing import List, Dict, Optional, Tuple
import csv
from pathlib import Path
import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

from models.data_models import (
    Behavior,
    Persona,
    KnowledgeBase,
    Conversation,
    Grade,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

CANNOT_HELP_ANSWER = "I'm sorry a human operator will need to help you with this"
FILE_OUTPUT_NAME = "output/generate_conversations.csv"


class ConversationGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.llm = OpenAI(model=model, api_key=api_key)
        logger.info(f"Initialized ConversationGenerator with model {model}")
        # self.llm = Ollama(model="mistral", request_timeout=30.0)

    def _get_valid_questions(
        self, response_text: str, expected_count: int
    ) -> List[str]:
        """Parse and validate questions from LLM response"""
        # Split response into lines and clean
        questions = [
            line.strip()
            for line in response_text.split("\n")
            if line.strip() and "?" in line  # Ensure it's a question
        ]

        # Remove any lines that look like instructions or prompts
        questions = [
            question
            for question in questions
            if not any(
                keyword in question.lower()
                for keyword in ["generate", "variation", "faq", "question:", "example"]
            )
        ]

        # Ensure we have the expected number of questions
        if len(questions) < expected_count:
            logger.warning(
                f"Got fewer questions than expected: {len(questions)} vs {expected_count}"
            )
        if len(questions) > expected_count:
            logger.debug(
                f"Limiting questions from {len(questions)} to {expected_count}"
            )
            return questions[:expected_count]
        return questions

    def _generate_followup_turns(
        self,
        initial_question: str,
        expected_answer: str,
        persona: Persona,
        behavior: Behavior,
        num_turns: int = 3,
    ) -> List[Tuple[str, str]]:
        """Generate a multi-turn conversation between a user and assistant"""

        system_prompt = f"""
        You are simulating a conversation between a customer and a customer service assistant.
        The customer has the following persona: {persona.name}
        Traits: {', '.join(persona.traits)}
        
        The customer is currently exhibiting this behavior: {behavior.name}
        Behavior Characteristics: {', '.join(behavior.characteristics)}
        
        The initial question from the customer is: "{initial_question}"
        
        The customer service assistant is expected to eventually provide this information: "{expected_answer}"
        
        Generate a natural {num_turns}-turn conversation between the customer and assistant.
        For each turn, provide both what the customer says and how the assistant responds.
        
        Format your response strictly as:
        
        USER: [customer's message]
        ASSISTANT: [assistant's response]
        
        USER: [customer's follow-up]
        ASSISTANT: [assistant's response]
        
        And so on for {num_turns} turns.
        """

        try:
            logger.debug(
                f"Generating conversation for question: {initial_question[:50]}..."
            )
            response = self.llm.complete(system_prompt)
            conversation_text = response.text

            # Parse the conversation
            turns = []
            parts = conversation_text.split("USER: ")

            for part in parts[1:]:  # Skip the first empty part
                if "ASSISTANT: " in part:
                    user_msg, ai_part = part.split("ASSISTANT: ", 1)
                    if "USER: " in ai_part:
                        ai_msg = ai_part.split("USER: ")[0].strip()
                    else:
                        ai_msg = ai_part.strip()

                    turns.append((user_msg.strip(), ai_msg))

            if len(turns) < num_turns:
                logger.warning(
                    f"Generated only {len(turns)} turns instead of requested {num_turns}"
                )

            return turns[:num_turns]  # Limit to the requested number of turns

        except Exception as e:
            logger.error(f"Error generating conversation: {str(e)}", exc_info=True)
            # Return a simple one-turn conversation as fallback
            logger.warning(
                f"Using fallback conversation for initial question: {initial_question[:50]}..."
            )
            return [(initial_question, expected_answer)]

    def generate_questions(
        self,
        knowledge_base: KnowledgeBase,
        personas: List[Persona],
        behaviors: List[Behavior],
        questions_per_faq: int,
        out_of_scope_questions_per_persona: int,
        output_file: Path,
        turns_per_conversation: int = 3,
    ) -> None:
        """Generate multi-turn conversations based on FAQs and personas"""

        system_prompt = """
        You are a question generator creating customer service inquiries.
        Generate questions that match the persona's style and current behavior.
        
        IMPORTANT:
        - Return ONLY the questions, one per line
        - Each line must be a complete question ending with a question mark
        - Do not include any other text, numbering, or explanations
        - Match the persona's communication style
        - Incorporate the specified behavior naturally
        """

        conversations = []
        faqs = knowledge_base.faqs
        idx = 0

        logger.info(
            f"Starting conversation generation with {len(personas)} personas, {len(behaviors)} behaviors"
        )
        logger.info(
            f"Generating {questions_per_faq} questions per FAQ and {out_of_scope_questions_per_persona} out-of-scope questions per persona"
        )

        # Generate FAQ-based questions
        for persona in personas:
            logger.info(f"Processing persona: {persona.name}")
            for behavior in behaviors:
                logger.info(
                    f"Processing behavior: {behavior.name} for persona {persona.name}"
                )
                # Generate variations of FAQ questions
                for faq_question, faq_answer in faqs.items():
                    prompt = f"""
                    Generate {questions_per_faq} different ways to ask this question:
                    "{faq_question}"
                    
                    Persona: {persona.name}
                    Traits: {', '.join(persona.traits)}
                    Current Behavior: {behavior.name}
                    Behavior Characteristics: {', '.join(behavior.characteristics)}
                    
                    Remember: Return only the questions, one per line.
                    """

                    try:
                        logger.debug(
                            f"Generating variations for FAQ: {faq_question[:50]}..."
                        )
                        response = self.llm.complete(system_prompt + "\n\n" + prompt)
                        variations = self._get_valid_questions(
                            response.text, questions_per_faq
                        )

                        for variation in variations:
                            idx += 1

                            # Create a new conversation with the initial question
                            conversation = Conversation(
                                id=str(idx),
                                persona=persona,
                                behavior=behavior,
                                question=variation,
                                expected_outputs=faq_answer,
                                conversation_turns=[],
                                grade=Grade(),
                            )

                            # Generate follow-up turns
                            turns = self._generate_followup_turns(
                                initial_question=variation,
                                expected_answer=faq_answer,
                                persona=persona,
                                behavior=behavior,
                                num_turns=turns_per_conversation,
                            )

                            # Add turns to the conversation
                            for user_msg, ai_response in turns:
                                conversation.add_turn(user_msg, ai_response)

                            if not conversation.validate():
                                logger.debug(f"Conversation {idx} failed validation")
                                continue

                            logger.debug(f"Added valid conversation {idx}")
                            conversations.append(conversation)
                    except Exception as e:
                        logger.error(
                            f"Error generating FAQ questions: {str(e)}", exc_info=True
                        )
                        continue

                # Generate out-of-scope questions
                prompt = f"""
                Generate {out_of_scope_questions_per_persona} questions that are NOT covered by these FAQs:
                {list(faqs.keys())}
                
                Persona: {persona.name}
                Traits: {', '.join(persona.traits)}
                Current Behavior: {behavior.name}
                Behavior Characteristics: {', '.join(behavior.characteristics)}
                
                Questions should be clearly outside the scope of the FAQs.
                Return only the questions, one per line.
                """

                try:
                    logger.info(
                        f"Generating out-of-scope questions for persona {persona.name} with behavior {behavior.name}"
                    )
                    response = self.llm.complete(system_prompt + "\n\n" + prompt)
                    out_of_scope = self._get_valid_questions(
                        response.text, out_of_scope_questions_per_persona
                    )

                    for question in out_of_scope:
                        idx += 1

                        # Create conversation with out-of-scope question
                        conversation = Conversation(
                            id=str(idx),
                            persona=persona,
                            behavior=behavior,
                            question=question,
                            expected_outputs=CANNOT_HELP_ANSWER,
                            conversation_turns=[],
                            grade=Grade(),
                        )

                        # Generate follow-up turns for out-of-scope questions
                        turns = self._generate_followup_turns(
                            initial_question=question,
                            expected_answer=CANNOT_HELP_ANSWER,
                            persona=persona,
                            behavior=behavior,
                            num_turns=turns_per_conversation,
                        )

                        # Add turns to the conversation
                        for user_msg, ai_response in turns:
                            conversation.add_turn(user_msg, ai_response)

                        if not conversation.validate():
                            logger.debug(
                                f"Out-of-scope conversation {idx} failed validation"
                            )
                            continue

                        logger.debug(f"Added valid out-of-scope conversation {idx}")
                        conversations.append(conversation)
                except Exception as e:
                    logger.error(
                        f"Error generating out-of-scope questions for persona {persona.name}: {str(e)}",
                        exc_info=True,
                    )
                    continue

        logger.info(
            f"Generation complete. Total valid conversations: {len(conversations)}"
        )
        export_conversations_to_csv(conversations, output_file)

        logger.info(
            f"Generated {len(conversations)} valid conversations and saved to {output_file}"
        )


# CSV export function that works with the new multi-turn conversation format
def export_conversations_to_csv(conversations: List[Conversation], filename: str):
    """
    Export conversations to a CSV file with proper error handling and logging

    Args:
        conversations: List of Conversation objects to export
        filename: Path to the output CSV file
    """
    logger.info(f"Exporting {len(conversations)} conversations to {filename}")

    # Create directories if they don't exist
    output_dir = Path(filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Writing the header
            header = [
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
            writer.writerow(header)
            logger.debug(f"Wrote CSV header with {len(header)} columns")

            # Writing the rows with progress logging
            success_count = 0
            for i, convo in enumerate(conversations):
                try:
                    writer.writerow(
                        [
                            convo.id,
                            convo.persona.name,
                            ", ".join(convo.persona.traits),
                            convo.behavior.name,
                            ", ".join(convo.behavior.characteristics),
                            convo.question,
                            convo.expected_outputs,
                            convo.actual_outputs,  # This calls the property that formats all turns
                            convo.turns_count,
                            -1,  # empathy placeholder
                            -1,  # accuracy placeholder
                            -1,  # response_time placeholder
                        ]
                    )
                    success_count += 1

                    # Log progress periodically
                    if (i + 1) % 50 == 0 or i == len(conversations) - 1:
                        logger.debug(
                            f"Exported {i + 1}/{len(conversations)} conversations"
                        )

                except Exception as e:
                    logger.error(f"Error exporting conversation {convo.id}: {str(e)}")

            logger.info(
                f"Successfully exported {success_count}/{len(conversations)} conversations to {filename}"
            )

    except Exception as e:
        logger.error(
            f"Error creating or writing to CSV file {filename}: {str(e)}", exc_info=True
        )
        raise
