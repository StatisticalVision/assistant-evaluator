from typing import List, Dict, Any
import logging
import csv
from pathlib import Path
import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

from models.data_models import Behavior, Persona, KnowledgeBase, Conversation, Grade

logger = logging.getLogger(__name__)

CANNOT_HELP_ANSWER = "I'm sorry a human operator will need to help you with this"
FILE_OUTPUT_NAME = "generate_conversations.csv"


class ConversationGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.llm = OpenAI(model=model, api_key=api_key)
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
            return questions[:expected_count]
        return questions

    def generate_questions(
        self,
        knowledge_base: KnowledgeBase,
        personas: List[Persona],
        behaviors: List[Behavior],
        questions_per_faq: int,
        out_of_scope_questions_per_persona: int,
        output_file: Path,
    ) -> None:
        """Generate questions based on FAQs and personas"""

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

        # Generate FAQ-based questions
        for persona in personas:
            for behavior in behaviors:
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
                        response = self.llm.complete(system_prompt + "\n\n" + prompt)
                        variations = self._get_valid_questions(
                            response.text, questions_per_faq
                        )

                        for variation in variations:
                            idx += 1
                            # TODO if you use pydantic we can do validation here
                            conversation = Conversation(
                                id=idx,
                                persona=persona,
                                behavior=behavior,
                                question=variation,
                                actual_outputs="",
                                expected_outputs=faq_answer,
                                grade=Grade(),
                            )

                            if not conversation.validate():
                                continue
                            conversations.append(conversation)
                    except Exception as e:
                        logger.error(f"Error generating FAQ questions: {str(e)}")
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
                    response = self.llm.complete(system_prompt + "\n\n" + prompt)
                    out_of_scope = self._get_valid_questions(
                        response.text, out_of_scope_questions_per_persona
                    )

                    for question in out_of_scope:
                        conversation = Conversation(
                            id=idx,
                            persona=persona,
                            behavior=behavior,
                            question=variation,
                            actual_outputs="",
                            expected_outputs=CANNOT_HELP_ANSWER,
                            grade=Grade(),
                        )
                        if not conversation.validate():
                            continue

                        conversations.append(conversation)
                except Exception as e:
                    logger.error(
                        f"Error generating out-of-scope questions: {str(e)}",
                        exc_info=True,
                    )
                    continue

        export_conversations_to_csv(conversations, "generate_conversations.csv")

        logger.info(
            f"Generated {len(conversations)} valid questions and saved to {output_file}"
        )


# TODO this can be done implicitly in pydantic if we switch from dataclasses
def export_conversations_to_csv(conversations: List[Conversation], filename: str):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Writing the header
        writer.writerow(
            [
                "conversation_id",
                "persona_name",
                "persona_traits",
                "behavior_name",
                "behavior_characteristics",
                "question",
                "expected_outputs",
                "actual_outputs",
                "empathy",
                "accuracy",
                "response_time",
            ]
        )

        # Writing the rows
        for convo in conversations:
            writer.writerow(
                [
                    convo.id,
                    convo.persona.name,
                    ", ".join(convo.persona.traits),
                    convo.behavior.name,
                    ", ".join(convo.behavior.characteristics),
                    convo.question,
                    convo.expected_outputs,
                    convo.actual_outputs,
                    -1,
                    -1,
                    -1,
                ]
            )
