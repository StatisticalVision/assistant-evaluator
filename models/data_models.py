from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Persona:
    name: str
    traits: List[str]


@dataclass
class Behavior:
    name: str
    characteristics: List[str]


@dataclass
class Grade:
    empathy: float = -1.0
    accuracy: float = -1.0
    # Avg
    response_time: float = -1.0


@dataclass
class ConversationTurn:
    user_message: str
    ai_response: str
    timestamp: float = 0.0


@dataclass
class Conversation:
    id: str
    persona: Persona
    behavior: Behavior
    # The initial question that starts the conversation
    question: str
    # The expected output for the entire conversation (could be a specific piece of information or goal)
    expected_outputs: str
    # A list of conversation turns (user message, ai response pairs)
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    grade: Optional[Grade] = field(default_factory=Grade)

    def add_turn(
        self, user_message: str, ai_response: str, timestamp: float = 0.0
    ) -> None:
        """Add a new turn to the conversation"""
        turn = ConversationTurn(
            user_message=user_message, ai_response=ai_response, timestamp=timestamp
        )
        self.conversation_turns.append(turn)

    @property
    def actual_outputs(self) -> str:
        """Get the entire conversation as a formatted string - for backward compatibility"""
        if not self.conversation_turns:
            return ""

        conversation_text = ""
        for i, turn in enumerate(self.conversation_turns):
            conversation_text += f"User: {turn.user_message}\n"
            conversation_text += f"Assistant: {turn.ai_response}\n\n"

        return conversation_text.strip()

    @property
    def turns_count(self) -> int:
        """Get the number of turns in the conversation"""
        return len(self.conversation_turns)

    @property
    def last_ai_response(self) -> str:
        """Get the last AI response"""
        if not self.conversation_turns:
            return ""
        return self.conversation_turns[-1].ai_response

    @classmethod
    def from_csv_row(cls, row: pd.Series) -> Optional["Conversation"]:
        """Create a Conversation instance from a CSV row"""
        try:
            # Get persona and behavior from manager
            behavior = Behavior(
                name=row["behavior_name"],
                characteristics=row["behavior_characteristics"].split(", "),
            )
            persona = Persona(
                name=row["persona_name"], traits=row["persona_traits"].split(", ")
            )

            if not persona or not behavior:
                logger.error(
                    f"Could not find persona or behavior for row {row['conversation_id']}"
                )
                return None

            # Create grade if metrics exist
            grade = None
            if all(
                metric != -1
                for metric in [
                    row.get("empathy", -1),
                    row.get("accuracy", -1),
                    row.get("response_time", -1),
                ]
            ):
                grade = Grade(
                    empathy=row["empathy"],
                    accuracy=row["accuracy"],
                    response_time=row["response_time"],
                )

            # Create conversation object
            conversation = cls(
                id=str(row["conversation_id"]),
                persona=persona,
                behavior=behavior,
                question=row["question"],
                expected_outputs=row["expected_outputs"],
                conversation_turns=[],  # Initialize with empty turns
                grade=grade,
            )

            # If there's actual_outputs in the CSV, parse it into conversation turns
            if row.get("actual_outputs") and row["actual_outputs"]:
                try:
                    # Try to parse the actual_outputs as a conversation
                    conversation_text = row["actual_outputs"]
                    # Simple parsing logic - can be improved based on your format
                    # This assumes "User:" and "Assistant:" prefixes
                    parts = conversation_text.split("User: ")
                    for part in parts[1:]:  # Skip the first empty part
                        if "Assistant: " in part:
                            user_msg, ai_part = part.split("Assistant: ", 1)
                            ai_msg = ai_part.split("User: ")[0].strip()
                            conversation.add_turn(user_msg.strip(), ai_msg)
                except Exception as e:
                    # If parsing fails, add it as a single turn with the question
                    logger.warning(f"Failed to parse conversation: {e}")
                    conversation.add_turn(row["question"], row["actual_outputs"])

            # Validate the conversation
            if conversation.validate():
                return conversation
            else:
                logger.warning(
                    f"Validation failed for conversation {row['conversation_id']}"
                )
                return None

        except KeyError as e:
            logger.error(f"Missing required field in row {row['conversation_id']}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"Error creating conversation from row {row['conversation_id']}: {e}"
            )
            return None

    @classmethod
    def load_from_csv(cls, csv_path: Path) -> List["Conversation"]:
        """Load all conversations from a CSV file"""
        try:
            # Read CSV
            df = pd.read_csv(csv_path, na_filter=False)
            logger.info(f"Loading {len(df)} conversations from {csv_path}")

            # Convert rows to conversations
            conversations = []
            for _, row in df.iterrows():
                conversation = cls.from_csv_row(row)
                if conversation:
                    conversations.append(conversation)

            logger.info(f"Successfully loaded {len(conversations)} valid conversations")
            return conversations

        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {csv_path}")
            return []
        except pd.errors.ParserError:
            logger.error(f"Error parsing CSV file: {csv_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
            return []

    def validate(self):
        # TODO with pydantic this goes inside dataclass
        # filter out short questions
        if len(self.question) < 10:
            return False
        # only use actual questions
        if "?" not in self.question:
            return False
        return True


@dataclass
class KnowledgeBase:
    fname: Path
    list_faqs: List[Dict[str, str]]
    faqs: Dict[str, str]

    def __init__(self, fname: Path):
        self.fname = fname
        self.load_knowledge(self.fname)

    # TODO Would we want to add any other info?
    def load_knowledge(self, kb_file: Path):
        # Currently only using the FAQ section of a knowledge base as the entire knowledge base and unifies it into a single dictionary
        try:
            with open(kb_file) as f:
                data = json.load(f)
                self.list_faqs = data["faqs"]
                self.faqs = self.load_faqs(self.list_faqs)
            logger.info(f"Successfully loaded knowledge base from {kb_file}")
        except FileNotFoundError:
            logger.error(f"Knowledge base file not found: {kb_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in knowledge base file: {kb_file}")
            raise

    def load_faqs(self, faqs: List[Dict[str, str]]):
        unified_faq = {}
        for faq in faqs:
            unified_faq.update(faq)
        return unified_faq
