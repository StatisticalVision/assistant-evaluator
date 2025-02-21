from dataclasses import dataclass
from typing import List, Dict, Optional
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
    response_time: float = -1.0


@dataclass
class Conversation:
    id: str
    persona: Persona
    behavior: Behavior
    question: str
    expected_outputs: str
    actual_outputs: str
    grade: Optional[Grade] = None

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
                actual_outputs=row["actual_outputs"],
                grade=grade,
            )

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
