from pathlib import Path
from typing import List, Optional
import json
import logging
from models.data_models import Persona, Behavior

logger = logging.getLogger(__name__)


class PersonaBehaviorManager:
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.personas: List[Persona] = []
        self.behaviors: List[Behavior] = []
        self._load_config()

    def _load_config(self) -> None:
        """Load both personas and behaviors from the config file"""
        try:
            with open(self.config_file) as f:
                data = json.load(f)

            # Load personas
            self.personas = [
                Persona(
                    name=p["name"],
                    traits=p["traits"],
                )
                for p in data.get("personas", [])
            ]
            logger.info(f"Loaded {len(self.personas)} personas")

            # Load behaviors
            self.behaviors = [
                Behavior(
                    name=b["name"],
                    characteristics=b["characteristics"],
                )
                for b in data.get("behaviors", [])
            ]
            logger.info(f"Loaded {len(self.behaviors)} behaviors")

        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file: {self.config_file}")
            raise
        except KeyError as e:
            logger.error(f"Missing required field in config: {str(e)}")
            raise

    def get_persona(self, name: str) -> Optional[Persona]:
        """Get a specific persona by name"""
        for persona in self.personas:
            if persona.name.lower() == name.lower():
                return persona
        return None

    def get_behavior(self, name: str) -> Optional[Behavior]:
        """Get a specific behavior by name"""
        for behavior in self.behaviors:
            if behavior.name.lower() == name.lower():
                return behavior
        return None

    def list_personas(self) -> List[str]:
        """Get list of all persona names"""
        return [persona.name for persona in self.personas]

    def list_behaviors(self) -> List[str]:
        """Get list of all behavior names"""
        return [behavior.name for behavior in self.behaviors]

    def validate_config(self) -> bool:
        """Validate that the config has all required fields"""
        if not self.personas:
            logger.error("No personas found in config")
            return False

        if not self.behaviors:
            logger.error("No behaviors found in config")
            return False

        # Validate persona fields
        for persona in self.personas:
            if not all(
                [
                    persona.name,
                    persona.traits,
                ]
            ):
                logger.error(f"Missing required fields in persona: {persona.name}")
                return False

        # Validate behavior fields
        for behavior in self.behaviors:
            if not all([behavior.name, behavior.characteristics]):
                logger.error(f"Missing required fields in behavior: {behavior.name}")
                return False

        return True


# Example usage
if __name__ == "__main__":
    manager = PersonaBehaviorManager(Path("data/personas.json"))

    # List all personas and behaviors
    logger.info("Personas:", manager.list_personas())
    logger.info("Behaviors:", manager.list_behaviors())

    # Get specific persona and behavior
    busy_prof = manager.get_persona("Busy Professional")
    frustrated = manager.get_behavior("frustrated")

    if busy_prof and frustrated:
        logger.info(f"\nPersona '{busy_prof.name}' with behavior '{frustrated.name}':")
        logger.info(f"Behavior characteristics: {frustrated.characteristics}")
