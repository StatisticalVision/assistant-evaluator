from pathlib import Path
from typing import List, Optional, Dict
import json
from models.data_models import Persona, Behavior
from utils.logging_config import get_logger

logger = get_logger(__name__)


class PersonaBehaviorManager:
    def __init__(self, config_file: Path):
        """
        Initialize the persona and behavior manager with a configuration file

        Args:
            config_file: Path to the configuration file containing personas and behaviors
        """
        self.config_file = config_file
        self.personas: List[Persona] = []
        self.behaviors: List[Behavior] = []
        logger.info(
            f"Initializing PersonaBehaviorManager with config file: {config_file}"
        )
        self._load_config()

    def _load_config(self) -> None:
        """Load both personas and behaviors from the config file"""
        try:
            logger.debug(f"Loading config from {self.config_file}")
            with open(self.config_file) as f:
                data = json.load(f)

            # Load personas
            if "personas" not in data:
                logger.error("No 'personas' key found in config file")
                raise KeyError("Missing 'personas' key in config file")

            self.personas = [
                Persona(
                    name=p["name"],
                    traits=p["traits"],
                )
                for p in data.get("personas", [])
            ]
            logger.info(f"Loaded {len(self.personas)} personas")
            for persona in self.personas:
                logger.debug(
                    f"  - Persona: {persona.name} with {len(persona.traits)} traits"
                )

            # Load behaviors
            if "behaviors" not in data:
                logger.error("No 'behaviors' key found in config file")
                raise KeyError("Missing 'behaviors' key in config file")

            self.behaviors = [
                Behavior(
                    name=b["name"],
                    characteristics=b["characteristics"],
                )
                for b in data.get("behaviors", [])
            ]
            logger.info(f"Loaded {len(self.behaviors)} behaviors")
            for behavior in self.behaviors:
                logger.debug(
                    f"  - Behavior: {behavior.name} with {len(behavior.characteristics)} characteristics"
                )

        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {self.config_file}: {str(e)}")
            raise
        except KeyError as e:
            logger.error(f"Missing required field in config: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config: {str(e)}", exc_info=True)
            raise

    def get_persona(self, name: str) -> Optional[Persona]:
        """
        Get a specific persona by name

        Args:
            name: Name of the persona to retrieve

        Returns:
            The persona object if found, None otherwise
        """
        logger.debug(f"Looking up persona: {name}")
        for persona in self.personas:
            if persona.name.lower() == name.lower():
                return persona
        logger.warning(f"Persona not found: {name}")
        return None

    def get_behavior(self, name: str) -> Optional[Behavior]:
        """
        Get a specific behavior by name

        Args:
            name: Name of the behavior to retrieve

        Returns:
            The behavior object if found, None otherwise
        """
        logger.debug(f"Looking up behavior: {name}")
        for behavior in self.behaviors:
            if behavior.name.lower() == name.lower():
                return behavior
        logger.warning(f"Behavior not found: {name}")
        return None

    def list_personas(self) -> List[str]:
        """
        Get list of all persona names

        Returns:
            List of persona names
        """
        names = [persona.name for persona in self.personas]
        logger.debug(f"Listed {len(names)} personas")
        return names

    def list_behaviors(self) -> List[str]:
        """
        Get list of all behavior names

        Returns:
            List of behavior names
        """
        names = [behavior.name for behavior in self.behaviors]
        logger.debug(f"Listed {len(names)} behaviors")
        return names

    def validate_config(self) -> bool:
        """
        Validate that the config has all required fields

        Returns:
            True if config is valid, False otherwise
        """
        logger.info("Validating config...")

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

        logger.info("Config validation successful")
        return True

    def get_persona_behavior_combinations(self) -> List[Dict]:
        """
        Get all combinations of personas and behaviors

        Returns:
            List of dictionaries with persona and behavior objects
        """
        combinations = []
        for persona in self.personas:
            for behavior in self.behaviors:
                combinations.append({"persona": persona, "behavior": behavior})

        logger.debug(f"Generated {len(combinations)} persona-behavior combinations")
        return combinations
