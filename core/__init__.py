from .persona_manager import PersonaBehaviorManager
from .conversation_generator import ConversationGenerator
from .evaluator import ConversationEvaluator
from .report_generator import ReportGenerator

__all__ = [
    "PersonaManager",
    "ConversationGenerator",
    "AssistantInterface",
    "ConversationEvaluator",
    "ReportGenerator",
]

# Version info
__version__ = "0.1.0"

# Optional: Add any package-level constants or settings
DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_LOG_LEVEL = "INFO"


# Optional: Add any package initialization code if needed
def initialize_core(config_path=DEFAULT_CONFIG_PATH):
    """Initialize all core components with given configuration."""
    # Add initialization logic here if needed
    pass
