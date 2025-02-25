# Conversation Simulator

A comprehensive framework for generating, evaluating, and analyzing simulated customer service conversations based on different customer personas and behaviors.

## Overview

This project provides tools to:

1. Generate realistic multi-turn conversations between customers and support agents
2. Simulate different customer personas (Tech-Savvy, Non-Native Speaker, etc.)
3. Model various customer behaviors (frustrated, confused, urgent)
4. Evaluate conversation quality based on empathy and accuracy
5. Generate detailed reports on conversation performance

## Features

- **Persona Management**: Define and load customer personas with specific traits
- **Behavior Simulation**: Model different customer behaviors and interaction styles
- **Knowledge Base Integration**: Load FAQs and other support content for accurate responses
- **Conversation Generation**: Create multi-turn, realistic conversations based on personas and behaviors
- **Quality Evaluation**: Assess conversations based on empathy, accuracy, and response time
- **Reporting**: Generate detailed reports with performance breakdowns by persona and behavior

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key (for using the GPT models)

### Installation

1. Clone this repository
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key as an environment variable or create a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Project Structure

```
conversation-simulator/
├── config/                   # Configuration files and settings
├── core/                     # Core functionality modules
│   ├── conversation_generator.py  # Generates multi-turn conversations
│   ├── evaluator.py          # Evaluates conversation quality
│   ├── persona_manager.py    # Manages persona and behavior definitions
│   └── report_generator.py   # Creates performance reports
├── data/                     # Data files
│   ├── behaviorPersona.json  # Persona and behavior definitions
│   ├── instructions.txt      # Agent instructions
│   └── kb.json               # Knowledge base with FAQs
├── models/                   # Data models
│   └── data_models.py        # Dataclasses for conversations, personas, etc.
├── utils/                    # Utility functions
│   └── logging_config.py     # Logging configuration
├── .devcontainer/            # Development container configuration
├── .gitignore                # Git ignore file
├── main.py                   # Main application entry point
└── requirements.txt          # Python dependencies
```

## Usage

### Basic Usage

Run the main script with default settings:

```bash
python main.py
```

This will:
1. Load the knowledge base and persona definitions
2. Generate conversations based on the default settings
3. Save the generated conversations to `output/multi_turn_conversations.csv`

### Advanced Usage

Customize the generation and evaluation with command-line arguments:

```bash
python main.py \
  --kb data/kb.json \
  --personas data/behaviorPersona.json \
  --questions_per_faq 5 \
  --out_of_scope_questions 2 \
  --turns 4 \
  --output output/custom_conversations.csv \
  --evaluate \
  --eval_output output/evaluation_results.csv \
  --log_level DEBUG
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--kb` | Knowledge base JSON file | `data/kb.json` |
| `--personas` | Personas and behaviors JSON file | `data/behaviorPersona.json` |
| `--questions_per_faq` | Number of question variations per FAQ | `2` |
| `--out_of_scope_questions` | Number of out-of-scope questions per persona | `1` |
| `--turns` | Number of turns per conversation | `3` |
| `--output` | Output file for generated conversations | `output/multi_turn_conversations.csv` |
| `--generate/--no-generate` | Enable/disable conversation generation | `True` |
| `--evaluate` | Evaluate the generated conversations | `False` |
| `--eval_output` | Output file for evaluation results | `output/evaluation_results.csv` |
| `--log_level` | Set the logging level | `INFO` |
| `--log_file` | Path to log file | `logs/application.log` |
| `--log_max_bytes` | Maximum log file size before rotation | `10485760` (10MB) |
| `--log_backup_count` | Number of backup log files to keep | `5` |

## Customizing Personas and Behaviors

Modify the `data/behaviorPersona.json` file to define your own personas and behaviors. Each persona has a name and a list of traits, while behaviors have a name and a list of characteristics:

```json
{
  "personas": [
    {
      "name": "Tech-Savvy",
      "traits": ["knowledgeable", "efficient", "solution-oriented", "independent", "precise"]
    },
    ...
  ],
  "behaviors": [
    {
      "name": "frustrated",
      "characteristics": ["shows impatience", "may use stronger language", "emphasizes urgency", "references previous attempts", "seeks immediate resolution"]
    },
    ...
  ]
}
```

## Knowledge Base

The knowledge base in `data/kb.json` contains FAQs that are used to generate conversations. You can add or modify questions and answers as needed:

```json
{
  "faqs": [
    {
      "How can I find my Sendero Health Plan Member ID Card?": "You should have received an email with your digital ID cards on Dec. 24 or 26, 2024...",
      "How do I get into the Member Portal?": "Using your Member ID number, you can sign up for our Sendero Member Portal...",
      ...
    }
  ]
}
```

## Development

### Development Container

This project includes DevContainer configuration for Visual Studio Code, making it easy to set up a consistent development environment.

### Logging

Configure logging using the `--log_level`, `--log_file`, `--log_max_bytes`, and `--log_backup_count` command-line arguments.

## Evaluation Process

The conversation evaluation assesses:

1. **Empathy** (0 to 1 scale):
   - Understanding of customer's situation
   - Appropriate tone and language
   - Expression of support/concern
   - Consistency across multiple interactions

2. **Accuracy** (0 to 1 scale):
   - Correctness of information
   - Completeness of response
   - Alignment with expected answer
   - Consistency of information across turns

## Report Generation

The system generates detailed reports with:

- Overall statistics and average scores
- Performance breakdowns by persona
- Performance breakdowns by behavior
- Sample conversations with evaluation scores