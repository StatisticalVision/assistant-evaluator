# Sendero Conversation Evaluator

A tool for generating, simulating, and evaluating multi-turn conversations for Sendero Health Plans' customer service interactions. This system helps measure and improve the quality of AI-assisted customer support by analyzing empathy, accuracy, and response time metrics.

## Overview

This project provides a complete workflow for testing and evaluating customer service conversations:

1. **Generate** realistic customer questions based on personas and behaviors
2. **Simulate** multi-turn conversations with an AI assistant
3. **Evaluate** the quality of responses based on empathy and accuracy
4. **Report** on performance metrics to identify areas for improvement

## Features

- **Persona Management**: Create and manage customer personas with specific traits
- **Behavior Simulation**: Model different customer behaviors (frustrated, confused, urgent)
- **Knowledge Base Integration**: Test against real FAQs and company knowledge
- **Multi-turn Conversation**: Generate realistic back-and-forth customer interactions
- **Automated Evaluation**: Score conversations on empathy, accuracy, and response time
- **Comprehensive Reporting**: Generate detailed reports on conversation quality

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sendero-conversation-evaluator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create .env file with your API keys
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

### Using Docker

Alternatively, you can use the included DevContainer configuration with VS Code:

1. Ensure Docker and VS Code with Remote-Containers extension are installed
2. Open the project in VS Code
3. Click "Reopen in Container" when prompted
4. The container will automatically install all dependencies

## Usage

### Generating Conversations

Generate multi-turn conversations based on personas and behaviors:

```bash
python main.py --kb data/kb.json --personas data/behaviorPersona.json --questions_per_faq 2 --turns 3 --output output/conversations.csv
```

Options:
- `--kb`: Path to knowledge base JSON file
- `--personas`: Path to personas/behaviors JSON file
- `--questions_per_faq`: Number of variations per FAQ to generate
- `--out_of_scope_questions`: Number of out-of-scope questions per persona
- `--turns`: Number of turns per conversation
- `--output`: Output file path for generated conversations
- `--no-generate` Skip the conversation generation step if one already exists

### Evaluating Conversations

Evaluate the quality of generated or real conversations:

```bash
python main.py --evaluate --output output/conversations.csv --eval_output output/evaluation_results.csv
```

Additional options:
- `--evaluate`: Enable evaluation mode
- `--eval_output`: Path for evaluation results

### Testing with Live API

Test individual conversations with the OpenAI API:

```bash
python conversation_evaluator.py --input test_cases.csv --output results.csv
```

## Project Structure

```
sendero-conversation-evaluator/
├── core/                      # Core functionality modules
│   ├── assistant_interface.py # Interface to AI assistant API
│   ├── conversation_generator.py # Generates multi-turn conversations
│   ├── evaluator.py           # Evaluates conversation quality
│   ├── persona_manager.py     # Manages personas and behaviors
│   └── report_generator.py    # Creates evaluation reports
├── data/                      # Data files
│   ├── behaviorPersona.json   # Personas and behaviors definitions
│   ├── instructions.txt       # System instructions for AI
│   └── kb.json                # Knowledge base with FAQs
├── models/                    # Data models
│   └── data_models.py         # Core data classes
├── utils/                     # Utility functions
│   └── logging_config.py      # Logging configuration
├── .devcontainer/             # DevContainer configuration
├── conversation_evaluator.py  # Entry point for testing
├── generate_conversations.py  # Entry point for generation
├── main.py                    # Main application entry point
└── requirements.txt           # Project dependencies
```

## Configuration

### Knowledge Base

The knowledge base is defined in `data/kb.json` and contains:
- FAQs with questions and answers
- IVR script information
- Other structured knowledge

### Personas and Behaviors

Customer personas and behaviors are defined in `data/behaviorPersona.json`:
- **Personas**: Different types of customers (Tech-Savvy, Non-Native Speaker, etc.)
- **Behaviors**: How customers might behave (frustrated, confused, urgent)

## Evaluation Metrics

The system evaluates conversations based on:
- **Empathy**: How well the assistant understands and addresses customer emotions
- **Accuracy**: Correctness and completeness of information provided
- **Response Time**: How quickly responses are generated

## Reports

Evaluation reports include:
- Overall statistics across all conversations
- Performance breakdown by persona and behavior
- Detailed results for individual conversations
- Sample conversation exchanges

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

[Your license information here]

## Contact

[Your contact information here]