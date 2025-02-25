import asyncio
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from core.conversation_generator import ConversationGenerator
from core.persona_manager import PersonaBehaviorManager
from models.data_models import KnowledgeBase
from core.evaluator import ConversationEvaluator
from utils.logging_config import setup_logging, get_logger

# Load environment variables (for API keys)
load_dotenv()


async def main():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate multi-turn conversations"
    )
    parser.add_argument(
        "--kb", type=str, default="data/kb.json", help="Knowledge base JSON file"
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="data/behaviorPersona.json",
        help="Personas and behaviors JSON file",
    )
    parser.add_argument(
        "--questions_per_faq",
        type=int,
        default=2,
        help="Number of question variations per FAQ",
    )
    parser.add_argument(
        "--out_of_scope_questions",
        type=int,
        default=1,
        help="Number of out-of-scope questions per persona",
    )
    parser.add_argument(
        "--turns", type=int, default=3, help="Number of turns per conversation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/multi_turn_conversations.csv",
        help="Output file for generated conversations",
    )
    parser.add_argument(
        "--generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generates conversations if true",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the generated conversations"
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default="output/evaluation_results.csv",
        help="Output file for evaluation results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/application.log",
        help="Path to log file (if not specified, logs to console only)",
    )
    parser.add_argument(
        "--log_max_bytes",
        type=int,
        default=10485760,  # 10MB
        help="Maximum log file size before rotation",
    )
    parser.add_argument(
        "--log_backup_count",
        type=int,
        default=5,
        help="Number of backup log files to keep",
    )
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(
        args.log_level, args.log_file, args.log_max_bytes, args.log_backup_count
    )

    # Get module-specific logger
    logger = get_logger(__name__)

    logger.info("Starting multi-turn conversation generation and evaluation")
    logger.debug(f"Command line arguments: {args}")

    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("Missing OpenAI API key. Set OPENAI_API_KEY environment variable.")
        raise ValueError(
            "Missing OpenAI API key. Set OPENAI_API_KEY environment variable."
        )

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    logger.debug("Created output directory if it didn't exist")

    # Step 1: Load knowledge base and personas
    logger.info(f"Loading knowledge base from {args.kb}...")
    knowledge_base = KnowledgeBase(args.kb)

    logger.info(f"Loading personas and behaviors from {args.personas}...")
    persona_manager = PersonaBehaviorManager(Path(args.personas))

    logger.info(
        f"Found {len(persona_manager.personas)} personas and {len(persona_manager.behaviors)} behaviors"
    )

    # Print what we're about to generate
    total_scenarios = len(persona_manager.personas) * len(persona_manager.behaviors)
    total_expected_questions = (
        total_scenarios * len(knowledge_base.faqs) * args.questions_per_faq
        + total_scenarios * args.out_of_scope_questions
    )
    logger.info(
        f"Will generate approximately {total_expected_questions} multi-turn conversations ({args.turns} turns each)"
    )

    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")

    # Step 2: Generate multi-turn conversations
    if args.generate:
        logger.info("Generating multi-turn conversations...")
        generator = ConversationGenerator(api_key=openai_api_key)
        logger.debug(f"ConversationGenerator initialized with API key")

        try:
            generator.generate_questions(
                knowledge_base=knowledge_base,
                personas=persona_manager.personas,
                behaviors=persona_manager.behaviors,
                questions_per_faq=args.questions_per_faq,
                out_of_scope_questions_per_persona=args.out_of_scope_questions,
                output_file=Path(args.output),
                turns_per_conversation=args.turns,
            )
            logger.info(f"Conversations generated and saved to {args.output}")
        except Exception as e:
            logger.error(f"Error generating conversations: {str(e)}", exc_info=True)
            raise

    # Step 3: Optionally evaluate the generated conversations
    if args.evaluate:
        logger.info("Loading conversations for evaluation...")
        # Load the generated conversations
        from models.data_models import Conversation

        try:
            conversations = Conversation.load_from_csv(Path(args.output))
            logger.info(f"Evaluating {len(conversations)} conversations...")

            # Initialize evaluator
            evaluator = ConversationEvaluator(
                conversations=conversations, api_key=openai_api_key, batch_size=5
            )
            logger.debug(f"ConversationEvaluator initialized with batch size 5")

            # Create a progress bar for the evaluation
            with tqdm(
                total=len(conversations), desc="Evaluating conversations", unit="conv"
            ) as eval_pbar:
                # Use a custom progress callback to update our tqdm bar
                def progress_callback(completed, total):
                    # Update to the exact position, rather than incrementing
                    eval_pbar.n = completed
                    eval_pbar.refresh()
                    # Log progress at certain milestones
                    if completed % 10 == 0 or completed == total:
                        logger.debug(
                            f"Evaluation progress: {completed}/{total} conversations"
                        )

                # Use parallel processing for faster evaluation
                max_workers = min(
                    5, os.cpu_count() or 4
                )  # Limit workers based on CPU count
                logger.debug(f"Using {max_workers} workers for parallel evaluation")

                evaluator.process_conversations_parallel(
                    Path(args.eval_output),
                    max_workers=max_workers,
                    progress_callback=progress_callback,
                )

            logger.info(f"Evaluation results saved to {args.eval_output}")

            # Generate a report
            logger.info("Generating evaluation report...")
            try:
                from core.report_generator import ReportGenerator

                report_generator = ReportGenerator()
                report = report_generator.generate_report(conversations)
                logger.info("Evaluation report generated successfully")

                # Log report summary
                logger.info("\nEvaluation Report Summary:")
                for line in report.split("\n"):
                    if line.startswith(("#", "##", "###")):  # Log headers as INFO
                        logger.info(line)
                    else:  # Log details as DEBUG
                        logger.debug(line)
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}", exc_info=True)

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        get_logger(__name__).warning("Process interrupted by user")
    except Exception as e:
        get_logger(__name__).exception(f"An error occurred: {str(e)}")
