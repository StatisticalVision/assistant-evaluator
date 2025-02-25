from typing import List, Dict, Any
import json
from datetime import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import random
from models.data_models import Conversation
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: Path = Path("reports")):
        """
        Initialize the report generator

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        logger.info(f"Initializing ReportGenerator with output directory: {output_dir}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Ensured report directory exists: {output_dir}")

    def generate_report(self, conversations: List[Conversation]) -> str:
        """
        Generate detailed evaluation report with progress indicators

        Args:
            conversations: List of evaluated conversations

        Returns:
            Formatted report text
        """
        logger.info(f"Generating report for {len(conversations)} conversations")
        print("Analyzing conversation data...")
        report_data = self._analyze_conversations(conversations)

        print("Formatting evaluation report...")
        report_text = self._format_report(report_data)

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{timestamp}.json"

        try:
            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Detailed report saved to {report_file}")

        except Exception as e:
            logger.error(f"Error saving report to file: {str(e)}", exc_info=True)

        return report_text

    def _analyze_conversations(
        self, conversations: List[Conversation]
    ) -> Dict[str, Any]:
        """
        Analyze conversations and compile statistics with progress bar

        Args:
            conversations: List of evaluated conversations

        Returns:
            Dictionary with analysis results
        """
        total_conversations = len(conversations)
        if not total_conversations:
            logger.warning("No conversations to analyze")
            return {"error": "No conversations to analyze"}

        # Compile statistics with progress reporting
        logger.info("Calculating metrics...")

        # Calculate average scores
        avg_scores = self._calculate_average_scores(conversations)

        # Analyze performance by persona
        logger.info("Analyzing per-persona performance...")
        persona_performance = self._analyze_persona_performance(conversations)

        # Analyze performance by behavior
        logger.info("Analyzing per-behavior performance...")
        behavior_performance = self._analyze_behavior_performance(conversations)

        # Compile detailed results - THIS IS THE METHOD THAT WAS MISSING
        logger.info("Compiling detailed results...")
        detailed_results = self.compile_detailed_results(
            conversations
        )  # RENAMED FROM _compile_detailed_results

        # Compile statistics
        stats = {
            "total_conversations": total_conversations,
            "average_scores": avg_scores,
            "persona_performance": persona_performance,
            "behavior_performance": behavior_performance,
            "detailed_results": detailed_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info("Analysis complete")
        return stats

    def _calculate_average_scores(
        self, conversations: List[Conversation]
    ) -> Dict[str, float]:
        """
        Calculate average scores across all conversations with progress bar

        Args:
            conversations: List of evaluated conversations

        Returns:
            Dictionary with average scores
        """
        all_scores = {
            "empathy": [],
            "accuracy": [],
            "response_time": [],
        }

        # Use tqdm to show progress
        for conv in tqdm(conversations, desc="Calculating average scores", unit="conv"):
            if conv.grade:
                if hasattr(conv.grade, "empathy") and conv.grade.empathy >= 0:
                    all_scores["empathy"].append(conv.grade.empathy)
                if hasattr(conv.grade, "accuracy") and conv.grade.accuracy >= 0:
                    all_scores["accuracy"].append(conv.grade.accuracy)
                if (
                    hasattr(conv.grade, "response_time")
                    and conv.grade.response_time >= 0
                ):
                    all_scores["response_time"].append(conv.grade.response_time)

        # Calculate the averages
        avg_scores = {}
        for metric, scores in all_scores.items():
            if scores:
                avg_scores[metric] = sum(scores) / len(scores)
                logger.debug(
                    f"Average {metric}: {avg_scores[metric]:.4f} from {len(scores)} samples"
                )
            else:
                avg_scores[metric] = 0
                logger.warning(f"No valid scores for {metric}")

        # Add an overall score if we have both empathy and accuracy
        if "empathy" in avg_scores and "accuracy" in avg_scores:
            avg_scores["overall"] = (avg_scores["empathy"] + avg_scores["accuracy"]) / 2
            logger.debug(f"Overall average score: {avg_scores['overall']:.4f}")

        return avg_scores

    def _analyze_persona_performance(self, conversations: List[Conversation]) -> Dict:
        """
        Analyze performance by persona with progress tracking

        Args:
            conversations: List of evaluated conversations

        Returns:
            Dictionary with performance metrics by persona
        """
        persona_stats = {}

        # Group conversations by persona
        for conv in tqdm(conversations, desc="Analyzing by persona", unit="conv"):
            persona_name = conv.persona.name
            if persona_name not in persona_stats:
                persona_stats[persona_name] = {
                    "total_conversations": 0,
                    "average_empathy": 0,
                    "average_accuracy": 0,
                    "average_response_time": 0,
                    "empathy_scores": [],
                    "accuracy_scores": [],
                    "response_times": [],
                }

            stats = persona_stats[persona_name]
            stats["total_conversations"] += 1

            if conv.grade:
                if hasattr(conv.grade, "empathy") and conv.grade.empathy >= 0:
                    stats["empathy_scores"].append(conv.grade.empathy)
                if hasattr(conv.grade, "accuracy") and conv.grade.accuracy >= 0:
                    stats["accuracy_scores"].append(conv.grade.accuracy)
                if (
                    hasattr(conv.grade, "response_time")
                    and conv.grade.response_time >= 0
                ):
                    stats["response_times"].append(conv.grade.response_time)

        # Calculate averages for each persona
        for persona_name, stats in persona_stats.items():
            logger.debug(f"Calculating averages for persona: {persona_name}")

            if stats["empathy_scores"]:
                stats["average_empathy"] = sum(stats["empathy_scores"]) / len(
                    stats["empathy_scores"]
                )
                logger.debug(
                    f"  Persona {persona_name} avg empathy: {stats['average_empathy']:.4f}"
                )
            else:
                logger.warning(f"  No valid empathy scores for persona {persona_name}")

            if stats["accuracy_scores"]:
                stats["average_accuracy"] = sum(stats["accuracy_scores"]) / len(
                    stats["accuracy_scores"]
                )
                logger.debug(
                    f"  Persona {persona_name} avg accuracy: {stats['average_accuracy']:.4f}"
                )
            else:
                logger.warning(f"  No valid accuracy scores for persona {persona_name}")

            if stats["response_times"]:
                stats["average_response_time"] = sum(stats["response_times"]) / len(
                    stats["response_times"]
                )
                logger.debug(
                    f"  Persona {persona_name} avg response time: {stats['average_response_time']:.4f}"
                )
            else:
                logger.warning(f"  No valid response times for persona {persona_name}")

            # Calculate overall score
            if stats["average_empathy"] > 0 and stats["average_accuracy"] > 0:
                stats["average_overall"] = (
                    stats["average_empathy"] + stats["average_accuracy"]
                ) / 2
                logger.debug(
                    f"  Persona {persona_name} overall score: {stats['average_overall']:.4f}"
                )

            # Remove the raw score lists to keep the report clean
            del stats["empathy_scores"]
            del stats["accuracy_scores"]
            del stats["response_times"]

        return persona_stats

    def _analyze_behavior_performance(self, conversations: List[Conversation]) -> Dict:
        """
        Analyze performance by behavior with progress tracking

        Args:
            conversations: List of evaluated conversations

        Returns:
            Dictionary with performance metrics by behavior
        """
        behavior_stats = {}

        # Group conversations by behavior
        for conv in tqdm(conversations, desc="Analyzing by behavior", unit="conv"):
            behavior_name = conv.behavior.name
            if behavior_name not in behavior_stats:
                behavior_stats[behavior_name] = {
                    "total_conversations": 0,
                    "average_empathy": 0,
                    "average_accuracy": 0,
                    "average_response_time": 0,
                    "empathy_scores": [],
                    "accuracy_scores": [],
                    "response_times": [],
                }

            stats = behavior_stats[behavior_name]
            stats["total_conversations"] += 1

            if conv.grade:
                if hasattr(conv.grade, "empathy") and conv.grade.empathy >= 0:
                    stats["empathy_scores"].append(conv.grade.empathy)
                if hasattr(conv.grade, "accuracy") and conv.grade.accuracy >= 0:
                    stats["accuracy_scores"].append(conv.grade.accuracy)
                if (
                    hasattr(conv.grade, "response_time")
                    and conv.grade.response_time >= 0
                ):
                    stats["response_times"].append(conv.grade.response_time)

        # Calculate averages for each behavior
        for behavior_name, stats in behavior_stats.items():
            logger.debug(f"Calculating averages for behavior: {behavior_name}")

            if stats["empathy_scores"]:
                stats["average_empathy"] = sum(stats["empathy_scores"]) / len(
                    stats["empathy_scores"]
                )
                logger.debug(
                    f"  Behavior {behavior_name} avg empathy: {stats['average_empathy']:.4f}"
                )
            else:
                logger.warning(
                    f"  No valid empathy scores for behavior {behavior_name}"
                )

            if stats["accuracy_scores"]:
                stats["average_accuracy"] = sum(stats["accuracy_scores"]) / len(
                    stats["accuracy_scores"]
                )
                logger.debug(
                    f"  Behavior {behavior_name} avg accuracy: {stats['average_accuracy']:.4f}"
                )
            else:
                logger.warning(
                    f"  No valid accuracy scores for behavior {behavior_name}"
                )

            if stats["response_times"]:
                stats["average_response_time"] = sum(stats["response_times"]) / len(
                    stats["response_times"]
                )
                logger.debug(
                    f"  Behavior {behavior_name} avg response time: {stats['average_response_time']:.4f}"
                )
            else:
                logger.warning(
                    f"  No valid response times for behavior {behavior_name}"
                )

            # Calculate overall score
            if stats["average_empathy"] > 0 and stats["average_accuracy"] > 0:
                stats["average_overall"] = (
                    stats["average_empathy"] + stats["average_accuracy"]
                ) / 2
                logger.debug(
                    f"  Behavior {behavior_name} overall score: {stats['average_overall']:.4f}"
                )

            # Remove the raw score lists to keep the report clean
            del stats["empathy_scores"]
            del stats["accuracy_scores"]
            del stats["response_times"]

        return behavior_stats

    def compile_detailed_results(self, conversations: List[Conversation]) -> List[Dict]:
        """
        Compile detailed results for each conversation

        Args:
            conversations: List of evaluated conversations

        Returns:
            List of dictionaries with detailed results for each conversation
        """
        detailed_results = []
        logger.info(
            f"Compiling detailed results for {len(conversations)} conversations"
        )

        for conv in tqdm(conversations, desc="Compiling detailed results", unit="conv"):
            # Extract sample exchanges (first 2 turns or fewer if not available)
            sample_exchanges = []
            max_turns_to_include = min(2, len(conv.conversation_turns))

            for i, turn in enumerate(conv.conversation_turns[:max_turns_to_include]):
                sample_exchanges.append(
                    {
                        "turn": i + 1,
                        "user": turn.user_message,
                        "assistant": turn.ai_response,
                    }
                )

            logger.debug(
                f"Compiled details for conversation {conv.id} with {len(sample_exchanges)} sample exchanges"
            )

            # Add conversation details to results
            result = {
                "conversation_id": conv.id,
                "persona": conv.persona.name,
                "behavior": conv.behavior.name,
                "turns_count": len(conv.conversation_turns),
                "initial_question": conv.question,
                "expected_output": conv.expected_outputs,
                "grades": {
                    "empathy": conv.grade.empathy if conv.grade else -1,
                    "accuracy": conv.grade.accuracy if conv.grade else -1,
                    "response_time": conv.grade.response_time if conv.grade else -1,
                },
                "sample_exchanges": sample_exchanges,
            }

            detailed_results.append(result)

        return detailed_results

    def _format_report(self, report_data: Dict) -> str:
        """
        Format the report data into a readable text summary

        Args:
            report_data: Dictionary with analysis results

        Returns:
            Formatted report text
        """
        logger.info("Formatting evaluation report")

        if "error" in report_data:
            error_msg = f"Error: {report_data['error']}"
            logger.error(error_msg)
            return error_msg

        # Create a formatted summary report
        summary = []
        summary.append("# Conversation Evaluation Report")
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary.append(f"Generated: {report_time}")
        summary.append("")

        logger.debug(f"Creating report dated {report_time}")

        # Overall statistics
        logger.debug("Adding overall statistics to report")
        summary.append("## Overall Statistics")
        summary.append(
            f"Total conversations evaluated: {report_data['total_conversations']}"
        )

        # Average scores
        logger.debug("Adding average scores to report")
        avg_scores = report_data["average_scores"]
        summary.append("\n## Average Scores")
        if "empathy" in avg_scores:
            summary.append(f"Empathy: {avg_scores['empathy']:.2f}/1.0")
        if "accuracy" in avg_scores:
            summary.append(f"Accuracy: {avg_scores['accuracy']:.2f}/1.0")
        if "overall" in avg_scores:
            summary.append(f"Overall: {avg_scores['overall']:.2f}/1.0")
        if "response_time" in avg_scores:
            summary.append(
                f"Average response time: {avg_scores['response_time']:.2f} seconds"
            )

        # Persona performance
        logger.debug("Adding persona performance to report")
        summary.append("\n## Performance by Persona")
        for persona, stats in report_data["persona_performance"].items():
            logger.debug(f"Adding report section for persona: {persona}")
            summary.append(f"\n### {persona}")
            summary.append(f"Conversations: {stats['total_conversations']}")
            if "average_empathy" in stats and stats["average_empathy"] > 0:
                summary.append(f"Empathy: {stats['average_empathy']:.2f}/1.0")
            if "average_accuracy" in stats and stats["average_accuracy"] > 0:
                summary.append(f"Accuracy: {stats['average_accuracy']:.2f}/1.0")
            if "average_overall" in stats:
                summary.append(f"Overall: {stats['average_overall']:.2f}/1.0")

        # Behavior performance
        logger.debug("Adding behavior performance to report")
        summary.append("\n## Performance by Behavior")
        for behavior, stats in report_data["behavior_performance"].items():
            logger.debug(f"Adding report section for behavior: {behavior}")
            summary.append(f"\n### {behavior}")
            summary.append(f"Conversations: {stats['total_conversations']}")
            if "average_empathy" in stats and stats["average_empathy"] > 0:
                summary.append(f"Empathy: {stats['average_empathy']:.2f}/1.0")
            if "average_accuracy" in stats and stats["average_accuracy"] > 0:
                summary.append(f"Accuracy: {stats['average_accuracy']:.2f}/1.0")
            if "average_overall" in stats:
                summary.append(f"Overall: {stats['average_overall']:.2f}/1.0")

        # Sample results
        logger.debug("Adding sample conversations to report")
        summary.append("\n## Sample Conversations")

        # Get samples - one high-scoring, one medium-scoring, and one low-scoring if possible
        detailed_results = report_data["detailed_results"]

        # Sort conversations by overall score (average of empathy and accuracy)
        def calc_overall_score(conv):
            grades = conv["grades"]
            empathy = grades["empathy"] if grades["empathy"] >= 0 else 0
            accuracy = grades["accuracy"] if grades["accuracy"] >= 0 else 0
            if empathy > 0 and accuracy > 0:
                return (empathy + accuracy) / 2
            elif empathy > 0:
                return empathy
            elif accuracy > 0:
                return accuracy
            else:
                return 0

        # Try to get a diverse set of samples
        try:
            valid_convs = [c for c in detailed_results if calc_overall_score(c) > 0]

            if len(valid_convs) >= 3:
                # Sort by score
                sorted_convs = sorted(valid_convs, key=calc_overall_score)

                # Get low, medium, and high scoring samples
                low_sample = sorted_convs[0]
                mid_idx = len(sorted_convs) // 2
                mid_sample = sorted_convs[mid_idx]
                high_sample = sorted_convs[-1]

                samples = [low_sample, mid_sample, high_sample]
                logger.debug("Selected diverse samples (low, medium, high scores)")
            else:
                # If we don't have enough scored conversations, use random selection
                sample_size = min(3, len(detailed_results))
                samples = random.sample(detailed_results, sample_size)
                logger.debug(f"Selected {sample_size} random samples")
        except Exception as e:
            logger.warning(f"Error selecting diverse samples: {str(e)}")
            # Fallback to random selection
            sample_size = min(3, len(detailed_results))
            samples = random.sample(detailed_results, sample_size)
            logger.debug(f"Fell back to {sample_size} random samples")

        # Add samples to report
        for i, sample in enumerate(samples):
            sample_id = sample["conversation_id"]
            logger.debug(f"Adding sample {i+1} (ID: {sample_id})")

            summary.append(f"\n### Sample {i+1} (ID: {sample_id})")
            summary.append(f"Persona: {sample['persona']}")
            summary.append(f"Behavior: {sample['behavior']}")
            summary.append(f"Turns: {sample['turns_count']}")
            summary.append(f"Question: {sample['initial_question']}")

            # Show the exchange
            summary.append("\nExchanges:")
            for exchange in sample["sample_exchanges"]:
                summary.append(f"\nTurn {exchange['turn']}:")
                summary.append(f"User: {exchange['user']}")
                summary.append(f"Assistant: {exchange['assistant']}")

            # Show grades
            grades = sample["grades"]
            summary.append("\nGrades:")
            empathy_score = grades["empathy"]
            accuracy_score = grades["accuracy"]

            if empathy_score >= 0:
                summary.append(f"Empathy: {empathy_score:.2f}/1.0")
            else:
                summary.append("Empathy: Not evaluated")

            if accuracy_score >= 0:
                summary.append(f"Accuracy: {accuracy_score:.2f}/1.0")
            else:
                summary.append("Accuracy: Not evaluated")

            # Add overall score if both metrics are available
            if empathy_score >= 0 and accuracy_score >= 0:
                overall = (empathy_score + accuracy_score) / 2
                summary.append(f"Overall: {overall:.2f}/1.0")

        logger.info("Report formatting complete")
        return "\n".join(summary)
