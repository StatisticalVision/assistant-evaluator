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

            # Also save a text version
            text_report_file = self.output_dir / f"evaluation_report_{timestamp}.txt"
            with open(text_report_file, "w") as f:
                f.write(report_text)

            logger.info(f"Text report saved to {text_report_file}")

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

        # Compile detailed results
        logger.info("Compiling detailed results...")
        detailed_results = self._compile_detailed_results(conversations)

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
