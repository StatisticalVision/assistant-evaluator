from typing import List, Dict
import json
from datetime import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from models.data_models import Conversation

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(self, conversations: List[Conversation]) -> str:
        """Generate detailed evaluation report with progress indicators"""
        print("Analyzing conversation data...")
        report_data = self._analyze_conversations(conversations)

        print("Formatting evaluation report...")
        report_text = self._format_report(report_data)

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Report generated and saved to {report_file}")
        return report_text

    def _analyze_conversations(self, conversations: List[Conversation]) -> Dict:
        """Analyze conversations and compile statistics with progress bar"""
        total_conversations = len(conversations)
        if not total_conversations:
            return {"error": "No conversations to analyze"}

        # Compile statistics with progress reporting
        print("Calculating metrics...")

        # Calculate average scores
        avg_scores = self._calculate_average_scores(conversations)

        # Analyze performance by persona
        print("Analyzing per-persona performance...")
        persona_performance = self._analyze_persona_performance(conversations)

        # Compile detailed results
        print("Compiling detailed results...")
        detailed_results = self._compile_detailed_results(conversations)

        # Compile statistics
        stats = {
            "total_conversations": total_conversations,
            "average_scores": avg_scores,
            "persona_performance": persona_performance,
            "detailed_results": detailed_results,
        }

        return stats

    def _calculate_average_scores(
        self, conversations: List[Conversation]
    ) -> Dict[str, float]:
        """Calculate average scores across all conversations with progress bar"""
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
            else:
                avg_scores[metric] = 0

        # Add an overall score if we have both empathy and accuracy
        if "empathy" in avg_scores and "accuracy" in avg_scores:
            avg_scores["overall"] = (avg_scores["empathy"] + avg_scores["accuracy"]) / 2

        return avg_scores

    def _analyze_persona_performance(self, conversations: List[Conversation]) -> Dict:
        """Analyze performance by persona with progress tracking"""
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
            if stats["empathy_scores"]:
                stats["average_empathy"] = sum(stats["empathy_scores"]) / len(
                    stats["empathy_scores"]
                )
            if stats["accuracy_scores"]:
                stats["average_accuracy"] = sum(stats["accuracy_scores"]) / len(
                    stats["accuracy_scores"]
                )
            if stats["response_times"]:
                stats["average_response_time"] = sum(stats["response_times"]) / len(
                    stats["response_times"]
                )

            # Calculate overall score
            if stats["average_empathy"] > 0 and stats["average_accuracy"] > 0:
                stats["average_overall"] = (
                    stats["average_empathy"] + stats["average_accuracy"]
                ) / 2

            # Remove the raw score lists to keep the report clean
            del stats["empathy_scores"]
            del stats["accuracy_scores"]
            del stats["response_times"]

        return persona_stats

    def _compile_detailed_results(
        self, conversations: List[Conversation]
    ) -> List[Dict]:
        """Compile detailed results for each conversation"""
        detailed_results = []

        for conv in tqdm(conversations, desc="Compiling detailed results", unit="conv"):
            # Extract sample exchanges (first 2 turns)
            sample_exchanges = []
            for i, turn in enumerate(
                conv.conversation_turns[:2]
            ):  # Only include first 2 exchanges
                sample_exchanges.append(
                    {
                        "turn": i + 1,
                        "user": turn.user_message,
                        "assistant": turn.ai_response,
                    }
                )

            # Add conversation details to results
            detailed_results.append(
                {
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
            )

        return detailed_results

    def _format_report(self, report_data: Dict) -> str:
        """Format the report data into a readable text summary"""
        if "error" in report_data:
            return f"Error: {report_data['error']}"

        # Create a formatted summary report
        summary = []
        summary.append("# Conversation Evaluation Report")
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")

        # Overall statistics
        summary.append("## Overall Statistics")
        summary.append(
            f"Total conversations evaluated: {report_data['total_conversations']}"
        )

        # Average scores
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
        summary.append("\n## Performance by Persona")
        for persona, stats in report_data["persona_performance"].items():
            summary.append(f"\n### {persona}")
            summary.append(f"Conversations: {stats['total_conversations']}")
            if "average_empathy" in stats and stats["average_empathy"] > 0:
                summary.append(f"Empathy: {stats['average_empathy']:.2f}/1.0")
            if "average_accuracy" in stats and stats["average_accuracy"] > 0:
                summary.append(f"Accuracy: {stats['average_accuracy']:.2f}/1.0")
            if "average_overall" in stats:
                summary.append(f"Overall: {stats['average_overall']:.2f}/1.0")

        # Sample results
        summary.append("\n## Sample Conversations")
        # Include details for 3 random conversations as samples
        import random

        sample_size = min(3, len(report_data["detailed_results"]))
        samples = random.sample(report_data["detailed_results"], sample_size)

        for i, sample in enumerate(samples):
            summary.append(f"\n### Sample {i+1} (ID: {sample['conversation_id']})")
            summary.append(f"Persona: {sample['persona']}")
            summary.append(f"Behavior: {sample['behavior']}")
            summary.append(f"Turns: {sample['turns_count']}")

            # Show the exchange
            summary.append("\nExchanges:")
            for exchange in sample["sample_exchanges"]:
                summary.append(f"\nTurn {exchange['turn']}:")
                summary.append(f"User: {exchange['user']}")
                summary.append(f"Assistant: {exchange['assistant']}")

            # Show grades
            grades = sample["grades"]
            summary.append("\nGrades:")
            summary.append(f"Empathy: {grades['empathy']:.2f}/1.0")
            summary.append(f"Accuracy: {grades['accuracy']:.2f}/1.0")

        return "\n".join(summary)
