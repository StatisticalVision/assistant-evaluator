from typing import List, Dict
import json
from datetime import datetime
import logging
from pathlib import Path
from models.data_models import Conversation

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(self, conversations: List[Conversation]) -> str:
        """Generate detailed evaluation report"""
        report_data = self._analyze_conversations(conversations)
        report_text = self._format_report(report_data)

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Report generated and saved to {report_file}")
        return report_text

    def _analyze_conversations(self, conversations: List[Conversation]) -> Dict:
        """Analyze conversations and compile statistics"""
        total_conversations = len(conversations)
        if not total_conversations:
            return {"error": "No conversations to analyze"}

        # Compile statistics
        stats = {
            "total_conversations": total_conversations,
            "average_scores": self._calculate_average_scores(conversations),
            "persona_performance": self._analyze_persona_performance(conversations),
            "detailed_results": self._compile_detailed_results(conversations),
        }

        return stats

    def _calculate_average_scores(
        self, conversations: List[Conversation]
    ) -> Dict[str, float]:
        """Calculate average scores across all conversations"""
        all_scores = {
            "accuracy": [],
            "tone_matching": [],
            "completeness": [],
            "final_score": [],
        }

        for conv in conversations:
            if conv.grades:
                for metric, score in conv.grades.items():
                    if metric in all_scores:
                        all_scores[metric].append(score)

        return {
            metric: sum(scores) / len(scores) if scores else 0
            for metric, scores in all_scores.items()
        }

    def _analyze_persona_performance(self, conversations: List[Conversation]) -> Dict:
        """Analyze performance by persona"""
        persona_stats = {}

        for conv in conversations:
            persona_name = conv.persona.name
            if persona_name not in persona_stats:
                persona_stats[persona_name] = {
                    "total_conversations": 0,
                    "average_score": 0,
                    "scores": [],
                }

            stats = persona_stats[persona_name]
            stats["total_conversations"] += 1
            if conv.grades and "final_score" in conv.grades:
                stats["scores"].append(conv.grades["final_score"])

        # Calculate averages
        for stats in persona_stats.values():
            if stats["scores"]:
                stats["average_score"] = sum(stats["scores"]) / len(stats["scores"])
            del stats["scores"]  # Remove raw scores from final report

        return persona_stats

    def _compile_detailed_results(
        self, conversations: List[Conversation]
    ) -> List[Dict]:
        """Compile detailed results for each conversation"""
        return [
            {
                "conversation_id": conv.id,
                "persona": conv.persona.name,
                "num_turns": len(conv.question),
                "grades": conv.grades,
                "sample_exchanges": [
                    {"input": i, "expected": e, "actual": a}
                    for i, e, a in zip(
                        # Only include first 2 exchanges as samples
                        conv.question[:2],
                        conv.expected_outputs[:2],
                        conv.actual_outputs[:2],
                    )
                ],
            }
            for conv in conversations
        ]
