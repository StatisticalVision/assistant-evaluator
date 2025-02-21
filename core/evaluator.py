from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import time
import logging
from llama_index.llms.openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.data_models import Conversation, Grade
from typing import List

logger = logging.getLogger(__name__)


class ConversationEvaluator:
    def __init__(
        self,
        conversations: List[Conversation],
        api_key: str,
        model: str = "gpt-4",
        batch_size: int = 5,
    ):
        self.llm = OpenAI(model=model, api_key=api_key)
        self.conversations = conversations
        self.batch_size = batch_size

    def evaluate_response(self, conv: Conversation) -> Conversation:
        """Evaluate a single response for empathy and accuracy"""

        system_prompt = """
        You are an expert conversation evaluator. Analyze the given customer service interaction
        and provide scores for empathy and accuracy. Return your evaluation as a JSON object.
        
        Evaluate:
        1. Empathy (0 or 1):
           - Understanding of customer's situation
           - Appropriate tone and language
           - Expression of support/concern
           
        2. Accuracy (0-1):
           - Correctness of information
           - Completeness of response
           - Alignment with expected answer
        
        Return dictionary format:
        {"empathy": float,"accuracy": float}
        """

        evaluation_prompt = f"""
        Customer Question: {conv.question}
        Expected Response: {conv.expected_outputs}
        Actual Response: {conv.actual_outputs}
        
        Evaluate the actual response compared to the expected response.
        """
        try:
            response = self.llm.complete(system_prompt + "\n\n" + evaluation_prompt)
            result = json.loads(response.text)

            conv.grade.empathy = result.get("empathy", -1)
            conv.grade.empathy = result.get("accuracy", -1)
            return conv
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return Grade(
                empathy=-1, accuracy=-1, response_time=conv.grade.response_time
            )
            return {"empathy": -1, "accuracy": -1}

    def process_conversations(self, output_path: Path) -> None:
        """Process all conversations in the CSV file"""
        # Read CSV
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        logger.info(f"Processing {total_rows} conversations")

        def process_batch(batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
            results = []
            for _, row in batch_df.iterrows():
                start_time = time.time()

                if pd.isna(row["actual_outputs"]) or row["actual_outputs"] == "":
                    results.append(
                        {
                            "conversation_id": row["conversation_id"],
                            "empathy": -1,
                            "accuracy": -1,
                            "response_time": -1,
                        }
                    )
                    continue

                evaluation = self.evaluate_response(
                    row["question"], row["expected_outputs"], row["actual_outputs"]
                )

                response_time = time.time() - start_time

                results.append(
                    {
                        "conversation_id": row["conversation_id"],
                        "empathy": evaluation.get("empathy", 0),
                        "accuracy": evaluation.get("accuracy", 0),
                        "response_time": response_time,
                    }
                )

            return results

        # Process in batches
        all_results = []
        for i in range(0, total_rows, self.batch_size):
            batch_df = df.iloc[i : i + self.batch_size]
            batch_results = process_batch(batch_df)
            all_results.extend(batch_results)

            # Log progress
            logger.info(
                f"Processed {min(i + self.batch_size, total_rows)}/{total_rows} conversations"
            )

        # Update DataFrame with results
        results_df = pd.DataFrame(all_results)
        df = df.merge(results_df, on="conversation_id", how="left")

        # Save updated CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    def process_conversations_parallel(
        self, csv_path: Path, output_path: Path, max_workers: int = 3
    ) -> None:
        """Process conversations in parallel with controlled concurrency"""
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        logger.info(f"Processing {total_rows} conversations with {max_workers} workers")

        def process_row(row):
            start_time = time.time()

            if pd.isna(row["actual_outputs"]) or row["actual_outputs"] == "":
                return {
                    "conversation_id": row["conversation_id"],
                    "empathy": -1,
                    "accuracy": -1,
                    "response_time": -1,
                }

            evaluation = self.evaluate_response(
                row["question"], row["expected_outputs"], row["actual_outputs"]
            )

            response_time = time.time() - start_time

            return {
                "conversation_id": row["conversation_id"],
                "empathy": evaluation.get("empathy", 0),
                "accuracy": evaluation.get("accuracy", 0),
                "response_time": response_time,
            }

        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {
                executor.submit(process_row, row): row for _, row in df.iterrows()
            }

            for future in as_completed(future_to_row):
                result = future.result()
                all_results.append(result)

                # Log progress
                logger.info(f"Processed {len(all_results)}/{total_rows} conversations")

        # Update DataFrame with results
        results_df = pd.DataFrame(all_results)
        df = df.merge(results_df, on="conversation_id", how="left")

        # Save updated CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
