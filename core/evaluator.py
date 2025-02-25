from typing import List, Callable, Optional
import pandas as pd
from pathlib import Path
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.data_models import Conversation, Grade
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ConversationEvaluator:
    def __init__(
        self,
        conversations: List[Conversation],
        api_key: str,
        model: str = "gpt-4",
        batch_size: int = 5,
    ):
        """
        Initialize the evaluator with conversations to evaluate

        Args:
            conversations: List of conversations to evaluate
            api_key: OpenAI API key
            model: Model to use for evaluation
            batch_size: Batch size for parallel processing
        """
        from llama_index.llms.openai import OpenAI

        self.llm = OpenAI(model=model, api_key=api_key)
        self.conversations = conversations
        self.batch_size = batch_size
        logger.info(
            f"Initialized ConversationEvaluator with {len(conversations)} conversations using model {model}"
        )
        logger.debug(f"Using batch size {batch_size}")

    def evaluate_response(self, conv: Conversation) -> Conversation:
        """
        Evaluate a full conversation for empathy and accuracy

        Args:
            conv: Conversation to evaluate

        Returns:
            The same conversation with updated grade
        """
        logger.debug(f"Evaluating conversation {conv.id}")

        system_prompt = """
        You are an expert conversation evaluator. Analyze the given customer service interaction
        and provide scores for empathy and accuracy. Return your evaluation as a JSON object.
        
        Evaluate:
        1. Empathy (0 to 1 scale):
           - Understanding of customer's situation
           - Appropriate tone and language
           - Expression of support/concern
           - Consistency across multiple interactions
           
        2. Accuracy (0 to 1 scale):
           - Correctness of information
           - Completeness of response
           - Alignment with expected answer
           - Consistency of information across turns
        
        Return dictionary format:
        {"empathy": float,"accuracy": float}
        """

        # Format the full conversation
        formatted_conversation = ""
        for i, turn in enumerate(conv.conversation_turns):
            formatted_conversation += f"Turn {i+1}:\n"
            formatted_conversation += f"User: {turn.user_message}\n"
            formatted_conversation += f"Assistant: {turn.ai_response}\n\n"

        evaluation_prompt = f"""
        Full Customer Conversation: 
        {formatted_conversation}
        
        Expected Response/Information: {conv.expected_outputs}
        
        Evaluate the entire conversation, considering all interactions between the user and assistant.
        Focus on whether the assistant maintained empathy throughout the conversation and
        provided accurate information that aligned with the expected response.
        """

        try:
            logger.debug(f"Sending evaluation request for conversation {conv.id}")
            response = self.llm.complete(system_prompt + "\n\n" + evaluation_prompt)

            try:
                result = json.loads(response.text)

                conv.grade.empathy = result.get("empathy", -1)
                conv.grade.accuracy = result.get("accuracy", -1)

                logger.debug(
                    f"Conversation {conv.id} evaluated - Empathy: {conv.grade.empathy:.2f}, Accuracy: {conv.grade.accuracy:.2f}"
                )
                return conv
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse JSON response for conversation {conv.id}: {response.text[:100]}..."
                )
                conv.grade = Grade(
                    empathy=-1,
                    accuracy=-1,
                    response_time=conv.grade.response_time if conv.grade else -1,
                )
                return conv

        except Exception as e:
            logger.error(f"Error evaluating conversation {conv.id}: {str(e)}")
            conv.grade = Grade(
                empathy=-1,
                accuracy=-1,
                response_time=conv.grade.response_time if conv.grade else -1,
            )
            return conv

    def process_conversations(
        self,
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Process all conversations sequentially

        Args:
            output_path: Path to save results
            progress_callback: Optional callback for progress reporting
        """
        total_conversations = len(self.conversations)
        logger.info(f"Processing {total_conversations} conversations sequentially")

        results = []
        for i, conversation in enumerate(self.conversations):
            start_time = time.time()

            logger.debug(
                f"Processing conversation {i+1}/{total_conversations}: ID {conversation.id}"
            )

            # Ensure there's a Grade object
            if conversation.grade is None:
                conversation.grade = Grade()

            # Skip conversations with no turns
            if not conversation.conversation_turns:
                logger.warning(f"Skipping conversation {conversation.id} - no turns")
                results.append(
                    {
                        "conversation_id": conversation.id,
                        "empathy": -1,
                        "accuracy": -1,
                        "response_time": -1,
                    }
                )

                # Update progress if callback provided
                if progress_callback:
                    progress_callback(i + 1, total_conversations)
                continue

            try:
                evaluated_conv = self.evaluate_response(conversation)
                response_time = time.time() - start_time

                # Update response time
                if evaluated_conv.grade:
                    evaluated_conv.grade.response_time = response_time
                else:
                    evaluated_conv.grade = Grade(
                        empathy=-1, accuracy=-1, response_time=response_time
                    )

                results.append(
                    {
                        "conversation_id": evaluated_conv.id,
                        "empathy": (
                            evaluated_conv.grade.empathy
                            if hasattr(evaluated_conv.grade, "empathy")
                            else -1
                        ),
                        "accuracy": (
                            evaluated_conv.grade.accuracy
                            if hasattr(evaluated_conv.grade, "accuracy")
                            else -1
                        ),
                        "response_time": (
                            evaluated_conv.grade.response_time
                            if hasattr(evaluated_conv.grade, "response_time")
                            else -1
                        ),
                    }
                )

                logger.debug(
                    f"Processed conversation {conversation.id} in {response_time:.2f}s"
                )

            except Exception as e:
                logger.error(
                    f"Error processing conversation {conversation.id}: {str(e)}",
                    exc_info=True,
                )
                results.append(
                    {
                        "conversation_id": conversation.id,
                        "empathy": -1,
                        "accuracy": -1,
                        "response_time": -1,
                    }
                )

            # Update progress if callback provided
            if progress_callback:
                progress_callback(i + 1, total_conversations)
            # Log progress periodically (if no callback)
            elif (i + 1) % 10 == 0 or i == total_conversations - 1:
                logger.info(f"Processed {i + 1}/{total_conversations} conversations")

        # Save results to CSV with progress
        self._save_results_to_csv(results, output_path)

    def process_conversations_parallel(
        self,
        output_path: Path,
        max_workers: int = 3,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Process conversations in parallel with controlled concurrency

        Args:
            output_path: Path to save results
            max_workers: Maximum number of parallel workers
            progress_callback: Optional callback for progress reporting
        """
        total_conversations = len(self.conversations)
        logger.info(
            f"Processing {total_conversations} conversations with {max_workers} workers in parallel"
        )

        def process_conversation(conv):
            """Worker function to process a single conversation"""
            start_time = time.time()
            conv_id = conv.id

            logger.debug(f"Worker processing conversation {conv_id}")

            # Ensure there's a Grade object
            if conv.grade is None:
                conv.grade = Grade()

            if not conv.conversation_turns:
                logger.warning(f"Worker skipping conversation {conv_id} - no turns")
                return {
                    "conversation_id": conv_id,
                    "empathy": -1,
                    "accuracy": -1,
                    "response_time": -1,
                }

            try:
                evaluated_conv = self.evaluate_response(conv)
                response_time = time.time() - start_time

                # Update response time
                if evaluated_conv.grade:
                    evaluated_conv.grade.response_time = response_time
                else:
                    evaluated_conv.grade = Grade(
                        empathy=-1, accuracy=-1, response_time=response_time
                    )

                logger.debug(
                    f"Worker completed conversation {conv_id} in {response_time:.2f}s"
                )

                return {
                    "conversation_id": evaluated_conv.id,
                    "empathy": (
                        evaluated_conv.grade.empathy
                        if hasattr(evaluated_conv.grade, "empathy")
                        else -1
                    ),
                    "accuracy": (
                        evaluated_conv.grade.accuracy
                        if hasattr(evaluated_conv.grade, "accuracy")
                        else -1
                    ),
                    "response_time": (
                        evaluated_conv.grade.response_time
                        if hasattr(evaluated_conv.grade, "response_time")
                        else -1
                    ),
                }
            except Exception as e:
                logger.error(
                    f"Worker error processing conversation {conv_id}: {str(e)}",
                    exc_info=True,
                )
                return {
                    "conversation_id": conv_id,
                    "empathy": -1,
                    "accuracy": -1,
                    "response_time": -1,
                }

        completed = 0
        all_results = []

        logger.debug(f"Starting ThreadPoolExecutor with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_conv = {
                executor.submit(process_conversation, conv): conv
                for conv in self.conversations
            }

            logger.debug(f"Submitted {len(future_to_conv)} tasks to worker pool")

            # Process results as they complete
            for future in as_completed(future_to_conv):
                try:
                    result = future.result()
                    all_results.append(result)

                    # Update progress counter
                    completed += 1

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(completed, total_conversations)
                    # Or log progress periodically
                    elif completed % 10 == 0 or completed == total_conversations:
                        logger.info(
                            f"Processed {completed}/{total_conversations} conversations"
                        )
                except Exception as e:
                    logger.error(
                        f"Error getting future result: {str(e)}", exc_info=True
                    )
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_conversations)

        # Save results to CSV
        self._save_results_to_csv(all_results, output_path)

    def _save_results_to_csv(self, results: List[dict], output_path: Path) -> None:
        """
        Save evaluation results to CSV file

        Args:
            results: List of evaluation result dictionaries
            output_path: Path to save the CSV file
        """
        logger.info(f"Saving {len(results)} evaluation results to {output_path}")

        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)

            # Log some statistics about the results
            valid_empathy = df[df["empathy"] >= 0]["empathy"]
            valid_accuracy = df[df["accuracy"] >= 0]["accuracy"]

            if not valid_empathy.empty:
                logger.info(f"Average empathy score: {valid_empathy.mean():.2f}")
            if not valid_accuracy.empty:
                logger.info(f"Average accuracy score: {valid_accuracy.mean():.2f}")

            logger.info(f"Results successfully saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}", exc_info=True)
            raise
