from typing import List
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
        """Evaluate a full conversation for empathy and accuracy"""

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
            response = self.llm.complete(system_prompt + "\n\n" + evaluation_prompt)
            result = json.loads(response.text)

            conv.grade.empathy = result.get("empathy", -1)
            conv.grade.accuracy = result.get("accuracy", -1)
            return conv
        except Exception as e:
            logger.error(f"Error evaluating conversation: {str(e)}")
            conv.grade = Grade(
                empathy=-1,
                accuracy=-1,
                response_time=conv.grade.response_time if conv.grade else -1,
            )
            return conv

    def process_conversations(self, output_path: Path, progress_callback=None) -> None:
        """Process all conversations"""
        total_conversations = len(self.conversations)
        logger.info(f"Processing {total_conversations} conversations")

        results = []
        for i, conversation in enumerate(self.conversations):
            start_time = time.time()

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
            except Exception as e:
                logger.error(
                    f"Error processing conversation {conversation.id}: {str(e)}"
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
        logger.info("Saving evaluation results to CSV...")
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    def process_conversations_parallel(
        self, output_path: Path, max_workers: int = 3, progress_callback=None
    ) -> None:
        """Process conversations in parallel with controlled concurrency"""
        total_conversations = len(self.conversations)
        logger.info(
            f"Processing {total_conversations} conversations with {max_workers} workers"
        )

        def process_conversation(conv):
            start_time = time.time()

            # Ensure there's a Grade object
            if conv.grade is None:
                conv.grade = Grade()

            if not conv.conversation_turns:
                logger.warning(f"Skipping conversation {conv.id} - no turns")
                return {
                    "conversation_id": conv.id,
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
                logger.error(f"Error processing conversation {conv.id}: {str(e)}")
                return {
                    "conversation_id": conv.id,
                    "empathy": -1,
                    "accuracy": -1,
                    "response_time": -1,
                }

        completed = 0
        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_conv = {
                executor.submit(process_conversation, conv): conv
                for conv in self.conversations
            }

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
                    logger.error(f"Error getting future result: {str(e)}")
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_conversations)

        # Save results to CSV
        logger.info("Saving evaluation results to CSV...")
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
