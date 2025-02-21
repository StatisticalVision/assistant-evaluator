import time
from typing import Optional, Dict
import logging
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class AssistantInterface:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.assistant-service.com/v1"
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def record_response(self, input_text: str) -> str:
        """Interface with the AI assistant and record its response"""
        try:
            start_time = time.time()

            response = self._make_api_call(input_text)

            # Record response time
            response_time = time.time() - start_time
            logger.info(f"Response received in {response_time:.2f} seconds")

            return response
        except RequestException as e:
            logger.error(f"Error getting response from assistant: {str(e)}")
            return ""

    def _make_api_call(self, input_text: str) -> str:
        """Make the actual API call to the assistant service"""
        try:
            response = self.session.post(
                f"{self.base_url}/chat", json={"input": input_text}
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except RequestException as e:
            logger.error(f"API call failed: {str(e)}")
            raise
