from __future__ import annotations

import os
import sys
import openai
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from templates import (
    SYSTEM_CREATIVE_QUALITY_PROMPT,
    USER_CREATIVE_QUALITY_PROMPT,
    SYSTEM_ARGUMENTATIVE_QUALITY_PROMPT,
    USER_ARGUMENTATIVE_QUALITY_PROMPT,
    SYSTEM_CREATIVE_DIVERSITY_PROMPT,
    USER_CREATIVE_DIVERSITY_PROMPT,
    SYSTEM_ARGUMENTATIVE_DIVERSITY_PROMPT,
    USER_ARGUMENTATIVE_DIVERSITY_PROMPT
)

class OpenAIEvaluator:

    def __init__(self, openai_api_key: str, model_name: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = model_name

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _evaluate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
                seed=42,
            )
            return chat_completion.choices[0].message.content
        except Exception as ex:
            print(ex)
            time.sleep(3)
        return 'error'

    def creative_quality_score(self, prompt: str, story: str) -> float:
        user_prompt = USER_CREATIVE_QUALITY_PROMPT.format(question=prompt, answer=story)
        content = self._evaluate(SYSTEM_CREATIVE_QUALITY_PROMPT, user_prompt)
        score_line = content.split('\n')[0]
        scores = [s for s in score_line.split() if s.strip() != '']
        try:
            return float(scores[0])
        except ValueError:
            return -1

    def argumentative_quality_score(self, prompt: str, essay: str) -> float:
        user_prompt = USER_ARGUMENTATIVE_QUALITY_PROMPT.format(question=prompt, answer=essay)
        content = self._evaluate(SYSTEM_ARGUMENTATIVE_QUALITY_PROMPT, user_prompt)
        score_line = content.split('\n')[0]
        scores = [s for s in score_line.split() if s.strip() != '']
        try:
            return float(scores[0])
        except ValueError:
            return -1

    def pairwise_creative_diversity(self, prompt: str, essay_1: str, essay_2: str) -> float:
        user_prompt = USER_CREATIVE_DIVERSITY_PROMPT.format(question=prompt, answer1=essay_1, answer2=essay_2)
        content = self._evaluate(SYSTEM_CREATIVE_DIVERSITY_PROMPT, user_prompt)
        score_line = content.split('\n')[0]
        scores = [s for s in score_line.split() if s.strip() != '']
        try:
            score1, score2 = float(scores[0]), float(scores[1])
            return score1, score2
        except ValueError:
            return -1, -1

    def pairwise_argumentative_diversity(self, prompt: str, essay_1: str, essay_2: str) -> float:
        user_prompt = USER_ARGUMENTATIVE_DIVERSITY_PROMPT.format(question=prompt, answer1=essay_1, answer2=essay_2)
        content = self._evaluate(SYSTEM_ARGUMENTATIVE_DIVERSITY_PROMPT, user_prompt)
        score_line = content.split('\n')[0]
        scores = [s for s in score_line.split() if s.strip() != '']
        try:
            score1, score2 = float(scores[0]), float(scores[1])
            return score1, score2
        except ValueError:
            return -1, -1