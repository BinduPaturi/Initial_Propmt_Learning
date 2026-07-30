import requests
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

GROQ_API_KEY = "#Your API Key Here"


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            }
        )
        return response.json()["choices"][0]["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


answerer = "llama-3.3-70b-versatile"
judge = GroqJudge(model_name="llama-3.1-8b-instant")


def get_llm_answer(prompt):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": answerer,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )
    return response.json()["choices"][0]["message"]["content"]


def test_hallucination():

    input_prompt = "Who is the CEO of Apple and where is Apple headquartered?"
    context = [
        "Elon Musk is the CEO of Apple.",
        "Apple was founded in 2001.",
        "Apple headquarters is in Austin, Texas."
    ]
    actual_output = get_llm_answer(input_prompt)
    print(f"\n LLaMA Answer:\n{actual_output}\n")

    test_case = LLMTestCase(
        input=input_prompt,
        actual_output=actual_output,
        context=context
    )
    metric = HallucinationMetric(
        threshold=0.1,
        model=judge,
        include_reason=True
    )
    evaluate(
        test_cases=[test_case],
        metrics=[metric]
    )
