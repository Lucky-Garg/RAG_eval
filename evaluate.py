"""
RAG evaluation with DeepEval (free LLM judge via Groq / Gemini / Ollama).

Retriever metrics:
  - Contextual Precision  : are the relevant chunks ranked higher than irrelevant ones?
  - Contextual Recall     : does the retrieved context contain everything needed (vs. expected answer)?
  - Contextual Relevancy  : how much of the retrieved context is actually relevant?
Generator metrics:
  - Faithfulness          : is the answer grounded in the context? (truthfulness / no hallucination)
  - Hallucination         : does the answer contradict the context?
  - Answer Relevancy      : does the answer address the question?
  - Correctness (G-Eval)  : is the answer factually correct vs. the ground truth?
"""
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from judge import JudgeLLM
from rag import RAGPipeline

# Golden dataset: question + ground-truth answer written by a human
GOLDEN = [
    {"input": "How many days of annual leave do employees get?",
     "expected_output": "Full-time employees get 24 days of paid annual leave per year."},
    {"input": "How many unused leave days can I carry forward?",
     "expected_output": "Up to 8 unused days can be carried forward; the rest expire on 31 March."},
    {"input": "Which days are mandatory in the office?",
     "expected_output": "Tuesday and Thursday are mandatory office days."},
    {"input": "Within how many days must I submit an expense claim?",
     "expected_output": "Expense claims must be submitted within 30 days of incurring the expense."},
    {"input": "Can I fly business class on a 4 hour flight?",
     "expected_output": "No. Business class is only allowed for flights longer than 6 hours and needs VP approval."},
    {"input": "What is the meal allowance for international travel?",
     "expected_output": "The daily meal allowance for international travel is 40 USD."},
    # Out-of-scope question: tests that the system refuses instead of hallucinating
    {"input": "What is the company's stock option vesting schedule?",
     "expected_output": "I don't have enough information to answer that."},
]


def build_test_cases(rag: RAGPipeline) -> list[LLMTestCase]:
    cases = []
    for g in GOLDEN:
        answer, contexts = rag.query(g["input"])
        cases.append(LLMTestCase(
            input=g["input"],
            actual_output=answer,
            expected_output=g["expected_output"],
            retrieval_context=contexts,   # used by the retriever + faithfulness metrics
            context=contexts,             # used by HallucinationMetric
        ))
    return cases


def main():
    rag = RAGPipeline()
    rag.ingest()
    test_cases = build_test_cases(rag)

    judge = JudgeLLM()   # LLM-as-a-judge (free provider from .env)
    metrics = [
        ContextualPrecisionMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False),
        ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False),
        ContextualRelevancyMetric(threshold=0.5, model=judge, include_reason=True, async_mode=False),
        FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False),
        HallucinationMetric(threshold=0.5, model=judge, include_reason=True, async_mode=False),  # lower score = better
        AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False),
        GEval(
            name="Correctness",
            criteria=("Determine whether the actual output is factually correct and consistent "
                      "with the expected output. Penalize missing key facts or wrong numbers."),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.7,
            model=judge,
            async_mode=False,
        ),
    ]

    try:
        from deepeval.evaluate.configs import AsyncConfig
        evaluate(test_cases=test_cases, metrics=metrics, async_config=AsyncConfig(run_async=False))
    except ImportError:
        evaluate(test_cases=test_cases, metrics=metrics)


if __name__ == "__main__":
    main()