import argparse
import traceback
from matheval import set_client, AIMEEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Test LLM API connectivity using a math evaluator.")

    parser.add_argument(
        "--api_base",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="The base URL of the API (default: DashScope compatible OpenAI endpoint)."
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        default="qwen-max-2025-01-25",
        help="Model deployment name (default: qwen-max-2025-01-25)."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for authorization (required)."
    )

    return parser.parse_args()

def test_llm_connectivity(api_base, deployment_name, api_key):
    # Initialize the client with provided credentials
    set_client(
        api_base=api_base,
        deployment_name=deployment_name,
        api_version="",
        api_key=api_key
    )

    evaluator = AIMEEvaluator()

    # Simple test case for equivalence checking
    solution_str = (
        "This problem is quite difficult. However, after thorough thinking and careful calculation, "
        "I arrived at the answer: 1."
    )
    ground_truth = "1"

    try:
        result = evaluator.llm_judge(solution_str, ground_truth)

        if result in [True, False]:
            print("✅ API connectivity test succeeded. Result:", result)
        else:
            print("⚠️ Unexpected response format from API. Result:", result)

    except Exception as e:
        print("❌ API connection failed. Error details:")
        traceback.print_exc()

if __name__ == '__main__':
    args = parse_args()
    test_llm_connectivity(args.api_base, args.deployment_name, args.api_key)