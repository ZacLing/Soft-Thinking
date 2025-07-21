import sglang as sgl
import json
import time
import argparse
import os
import csv
from transformers import AutoTokenizer
import torch

# ============================
#        Main Entry
# ============================
def main():
    parser = argparse.ArgumentParser(description='Run AIME2024 inference and save outputs.')
    parser.add_argument('--dataset', type=str, default="aime2024", help='Only aime2024 is supported')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save CSV output')
    parser.add_argument('--num_samples', type=int, default=1, help='Sampling number')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of samples')
    parser.add_argument('--end_idx', type=int, default=500, help='End index of samples')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=30, help='Top-k sampling')
    parser.add_argument('--min_p', type=float, default=0.0, help='Min-p sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')
    parser.add_argument('--early_stopping_entropy_threshold', type=float, default=0.0)
    parser.add_argument('--early_stopping_length_threshold', type=int, default=200)
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0e20)
    parser.add_argument('--max_generated_tokens', type=int, default=1024)
    parser.add_argument('--mem_fraction_static', type=float, default=0.5)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--max_batch', type=int, default=1000000)
    parser.add_argument('--sampling_backend', type=str, choices=["pytorch", "flashinfer"], default="flashinfer")
    parser.add_argument('--enable_soft_thinking', action='store_true')
    parser.add_argument('--think_end_str', type=str, default="</think>")
    parser.add_argument('--max_topk', type=int, default=30)
    parser.add_argument('--cuda_graph_max_bs', type=int, default=None)
    parser.add_argument('--max_running_requests', type=int, default=None)
    parser.add_argument('--after_thinking_temperature', type=float, default=0.6)
    parser.add_argument('--after_thinking_top_p', type=float, default=0.95)
    parser.add_argument('--after_thinking_top_k', type=int, default=30)
    parser.add_argument('--after_thinking_min_p', type=float, default=0.0)

    args = parser.parse_args()
    assert args.dataset == "aime2024", "This script only supports aime2024 dataset."

    with open("./datasets/aime2024.json") as f:
        samples = json.load(f)

    MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \boxed{{}}.

{Question}
""".strip()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_generated_tokens,
        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
        "early_stopping_length_threshold": args.early_stopping_length_threshold,
        "dirichlet_alpha": args.dirichlet_alpha,
        "n": 1,
        "think_end_str": args.think_end_str,
        "after_thinking_temperature": args.after_thinking_temperature,
        "after_thinking_top_p": args.after_thinking_top_p,
        "after_thinking_top_k": args.after_thinking_top_k,
        "after_thinking_min_p": args.after_thinking_min_p
    }

    prompt_list = []
    idx_list = []

    for idx in range(args.start_idx, min(args.end_idx, len(samples))):
        sample = samples[idx]
        chat = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
        prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        prompt_list.extend([prompt] * args.num_samples)
        idx_list.append(idx)

    decoded_text_list = []
    finish_generation_list = []

    idx = 0
    while idx < len(prompt_list):
        batch_prompts = prompt_list[idx:idx + args.max_batch]
        print(f"Generating batch: {idx} to {idx + len(batch_prompts)}")

        llm = sgl.Engine(
            model_path=args.model_name,
            tp_size=args.num_gpus,
            log_level="info",
            trust_remote_code=True,
            random_seed=0,
            max_running_requests=args.max_running_requests,
            mem_fraction_static=args.mem_fraction_static,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            enable_soft_thinking=args.enable_soft_thinking,
            max_topk=args.max_topk,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            sampling_backend=args.sampling_backend
        )

        outputs = llm.generate(batch_prompts, sampling_params)
        decoded_text_list.extend([o["text"] for o in outputs])
        finish_generation_list.extend([o["meta_info"]["finish_reason"] != "length" for o in outputs])

        llm.shutdown()
        torch.cuda.empty_cache()
        idx += args.max_batch

    output_csv = os.path.join(args.output_dir, "out_ans.csv")
    with open(output_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["decoded_text", "final_answer", "extracted_answer", "finish_generation"])
        for i, idx in enumerate(idx_list):
            sample = samples[idx]
            final_answer = sample["final_answer"]
            for j in range(args.num_samples):
                answer_text = decoded_text_list[i * args.num_samples + j]
                finish_flag = finish_generation_list[i * args.num_samples + j]
                writer.writerow([answer_text, final_answer, "", finish_flag])

    print(f"✅ Inference complete. Output saved to {output_csv}")

if __name__ == "__main__":
    main()