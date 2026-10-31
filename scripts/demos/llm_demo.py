"""Send a prompt to the configured Groq language model."""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.llm_client import (  # noqa: E402
    GroqChatClient,
    GroqLLMConfig,
    LLMError,
    build_llm_client,
)


def main() -> int:
    """Run a one-off language-model request or list the available models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--model")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--system")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.list_models and args.prompt is None:
        parser.error("prompt is required unless --list-models is used.")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive.")
    if args.stream and args.json:
        parser.error("--json cannot be used with --stream.")

    try:
        config = GroqLLMConfig.from_env()
        if args.model is not None:
            config = replace(config, model=args.model)
        client = build_llm_client(config)
        if args.list_models:
            if not isinstance(client, GroqChatClient):
                raise LLMError("The configured LLM client cannot list Groq models.")
            for model_id in client.list_models():
                print(model_id)
            return 0

        messages = []
        if args.system is not None:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})

        if args.stream:
            started_at = time.perf_counter()
            first_token_seconds = None
            print("Response:")
            for delta in client.stream(
                messages, max_tokens=args.max_tokens
            ):
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - started_at
                print(delta, end="", flush=True)
            total_seconds = time.perf_counter() - started_at
            print()
            print(f"Model: {config.model}")
            print(f"Latency: {total_seconds:.3f} seconds")
            print("Prompt tokens: unavailable")
            print("Completion tokens: unavailable")
            print("Finish reason: unavailable")
            first_token_text = (
                f"{first_token_seconds:.3f} seconds"
                if first_token_seconds is not None
                else "no text received"
            )
            print(f"Time to first token: {first_token_text}")
            print(f"Total time: {total_seconds:.3f} seconds")
        else:
            result = client.complete(
                messages,
                max_tokens=args.max_tokens,
                json_mode=args.json,
            )
            print(result.text)
            print(f"Model: {result.model}")
            print(f"Latency: {result.latency_seconds:.3f} seconds")
            print(f"Prompt tokens: {result.prompt_tokens}")
            print(f"Completion tokens: {result.completion_tokens}")
            print(f"Finish reason: {result.finish_reason}")
        return 0
    except (LLMError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
