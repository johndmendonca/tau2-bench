#!/usr/bin/env python3
"""Run a text simulation from argparse and export results artifacts.

This script builds a `TextRunConfig` from command-line arguments, runs
`tau2.runner.run_domain()`, and writes:

- the standard `results.json` produced by the runner
- a `metrics.json` file with computed agent metrics
- a `conversations.json` file with the serialized message history for each run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tau2 import TextRunConfig
from tau2.metrics.agent_metrics import compute_metrics
from tau2.runner import make_run_name, run_domain
from tau2.utils import DATA_DIR


def _json_arg(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected valid JSON, got: {value!r}"
        ) from exc


def _add_text_run_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain", default=None, help="Domain to run.")
    parser.add_argument(
        "--task-set-name",
        default=None,
        help="Task set to load instead of the domain default.",
    )
    parser.add_argument(
        "--task-split-name",
        default=None,
        help="Task split to run (defaults to base in the model).",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Specific task IDs to run.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Limit the number of tasks to run.",
    )

    parser.add_argument(
        "--agent",
        default=None,
        help="Agent implementation name.",
    )
    parser.add_argument(
        "--llm-agent",
        default=None,
        help="LLM model for the agent.",
    )
    parser.add_argument(
        "--llm-args-agent",
        type=_json_arg,
        default=None,
        help="JSON object with LLM arguments for the agent.",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="User simulator implementation name.",
    )
    parser.add_argument(
        "--llm-user",
        default=None,
        help="LLM model for the user simulator.",
    )
    parser.add_argument(
        "--llm-args-user",
        type=_json_arg,
        default=None,
        help="JSON object with LLM arguments for the user simulator.",
    )

    parser.add_argument("--num-trials", type=int, default=None, help="Trial count.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of conversation turns.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=None,
        help="Maximum number of consecutive tool errors.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Maximum wall-clock time in seconds for a single simulation.",
    )
    parser.add_argument(
        "--save-to",
        default=None,
        help="Run name used under data/simulations/.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent simulations.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level for the simulation run.",
    )

    parser.add_argument(
        "--verbose-logs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum retries for failed tasks.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=None,
        help="Delay in seconds between retries.",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from existing save files automatically.",
    )
    parser.add_argument(
        "--auto-review",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run LLM review after each simulation.",
    )
    parser.add_argument(
        "--review-mode",
        choices=("full", "user"),
        default=None,
        help="Review mode used when auto-review is enabled.",
    )
    parser.add_argument(
        "--hallucination-retries",
        type=int,
        default=None,
        help="Maximum retries when a user hallucination is detected.",
    )
    parser.add_argument(
        "--is-remote",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mark the run as remote.",
    )

    parser.add_argument(
        "--retrieval-config",
        default=None,
        help="Knowledge retrieval configuration name.",
    )
    parser.add_argument(
        "--retrieval-config-kwargs",
        type=_json_arg,
        default=None,
        help="JSON object with keyword args for the retrieval config.",
    )

    parser.add_argument(
        "--enforce-communication-protocol",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enforce communication protocol rules for text mode.",
    )
    parser.add_argument(
        "--text-streaming-config",
        type=_json_arg,
        default=None,
        help="JSON object with text streaming configuration.",
    )

    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Optional path for the exported metrics JSON file.",
    )
    parser.add_argument(
        "--conversations-path",
        default=None,
        help="Optional path for the exported conversations JSON file.",
    )


def _build_config(args: argparse.Namespace) -> TextRunConfig:
    kwargs: dict[str, Any] = {}
    for key in (
        "domain",
        "task_set_name",
        "task_split_name",
        "task_ids",
        "num_tasks",
        "agent",
        "llm_agent",
        "llm_args_agent",
        "user",
        "llm_user",
        "llm_args_user",
        "num_trials",
        "max_steps",
        "max_errors",
        "timeout",
        "save_to",
        "max_concurrency",
        "seed",
        "log_level",
        "verbose_logs",
        "max_retries",
        "retry_delay",
        "auto_resume",
        "auto_review",
        "review_mode",
        "hallucination_retries",
        "is_remote",
        "retrieval_config",
        "retrieval_config_kwargs",
        "enforce_communication_protocol",
        "text_streaming_config",
    ):
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value
    return TextRunConfig(**kwargs)


def _resolve_run_dir(config: TextRunConfig) -> Path:
    run_name = config.save_to or make_run_name(config)
    return Path(DATA_DIR) / "simulations" / run_name


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_handle:
        json.dump(payload, file_handle, indent=2)


def _serialize_conversations(results) -> list[dict[str, Any]]:
    conversations: list[dict[str, Any]] = []
    for simulation in results.simulations:
        conversation = simulation.model_dump(mode="json", exclude={"messages", "ticks"})
        conversation["messages"] = [
            message.model_dump(mode="json") for message in simulation.get_messages()
        ]
        conversations.append(
            conversation
        )
    return conversations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a text simulation from TextRunConfig arguments."
    )
    _add_text_run_config_arguments(parser)
    args = parser.parse_args(argv)

    config = _build_config(args)
    results = run_domain(config)
    metrics = compute_metrics(results)

    run_dir = _resolve_run_dir(config)
    metrics_path = Path(args.metrics_path) if args.metrics_path else run_dir / "metrics.json"
    conversations_path = (
        Path(args.conversations_path)
        if args.conversations_path
        else run_dir / "conversations.json"
    )

    _write_json(
        metrics_path,
        {
            "run_directory": str(run_dir),
            "config": config.model_dump(mode="json"),
            "metrics": metrics.model_dump(mode="json"),
        },
    )
    _write_json(
        conversations_path,
        {
            "run_directory": str(run_dir),
            "config": config.model_dump(mode="json"),
            "conversations": _serialize_conversations(results),
        },
    )

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved conversations to {conversations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())