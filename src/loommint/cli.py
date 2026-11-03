from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import (
    AgentError,
    AgentProviderError,
    FineTuningAgent,
    OpenAICompatibleProvider,
    load_agent_environment,
    load_plan,
    save_plan,
)
from .datasets import DatasetError, SUPPORTED_FORMATS, inspect_dataset, prepare_dataset
from .trl_backend import TrainerDependencyError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loommint", description="Prepare data and plan open-model fine-tuning")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="inspect structure without displaying row contents")
    inspect.add_argument("input")
    inspect.add_argument("--format", choices=("auto", *SUPPORTED_FORMATS), default="auto")
    prepare = commands.add_parser("prepare", help="normalize and split a supervised fine-tuning dataset")
    prepare.add_argument("input")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--format", choices=("auto", *SUPPORTED_FORMATS), default="auto")
    prepare.add_argument("--validation-ratio", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    agent = commands.add_parser("agent", help="plan or explicitly approve agent-proposed preparation/training")
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    plan = agent_commands.add_parser("plan", help="ask an LLM to propose a validated, metadata-only plan")
    plan.add_argument("input")
    plan.add_argument("--goal", required=True)
    plan.add_argument("--agent-model", default=os.getenv("LOOMMINT_AGENT_MODEL", "gpt-6-luna"))
    plan.add_argument("--base-url", default=os.getenv("LOOMMINT_AGENT_BASE_URL", "https://api.openai.com/v1"))
    plan.add_argument("--plan-file", required=True)
    apply = agent_commands.add_parser("apply", help="apply a saved plan after explicit approval")
    apply.add_argument("plan_file")
    apply.add_argument("--output-dir", required=True)
    apply.add_argument("--approve", action="store_true", help="approve dataset preparation")
    apply.add_argument("--train", action="store_true", help="also start the planned SFT training job")
    apply.add_argument("--approve-training", action="store_true", help="separately approve model training")
    apply.add_argument("--training-output-dir")
    apply.add_argument("--model-id", help="override the model proposed by the agent")
    return parser


def _run_agent(args: argparse.Namespace) -> dict:
    if args.agent_command == "plan":
        if not args.agent_model:
            raise AgentError("provide --agent-model or set LOOMMINT_AGENT_MODEL")
        provider = OpenAICompatibleProvider(
            args.base_url,
            args.agent_model,
            api_key=os.getenv("LOOMMINT_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )
        plan = FineTuningAgent(provider).plan(args.input, args.goal)
        save_plan(plan, args.plan_file)
        return plan.as_dict()
    plan = load_plan(args.plan_file)
    backend = None
    if args.train:
        if not args.training_output_dir:
            raise AgentError("--training-output-dir is required with --train")
        if plan.backend == "unsloth":
            from .trl_backend import UnslothSFTBackend

            backend = UnslothSFTBackend()
        else:
            from .trl_backend import TRLSFTBackend

            backend = TRLSFTBackend()
    elif args.approve_training or args.training_output_dir:
        raise AgentError("training options require --train")
    result = FineTuningAgent().execute(
        plan,
        args.output_dir,
        approve_preparation=args.approve,
        trainer_backend=backend,
        training_output_dir=args.training_output_dir,
        approve_training=args.approve_training,
        model_id_override=args.model_id,
    )
    return result.as_dict()


def main(argv: list[str] | None = None) -> int:
    try:
        load_agent_environment()
    except AgentError as exc:
        print(f"loommint: {exc}", file=sys.stderr)
        return 2
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_dataset(args.input, args.format).as_dict()
        elif args.command == "prepare":
            result = prepare_dataset(
                args.input,
                args.output_dir,
                selected_format=args.format,
                validation_ratio=args.validation_ratio,
                seed=args.seed,
            )
        else:
            result = _run_agent(args)
    except (AgentError, AgentProviderError, DatasetError, TrainerDependencyError, ValueError) as exc:
        print(f"loommint: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
