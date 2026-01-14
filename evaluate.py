# Evaluation script for lmms-eval, adapted for DualTowerVLM.

import argparse
import datetime
import importlib
import json
import logging
import os
import sys
import traceback
import warnings
from functools import partial
from typing import Union

import numpy as np
import torch
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from lmms_eval import evaluator, utils
from lmms_eval.evaluator import request_caching_arg_to_dict
from lmms_eval.loggers import EvaluationTracker, WandbLogger
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks import get_task_dict
from lmms_eval.utils import make_table, simple_parse_args_string

from lmms_eval.wrapper import DualTowerWrapper


def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        logging.warning(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'. Missing values will be filled with defaults."
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])

    return items


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    if isinstance(o, set):
        return list(o)
    return str(o)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _safe_int(value, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying eval arguments.")
    parser.add_argument("--model", default="patrickamadeus/dualtower-cauldron", help="Model repo or checkpoint path.")
    parser.add_argument("--tasks", default=None, help="Tasks list. Use `--tasks list` to see available tasks.")
    parser.add_argument(
        "--model_args",
        default="",
        help="Model args string, e.g. `config_path=/path/to/config.json,load_backbone=false`.",
    )
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of fewshot examples.")
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default="32",
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer.",
    )
    parser.add_argument("--max_batch_size", type=int, default=None, metavar="N", help="Max batch size with --batch_size auto.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cuda:0, cpu).")
    parser.add_argument(
        "--output_path",
        default="results/",
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="Directory or file for evaluation results.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit number of examples per task. If <1, it's a percentage.",
    )
    parser.add_argument("--use_cache", "-c", type=str, default=None, metavar="DIR", help="Path to sqlite cache.")
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Cache requests for datasets.",
    )
    parser.add_argument("--check_integrity", action="store_true", help="Run task integrity checks.")
    parser.add_argument("--write_out", "-w", action="store_true", default=False, help="Print prompt for first docs.")
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="Write out model outputs and documents for post-hoc analysis.",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="Log sample outputs to Weights & Biases.",
    )
    parser.add_argument("--log_samples_suffix", type=str, default="model_outputs", help="Suffix for sample logs.")
    parser.add_argument("--system_instruction", type=str, default=None, help="System instruction for prompts.")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="Apply chat template to prompt (handled in wrapper by default).",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="Use fewshot as multi-turn conversation.",
    )
    parser.add_argument("--show_config", action="store_true", default=False, help="Show full task config.")
    parser.add_argument("--include_path", type=str, default=None, help="Additional path for external tasks.")
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help="Generation kwargs, e.g. `temperature=0,top_k=0,top_p=0`.",
    )
    parser.add_argument("--verbosity", type=str, default="INFO", help="Log verbosity.")
    parser.add_argument("--wandb_args", default="", help="Comma-separated args for wandb.init.")
    parser.add_argument("--timezone", default="Asia/Singapore", help="Timezone for datetime strings.")
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma-separated args passed to HF Hub logging.",
    )
    parser.add_argument("--predict_only", "-x", action="store_true", default=False, help="Only save outputs.")
    default_seed_string = "0"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,
        help=(
            "Seed for python, numpy, torch, and fewshot sampling (comma-separated). "
            "Example: `--seed 0,None,8,52`."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set trust_remote_code to True to load HF datasets.",
    )
    parser.add_argument("--no_log_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument(
        "--process_with_media",
        action="store_true",
        help="Process dataset with media (image, audio).",
    )
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to model checkpoint directory.")
    parser.add_argument("--global_step", type=int, default=0, help="Global step for checkpoint.")
    parser.add_argument("--run_name", type=str, default="", help="Training run name.")
    parser.add_argument("--checkpoints_dir", type=str, default="", help="Path to checkpoints directory.")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="Specific steps to evaluate.")
    parser.add_argument("--eval_tasks", type=str, nargs="+", default=None, help="Evaluation task list.")
    parser.add_argument("--eval_results_dir", default="eval_results", help="Eval results directory.")
    parser.add_argument("--force", action="store_true", help="Force re-run evaluations.")
    args = parser.parse_args()
    return args


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    default_args = parse_eval_args()

    if args is None and len(sys.argv) == 1:
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                          │")
        print("│ `python evaluation.py --model path/or/repo --tasks mmstar`                     │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    if args:
        for key, value in vars(args).items():
            setattr(default_args, key, value)

    args = default_args

    if args.wandb_args and not args.no_log_wandb:
        if "name" not in args.wandb_args:
            name = f"{args.model}_{args.model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    eval_logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args_list = []
    results_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        accelerator = None
        is_main_process = torch.distributed.get_rank() == 0
    else:
        kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=6000))
        accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
        is_main_process = accelerator.is_main_process

    for args in args_list:
        try:
            results, samples = cli_evaluate_single(args)
            results_list.append(results)

            if accelerator:
                accelerator.wait_for_everyone()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if is_main_process and args.wandb_args and not args.no_log_wandb:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples)
                except Exception as exc:
                    eval_logger.info(f"Logging to Weights and Biases failed: {exc}")

        except Exception as exc:
            if args.verbosity == "DEBUG":
                raise exc
            traceback.print_exc()
            eval_logger.error(f"Error during evaluation: {exc}. Use --verbosity=DEBUG for details.")
            results_list.append(None)

    for args, results in zip(args_list, results_list):
        if results is not None:
            print(
                f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), "
                f"limit: {args.limit}, num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
            )
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

    if args.wandb_args and not args.no_log_wandb:
        wandb_logger.run.finish()

    return results_list


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    selected_task_list = args.tasks.split(",") if args.tasks else None

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)

    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"

    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    eval_logger.info(f"Evaluation tracker args: {evaluation_tracker_args}")
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError("If fewshot_as_multiturn is set, apply_chat_template must be True.")

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        raise ValueError("If fewshot_as_multiturn is set, num_fewshot must be greater than 0.")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning("Pushing samples to Hub requires --log_samples. Samples will not be pushed.")

    if args.limit:
        eval_logger.warning("--limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT USE LIMIT.")

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        args.include_path = [args.include_path] if args.include_path else []
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
            args.include_path.append(package_tasks_location)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    if args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(sorted(task_manager.all_tasks))))
        sys.exit()
    if args.tasks == "list_groups":
        eval_logger.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    if args.tasks == "list_tags":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    if args.tasks == "list_subtasks":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    if args.tasks == "list_with_num":
        if get_task_dict is None:
            eval_logger.error("get_task_dict is unavailable in this lmms-eval version.")
            sys.exit(1)
        log_message = (
            "\n" + "=" * 70 + "\n" + "\n\tYou are trying to check all the numbers in each task."
            + "\n\tThis action will download the complete dataset."
            + "\n\tIf the results are not clear initially, call this again.\n" + "\n" + "=" * 70
        )
        eval_logger.info(log_message)
        for task_name in sorted(task_manager.list_all_tasks()):
            try:
                task_dict = get_task_dict([task_name], model_name="llava")
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    _, task_obj = task_obj
                    if task_obj is None:
                        continue
                eval_logger.info(
                    f"\nTask : {task_obj.config.task}\n - #num : "
                    f"{len(task_obj.test_docs()) if task_obj.has_test_docs() else len(task_obj.validation_docs())}"
                )
            except Exception as exc:
                eval_logger.debug(f"\nTask : {task_name} fail to load\n Exception:\n {exc}")
        sys.exit()

    if os.path.isdir(args.tasks):
        import glob

        task_names = []
        yaml_path = os.path.join(args.tasks, "*.yaml")
        for yaml_file in glob.glob(yaml_path):
            config = utils.load_yaml_config(yaml_file)
            task_names.append(config)
    else:
        task_list = args.tasks.split(",")
        task_names = task_manager.match_tasks(task_list)
        for task in [task for task in task_list if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [task for task in task_list if task not in task_names and "*" not in task]
        if task_missing:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}\nTry `lmms-eval --tasks list` for list of tasks"
            )
            raise ValueError(
                f"Tasks not found: {missing}. Try `lmms-eval --tasks list` to list tasks."
            )

    eval_logger.info(f"Selected Tasks: {task_names}")
    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests)
    datetime_str = utils.get_datetime_str(timezone=args.timezone)

    model_kwargs = simple_parse_args_string(args.model_args) if args.model_args else {}
    config_path = model_kwargs.pop("config_path", None)
    load_backbone = _parse_bool(model_kwargs.pop("load_backbone", False))
    max_length = model_kwargs.pop("max_length", None)
    if max_length is not None:
        max_length = int(max_length)
    wrapper_batch_size = _safe_int(args.batch_size, default=1)

    wrapped_model = DualTowerWrapper(
        model=args.model,
        device=args.device,
        batch_size=wrapper_batch_size,
        config_path=config_path,
        load_backbone=load_backbone,
        max_length=max_length,
        **model_kwargs,
    )

    results = evaluator.simple_evaluate(
        model=wrapped_model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        cli_args=args,
        datetime_str=datetime_str,
        distributed_executor_backend=(
            "torchrun" if (torch.distributed.is_available() and torch.distributed.is_initialized()) else "accelerate"
        ),
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if args.log_samples else None, datetime_str=datetime_str
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        return results, samples
    return None, None


def print_results(args, results):
    print(
        f"{args.model} ({args.model_args}),\n"
        f"gen_kwargs: ({args.gen_kwargs}),\n"
        f"limit: {args.limit},\n"
        f"num_fewshot: {args.num_fewshot},\n"
        f"batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()
