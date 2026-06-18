import json
import os
import shlex
import sys
from datetime import datetime


def create_timestamped_run_dir(output_root):
    root = os.path.abspath(output_root)
    os.makedirs(root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(root, timestamp)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, timestamp


def output_path(run_dir, requested_path, default_name):
    name = os.path.basename(requested_path) if requested_path else default_name
    if not name:
        raise ValueError("Output filename cannot be empty.")
    return os.path.join(run_dir, name)


def save_run_metadata(run_dir, args):
    arguments_path = os.path.join(run_dir, "arguments.json")
    command_path = os.path.join(run_dir, "command.txt")
    with open(arguments_path, "w", encoding="utf-8") as arguments_file:
        json.dump(vars(args), arguments_file, indent=2, ensure_ascii=False)
    with open(command_path, "w", encoding="utf-8") as command_file:
        command_file.write(shlex.join(sys.argv) + "\n")
    return arguments_path, command_path
