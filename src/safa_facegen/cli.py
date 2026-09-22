"""Entry points for training status and approved model delivery."""
import argparse
import json
from pathlib import Path
from .common import project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "controller"])
    parser.add_argument("--root", default=str(project_root()))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "controller":
        from .controller import run_campaign
        run_campaign(root, root / "configs/campaign.json")
    else:
        status = root / "runs/controller/status.json"
        print(status.read_text() if status.exists() else json.dumps({"status": "not_started"}))
