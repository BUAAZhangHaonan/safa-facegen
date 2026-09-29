"""Entry points for training status and approved model delivery."""
import argparse
import json
from pathlib import Path
from .common import project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "controller"])
    parser.add_argument("--root", default=str(project_root()))
    parser.add_argument("--campaign", help="Explicit one-model campaign for controller")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "controller":
        if not args.campaign:
            parser.error("controller requires --campaign")
        from .controller import run_campaign
        campaign = Path(args.campaign)
        run_campaign(root, campaign if campaign.is_absolute() else root / campaign)
    else:
        status = root / "runs/controller/status.json"
        print(status.read_text() if status.exists() else json.dumps({"status": "not_started"}))


if __name__ == "__main__":
    main()
