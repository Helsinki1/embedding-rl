import argparse
import json
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Query embedding RL exercise")
    parser.add_argument("stage", choices=["prepare", "judge", "train", "final-test"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--root", type=Path, default=Path("work"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", choices=["baseline", "infonce", "rl"], default="rl")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.stage == "prepare":
        from .data import prepare
        result = prepare(args.root, config, args.device)
    elif args.stage == "judge":
        from .judge import judge
        result = judge(args.root, config)
    elif args.stage == "train":
        from .train import train
        result = train(args.root, config, args.method, args.device)
    else:
        if not args.run_id:
            parser.error("final-test requires --run-id")
        from .train import final_test
        result = final_test(args.root, args.run_id, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
