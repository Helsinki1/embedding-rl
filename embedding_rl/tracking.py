import json
import os


class Tracker:
    """Local JSONL always; optional W&B online/offline run and ranking examples."""

    def __init__(self, directory, config, method, metadata):
        self.directory = directory
        self.run = None
        if config.wandb_mode != "disabled":
            entity = os.environ.get("WANDB_ENTITY", config.wandb_entity)
            if config.wandb_mode == "online" and (not entity or entity == "YOUR_WANDB_ENTITY"):
                raise ValueError("Replace YOUR_WANDB_ENTITY or use wandb_mode=offline")
            if config.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
                raise ValueError("Set WANDB_API_KEY in your shell or Modal Secret")
            import wandb
            self.run = wandb.init(
                project=config.wandb_project,
                entity=entity if entity and entity != "YOUR_WANDB_ENTITY" else None,
                name=directory.name, job_type=method, group=metadata["experiment_id"],
                config={**config.to_dict(), **metadata}, mode=config.wandb_mode,
                dir=str(directory),
            )

    def log(self, metrics, step):
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        if self.run:
            self.run.log(metrics, step=step)

    def examples(self, before, after):
        if self.run:
            import wandb
            rows = [[phase, *row.values()] for phase, result in (("before", before), ("after", after))
                    for row in result["examples"]]
            columns = ["phase", *before["examples"][0].keys()]
            self.run.log({"rankings": wandb.Table(columns=columns, data=rows)})

    def finish(self, summary=None, success=True):
        if self.run:
            if summary:
                self.run.summary.update(summary)
            self.run.finish(exit_code=0 if success else 1)
