"""Editable terminal form and validation for Rosie training submissions."""

import argparse
import os
import re
import sys

from simulator.training.compiler import LEARNER_DEFAULTS, _learner
from simulator.training.plans import load_plan


LEARNER_FIELDS = {"HIDDEN": "hidden_dim", "EPOCHS": "epochs", "MINIBATCH": "minibatch_size", "LR": "lr",
                  **{key.upper(): key for key in LEARNER_DEFAULTS if key not in ("hidden_dim", "epochs", "minibatch_size", "lr")}}
LABELS = {"JOB_NAME": "Job/run name", "TIME": "Wall time", "ENVS": "Parallel worlds",
          "UPDATES": "Maximum PPO updates", "ROLLOUT": "Steps per rollout", "HIDDEN": "Hidden width",
          "EPOCHS": "PPO epochs", "MINIBATCH": "Minibatch size", "LR": "Learning rate",
          "GAMMA": "Discount gamma", "GAE_LAMBDA": "GAE lambda", "CLIP_EPS": "PPO clipping epsilon",
          "VF_COEF": "Value loss coefficient", "ENT_COEF": "Entropy coefficient", "MAX_GRAD_NORM": "Maximum gradient norm",
          "CPUS": "Allocated CPU cores", "MEMORY": "Host RAM", "THREADS": "Trainer threads (auto allowed)",
          "REPLAY": "Replay interval (0 disables)", "PROFILE_EVERY": "Verbose timing profile every N updates (0 off)",
          "SAVE_EVERY": "Full checkpoint interval (updates)", "RESUME": "Resume full checkpoint (optional)",
          "INITIALIZE": "Initialize from checkpoint (optional)", "START_STAGE": "Start at stage (blank = first)",
          "DEPENDENCY": "Wait for job IDs (or none)"}
ORDER = tuple(LABELS)
RESOURCE_FIELDS = ("TIME", "CPUS", "MEMORY")
LOCAL_ORDER = tuple(key for key in ORDER if key not in (*RESOURCE_FIELDS, "DEPENDENCY"))
OPTIONAL_TEXT_FIELDS = ("RESUME", "INITIALIZE", "START_STAGE")


def parse_values(values):
    settings, sweep = {}, {}
    for key, text in values.items():
        if key in OPTIONAL_TEXT_FIELDS:
            value = text.strip()
            if "," in value:
                raise ValueError(f"{LABELS[key]}: commas are not supported")
            settings[key] = value
            continue
        choices = [part.strip() for part in text.split(",")]
        if not all(choices):
            raise ValueError(f"{LABELS[key]}: a value is required")
        if key in ("PROFILE_EVERY", "SAVE_EVERY") and len(choices) > 1:
            raise ValueError(f"{LABELS[key]}: this setting cannot be swept")
        if len(choices) > 1 and (key in ("JOB_NAME", "DEPENDENCY") or sweep):
            raise ValueError("Only one training/resource field may contain a comma-separated sweep")
        seen = set()
        for value in choices:
            if key in LEARNER_FIELDS:
                name = LEARNER_FIELDS[key]
                try:
                    number = int(value) if name in ("hidden_dim", "epochs", "minibatch_size") else float(value)
                    _learner({name: number})
                except ValueError as error:
                    raise ValueError(f"{LABELS[key]}: {error}") from error
                if key == "HIDDEN" and number > 4096:
                    raise ValueError("Hidden width must be <=4096")
                identity = number
            elif key in ("ENVS", "UPDATES", "ROLLOUT", "CPUS", "THREADS", "REPLAY",
                         "PROFILE_EVERY", "SAVE_EVERY"):
                if key == "THREADS" and value == "auto" and len(choices) == 1:
                    identity = value
                else:
                    zero_allowed = key in ("REPLAY", "PROFILE_EVERY")
                    if (not re.fullmatch(r"0|[1-9][0-9]*", value)
                            or (int(value) == 0 and not zero_allowed)):
                        qualifier = " or zero" if zero_allowed else ""
                        raise ValueError(f"{LABELS[key]}: expected a positive integer{qualifier}")
                    identity = int(value)
                    if key == "ENVS" and identity > 65536:
                        raise ValueError("Parallel worlds must be <=65536")
            else:
                pattern = {"JOB_NAME": r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}",
                           "TIME": r"([0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]",
                           "MEMORY": r"[1-9][0-9]*[KMGT]",
                           "DEPENDENCY": r"none|[1-9][0-9]*(:[1-9][0-9]*)*"}[key]
                if not re.fullmatch(pattern, value):
                    raise ValueError(f"{LABELS[key]}: invalid value")
                identity = value
            if identity in seen:
                raise ValueError(f"{LABELS[key]}: sweep values must be distinct")
            seen.add(identity)
        settings[key] = choices[0]
        if len(choices) > 1:
            sweep = {"variable": key, "label": LABELS[key], "values": choices}
    return settings, sweep


def normalize(values, shared_gpu=False, max_parallel=0, local=False):
    settings, sweep = parse_values(values)
    if settings["RESUME"] and settings["INITIALIZE"]:
        raise ValueError("Choose resume or initialize from a checkpoint, not both")
    if settings["RESUME"] and settings["START_STAGE"]:
        raise ValueError("Resume restores the checkpoint's stage; clear the start-stage field")
    if shared_gpu and (settings["RESUME"] or settings["INITIALIZE"] or settings["START_STAGE"]
                       or int(settings["SAVE_EVERY"]) != 10):
        raise ValueError("Checkpoint controls, stage selection, and custom save cadence are not supported in shared-GPU mode")
    if local:
        if shared_gpu:
            raise ValueError("Local runs do not use shared-GPU Slurm mode")
        cores = os.cpu_count() or 1
        if settings["THREADS"] == "auto":
            settings["THREADS"] = str(min(4, cores))
        threads = list(map(int, sweep["values"])) if sweep.get("variable") == "THREADS" else [int(settings["THREADS"])]
        if max(threads) > cores:
            raise ValueError("Trainer threads exceed local CPU cores")
        return settings, sweep
    if shared_gpu and (not sweep or sweep["variable"] in RESOURCE_FIELDS):
        raise ValueError("Shared-GPU mode requires a sweep in one training setting, not allocation resources")
    count = len(sweep["values"]) if sweep else 1
    active = min(count, max_parallel or count) if shared_gpu else 1
    cores = min(map(int, sweep["values"])) if sweep.get("variable") == "CPUS" else int(settings["CPUS"])
    if settings["THREADS"] == "auto":
        if cores < active:
            raise ValueError("Allocate at least one CPU core per simultaneous trial, or lower max-parallel")
        settings["THREADS"] = str(min(4, cores // active))
    threads = list(map(int, sweep["values"])) if sweep.get("variable") == "THREADS" else [int(settings["THREADS"])] * active
    if sum(sorted(threads, reverse=True)[:active]) > cores:
        raise ValueError("Trainer threads exceed allocated CPU cores; reduce threads or trial concurrency")
    return settings, sweep


class SubmissionForm:
    def __init__(self, values, shared_gpu=False, max_parallel=0, input=None, output=None, local=False):
        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, VSplit
        from prompt_toolkit.layout.scrollable_pane import ScrollablePane
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Button, Label, TextArea

        self.local = local
        field_order = LOCAL_ORDER if local else ORDER
        self.fields = {key: TextArea(text=values[key], multiline=False, height=1,
                        wrap_lines=False, style="class:value") for key in field_order}
        self.error = Label("", style="class:error")

        def submit():
            try:
                result = normalize({key: field.text for key, field in self.fields.items()}, shared_gpu,
                                   max_parallel, local=self.local)
            except ValueError as error:
                self.error.text = str(error)
            else:
                self.application.exit(result=result)

        self.submit = Button("Submit", handler=submit)
        rows = [VSplit([Label(LABELS[key], width=32), field], padding=1) for key, field in self.fields.items()]
        bindings = KeyBindings()

        @bindings.add("up", eager=True)
        @bindings.add("s-tab", eager=True)
        def previous(event):
            event.app.layout.focus_previous()

        @bindings.add("down", eager=True)
        @bindings.add("tab", eager=True)
        def following(event):
            event.app.layout.focus_next()

        @bindings.add("enter", eager=True)
        def enter(event):
            if event.app.layout.has_focus(self.submit):
                submit()
            else:
                event.app.layout.focus_next()

        @bindings.add("c-u", eager=True)
        def clear(event):
            event.app.current_buffer.text = ""

        @bindings.add("c-c", eager=True)
        @bindings.add("c-d", eager=True)
        def cancel(event):
            event.app.exit(exception=KeyboardInterrupt())

        title = "Local Training Setup" if local else "Rosie Training Submission"
        layout = HSplit([Label(title),
                 Label("Up/Down: select   Type: edit   Ctrl+U: clear   Ctrl+C: cancel"),
                 Label("Sweep: comma-separated values in one field, e.g. 0,0.01,0.02"),
                         ScrollablePane(HSplit(rows)), self.error, self.submit])
        self.application = Application(layout=Layout(layout, focused_element=self.fields["JOB_NAME"]),
            key_bindings=bindings, full_screen=True, input=input, output=output,
            style=Style.from_dict({"value": "bg:#202020 #ffffff", "value focused": "bg:#005f5f #ffffff",
                                   "button.focused": "bg:#00875f #ffffff", "error": "#ff5f5f"}))

    def run(self):
        return self.application.run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--envs-default", default="8192")
    parser.add_argument("--rollout-default", default="128")
    parser.add_argument("--time-default", default="1-00:00:00")
    parser.add_argument("--shared-gpu", action="store_true")
    parser.add_argument("--local", action="store_true", help="Use the local host and omit Slurm resources")
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--profile-every-default", default="0")
    parser.add_argument("--save-every-default", default="10")
    parser.add_argument("--resume-default", default="")
    parser.add_argument("--initialize-default", default="")
    parser.add_argument("--start-stage-default", default="")
    parser.add_argument("--override", nargs=2, action="append", default=[])
    parser.add_argument("--learner-only", action="store_true")
    options = parser.parse_args(argv)
    if options.local and options.shared_gpu:
        parser.error("Local training does not use shared-GPU Slurm mode")
    plan = load_plan(options.plan)
    budgets = [stage["advance"].get("updates") for stage in plan["stages"]]
    updates = sum(budgets) if budgets and all(type(value) is int for value in budgets) else 1500
    values = {"JOB_NAME": options.plan.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".json"),
              "TIME": options.time_default, "ENVS": options.envs_default, "UPDATES": str(updates),
              "ROLLOUT": options.rollout_default, "CPUS": "8", "MEMORY": "24G", "THREADS": "auto",
              "REPLAY": "50000", "PROFILE_EVERY": options.profile_every_default,
              "SAVE_EVERY": options.save_every_default, "RESUME": options.resume_default,
              "INITIALIZE": options.initialize_default, "START_STAGE": options.start_stage_default,
              "DEPENDENCY": "none",
              **{key: str(plan["learner"][name]) for key, name in LEARNER_FIELDS.items()}}
    for key, value in options.override:
        if key not in LEARNER_FIELDS:
            parser.error("Unsupported learner override")
        values[key] = value
    try:
        if options.learner_only:
            result = parse_values({key: values[key] for key in LEARNER_FIELDS if key not in ("HIDDEN", "EPOCHS", "MINIBATCH", "LR")})
        else:
            from prompt_toolkit.output.defaults import create_output
            result = SubmissionForm(values, options.shared_gpu, options.max_parallel,
                                    output=create_output(stdout=sys.stderr), local=options.local).run()
    except (KeyboardInterrupt, EOFError):
        parser.exit(1, "Cancelled; nothing submitted.\n")
    except ImportError:
        parser.exit(1, "Install the training requirements (prompt-toolkit), or use --line-mode.\n")
    except ValueError as error:
        parser.exit(1, str(error) + "\nNothing submitted.\n")
    settings, sweep = result
    for key, value in settings.items():
        print(key + "\t" + value)
    if sweep:
        print("SWEEP_VARIABLE\t" + sweep["variable"])
        print("SWEEP_LABEL\t" + sweep["label"])
        for value in sweep["values"]:
            print("SWEEP_VALUE\t" + value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())