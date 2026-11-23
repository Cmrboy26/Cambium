"""Editable terminal form and validation for Rosie training submissions."""

import argparse
import os
from pathlib import Path
import re
import sys

from simulator.training.compiler import LEARNER_DEFAULTS, _learner
from simulator.training.plans import load_plan


LEARNER_FIELDS = {"HIDDEN": "hidden_dim", "EPOCHS": "epochs", "MINIBATCH": "minibatch_size", "LR": "lr",
                  **{key.upper(): key for key in LEARNER_DEFAULTS if key not in ("hidden_dim", "epochs", "minibatch_size", "lr")}}
LABELS = {"JOB_NAME": "Job/run name", "ROLLOUT_BACKEND": "Rollout backend", "WORKLOAD_PRESET": "Workload preset",
          "DURATION_PRESET": "Duration preset", "TIME": "Wall time", "ENVS": "Parallel worlds",
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


def matching_plans(plans, query):
    terms = query.casefold().split()
    return [plan for plan in plans if all(term in plan.casefold() for term in terms)]


class PlanPicker:
    def __init__(self, plans, input=None, output=None):
        from prompt_toolkit import Application
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.widgets import Label, TextArea

        self.plans = sorted(plans)
        self.matches = self.plans
        self.selected = next((index for index, plan in enumerate(self.plans)
                              if plan == "shield-axe-curriculum.json"), 0)
        self.search = TextArea(multiline=False, height=1, prompt="Search: ")

        def filter_results(_):
            self.matches = matching_plans(self.plans, self.search.text)
            self.selected = 0

        self.search.buffer.on_text_changed += filter_results

        def results():
            start = max(0, min(self.selected - 4, len(self.matches) - 8))
            lines = [("", f"{len(self.matches)} matching plans\n")]
            for index in range(start, min(start + 8, len(self.matches))):
                lines.append(("reverse" if index == self.selected else "", self.matches[index] + "\n"))
            return FormattedText(lines)

        bindings = KeyBindings()

        @bindings.add("up", eager=True)
        def previous(event):
            if self.matches:
                self.selected = (self.selected - 1) % len(self.matches)

        @bindings.add("down", eager=True)
        def following(event):
            if self.matches:
                self.selected = (self.selected + 1) % len(self.matches)

        @bindings.add("enter", eager=True)
        def select(event):
            if self.matches:
                event.app.exit(result=self.matches[self.selected])

        @bindings.add("c-c", eager=True)
        @bindings.add("c-d", eager=True)
        def cancel(event):
            event.app.exit(exception=KeyboardInterrupt())

        layout = HSplit([Label("Choose a training plan"),
                         Label("Type to filter paths; Up/Down to choose; Enter to select; Ctrl+C to cancel"),
                         self.search, Window(FormattedTextControl(results), height=9)])
        self.application = Application(layout=Layout(layout, focused_element=self.search),
                                       key_bindings=bindings, full_screen=True, input=input, output=output)

    def run(self):
        return self.application.run()


def replay_default(plan):
    return "0" if any(stage["scenario"].get("bridge") is not None for stage in plan["stages"]) else "50000"


def resident_reason(plan):
    from simulator.training.research import resident_unsupported
    return resident_unsupported(plan, "cuda")


def backend_status(backend, reason):
    if backend == "auto":
        return f"Auto will use host: {reason}" if reason else "Auto prefers CUDA resident but may fall back to host on the GPU node."
    if backend == "cuda-resident":
        return f"CUDA resident unavailable: {reason}" if reason else "CUDA resident is plan-compatible; runtime capabilities are checked on the GPU node."
    if backend == "host":
        return f"Host selected; CUDA resident unavailable: {reason}" if reason else "Host selected; CUDA resident is also plan-compatible."
    return "Choose host, cuda-resident, or auto."


def workload_presets(large):
    large = int(large)
    return {"small": max(1, large // 16), "medium": max(1, large // 4), "large": large}


def duration_presets(standard):
    standard = int(standard)
    return {"short": min(20, standard), "standard": standard, "extended": min(999999999, standard * 2)}


def stage_budgets(plan):
    stages = plan["stages"]
    result = {}
    for index in range(len(stages)):
        budgets = [stage["advance"].get("updates") for stage in stages[index:]]
        value = sum(budgets) if all(type(budget) is int and budget > 0 for budget in budgets) else 1500
        if index == 0:
            result[""] = value
        if stages[index].get("name"):
            result[stages[index]["name"]] = value
    return result


def resume_details(path):
    import torch
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Cannot read resume checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("research"), dict):
        raise ValueError("Resume requires a full research checkpoint")
    research = checkpoint["research"]
    if not all(key in research for key in ("world_count", "rollout_steps")):
        raise ValueError("Resume checkpoint is missing world count or rollout settings")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Resume checkpoint has invalid metadata")
    return research, metadata


def validate_checkpoints(settings, unsupported):
    for key in ("RESUME", "INITIALIZE"):
        if settings[key] and not Path(settings[key]).is_file():
            raise ValueError(f"{LABELS[key]} not found: {settings[key]}")
    if not settings["RESUME"]:
        return
    research, metadata = resume_details(settings["RESUME"])
    backend = settings["ROLLOUT_BACKEND"]
    effective = ("host" if unsupported else "cuda-resident") if backend == "auto" else backend
    saved_backend = research.get("rollout_backend", "host")
    if effective != saved_backend:
        raise ValueError(f"Resume checkpoint requires backend {saved_backend}; use initialize to change backend")
    for key, saved_value in (("ENVS", research["world_count"]), ("ROLLOUT", research["rollout_steps"]),
                             ("HIDDEN", metadata.get("hidden_dim"))):
        if saved_value is not None and settings[key] != str(saved_value):
            raise ValueError(f"Resume checkpoint requires {LABELS[key]} = {saved_value}")


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
            elif key in ("ROLLOUT_BACKEND", "WORKLOAD_PRESET", "DURATION_PRESET"):
                allowed = {"ROLLOUT_BACKEND": ("host", "cuda-resident", "auto"),
                           "WORKLOAD_PRESET": ("small", "medium", "large", "custom"),
                           "DURATION_PRESET": ("short", "standard", "extended", "custom")}[key]
                if len(choices) != 1 or value not in allowed:
                    raise ValueError(f"{LABELS[key]} must be one of: {', '.join(allowed)}")
                identity = value
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


def normalize(values, shared_gpu=False, max_parallel=0, local=False, bridge=False, unsupported=None):
    settings, sweep = parse_values(values)
    backend = settings["ROLLOUT_BACKEND"]
    if shared_gpu and backend != "host":
        raise ValueError("Shared-GPU sweeps require the host rollout backend")
    if backend == "cuda-resident" and unsupported:
        raise ValueError(f"CUDA resident unavailable: {unsupported}")
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
    def __init__(self, values, shared_gpu=False, max_parallel=0, input=None, output=None, local=False, bridge=False,
                 unsupported=None, workloads=None, durations=None, budgets=None):
        from prompt_toolkit import Application
        from prompt_toolkit.filters import Condition
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
        workloads = workloads or {}
        durations = durations or {}
        applying_preset = False

        def bind_preset(preset_key, value_key, presets):
            def apply(_):
                nonlocal applying_preset
                selected = self.fields[preset_key].text
                if selected in presets:
                    applying_preset = True
                    try:
                        self.fields[value_key].buffer.text = str(presets[selected])
                    finally:
                        applying_preset = False

            def custom(_):
                if not applying_preset:
                    self.fields[preset_key].buffer.text = "custom"

            self.fields[preset_key].buffer.on_text_changed += apply
            self.fields[value_key].buffer.on_text_changed += custom

        bind_preset("WORKLOAD_PRESET", "ENVS", workloads)
        bind_preset("DURATION_PRESET", "UPDATES", durations)

        def update_stage(_):
            nonlocal applying_preset
            stage = self.fields["START_STAGE"].text
            if budgets is None or stage not in budgets:
                return
            durations.clear()
            durations.update(duration_presets(budgets[stage]))
            selected = self.fields["DURATION_PRESET"].text
            if selected in durations:
                applying_preset = True
                try:
                    self.fields["UPDATES"].buffer.text = str(durations[selected])
                finally:
                    applying_preset = False

        self.fields["START_STAGE"].buffer.on_text_changed += update_stage

        def submit():
            try:
                result = normalize({key: field.text for key, field in self.fields.items()}, shared_gpu,
                                   max_parallel, local=self.local, bridge=bridge, unsupported=unsupported)
                if budgets is not None and result[0]["START_STAGE"] not in budgets:
                    raise ValueError("Unknown start stage; choose a stage in the selected plan")
                validate_checkpoints(result[0], unsupported)
            except ValueError as error:
                self.error.text = str(error)
            else:
                self.application.exit(result=result)

        self.submit = Button("Submit", handler=submit)
        rows = [VSplit([Label(LABELS[key], width=32), field], padding=1) for key, field in self.fields.items()]
        bindings = KeyBindings()
        choices = {"ROLLOUT_BACKEND": ("cuda-resident", "host", "auto"),
                   "WORKLOAD_PRESET": ("small", "medium", "large", "custom"),
                   "DURATION_PRESET": ("short", "standard", "extended", "custom")}

        def selected_choice():
            return next((key for key in choices if self.application.layout.has_focus(self.fields[key])), None)

        def cycle_choice(event, offset):
            key = selected_choice()
            if key is not None:
                options = choices[key]
                current = self.fields[key].text
                index = options.index(current) if current in options else 0
                self.fields[key].buffer.text = options[(index + offset) % len(options)]

        @bindings.add("left", filter=Condition(lambda: selected_choice() is not None), eager=True)
        def previous_choice(event):
            cycle_choice(event, -1)

        @bindings.add("right", filter=Condition(lambda: selected_choice() is not None), eager=True)
        def following_choice(event):
            cycle_choice(event, 1)

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
                 Label("Left/Right: change backend or presets; preset values remain editable."),
                 Label("Short duration may stop before curriculum stages are completed."),
                 Label("Sweep: comma-separated values in one field, e.g. 0,0.01,0.02"),
             Label(lambda: backend_status(self.fields["ROLLOUT_BACKEND"].text, unsupported)),
                         ScrollablePane(HSplit(rows)), self.error, self.submit])
        self.application = Application(layout=Layout(layout, focused_element=self.fields["JOB_NAME"]),
            key_bindings=bindings, full_screen=True, input=input, output=output,
            style=Style.from_dict({"value": "bg:#202020 #ffffff", "value focused": "bg:#005f5f #ffffff",
                                   "button.focused": "bg:#00875f #ffffff", "error": "#ff5f5f"}))

    def run(self):
        return self.application.run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--choose-plan", action="store_true")
    parser.add_argument("--backend-default", choices=("host", "cuda-resident", "auto"), default="host")
    parser.add_argument("--recommend-backend", action="store_true")
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
    if options.choose_plan:
        if options.plan:
            parser.error("--choose-plan cannot be combined with --plan")
        root = Path("simulator/setups/curricula")
        plans = [path.relative_to(root).as_posix() for path in root.rglob("*.json") if path.is_file()]
        if not plans:
            parser.exit(1, "No training plans found.\n")
        try:
            from prompt_toolkit.output.defaults import create_output
            selected = PlanPicker(plans, output=create_output(stdout=sys.stderr)).run()
        except (KeyboardInterrupt, EOFError):
            parser.exit(1, "Cancelled; nothing submitted.\n")
        except ImportError:
            parser.exit(1, "Install the training requirements (prompt-toolkit), or use --line-mode.\n")
        print((root / selected).as_posix())
        return 0
    if not options.plan:
        parser.error("--plan is required")
    plan = load_plan(options.plan)
    unsupported = None if options.learner_only else resident_reason(plan)
    budgets = stage_budgets(plan)
    updates = budgets.get(options.start_stage_default, budgets[""])
    backend = ("cuda-resident" if options.recommend_backend and unsupported is None and not options.shared_gpu
               else options.backend_default)
    workloads = workload_presets(options.envs_default)
    durations = duration_presets(updates)
    workload = "custom" if options.resume_default else "medium"
    values = {"JOB_NAME": options.plan.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".json"),
              "ROLLOUT_BACKEND": backend,
              "WORKLOAD_PRESET": workload, "DURATION_PRESET": "standard",
              "TIME": options.time_default, "ENVS": options.envs_default if workload == "custom" else str(workloads[workload]),
              "UPDATES": str(updates),
              "ROLLOUT": options.rollout_default, "CPUS": "8", "MEMORY": "24G", "THREADS": "auto",
              "REPLAY": replay_default(plan), "PROFILE_EVERY": options.profile_every_default,
              "SAVE_EVERY": options.save_every_default, "RESUME": options.resume_default,
              "INITIALIZE": options.initialize_default, "START_STAGE": options.start_stage_default,
              "DEPENDENCY": "none",
              **{key: str(plan["learner"][name]) for key, name in LEARNER_FIELDS.items()}}
    if options.resume_default and not options.learner_only:
        try:
            research, metadata = resume_details(options.resume_default)
            values["ENVS"] = str(research["world_count"])
            values["ROLLOUT"] = str(research["rollout_steps"])
            if metadata.get("hidden_dim") is not None:
                values["HIDDEN"] = str(metadata["hidden_dim"])
            if options.recommend_backend:
                values["ROLLOUT_BACKEND"] = research.get("rollout_backend", "host")
        except ValueError:
            pass
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
                                    output=create_output(stdout=sys.stderr), local=options.local,
                                    bridge=replay_default(plan) == "0", unsupported=unsupported,
                                    workloads=workloads, durations=durations, budgets=budgets).run()
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