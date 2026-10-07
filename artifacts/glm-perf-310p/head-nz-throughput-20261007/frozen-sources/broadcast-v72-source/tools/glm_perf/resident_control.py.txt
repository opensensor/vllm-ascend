"""Python patch transactions for the experimental resident GLM harness."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import regex as re

MODES = ("graph", "direct-target", "direct-draft", "direct-both")


@dataclass(frozen=True)
class Control:
    generation: str
    mode: str = "graph"
    candidate: str = "baseline"
    source: str = ""
    recapture: bool = False

    @classmethod
    def from_dict(cls, value: Any) -> Control:
        if not isinstance(value, dict):
            raise ValueError("control must be a JSON object")
        try:
            control = cls(**value)
        except TypeError as error:
            raise ValueError("invalid control fields") from error
        if not isinstance(control.generation, str) or not re.fullmatch(r"[a-f0-9]{32}", control.generation):
            raise ValueError("generation must be a UUID hex string")
        if not isinstance(control.mode, str) or control.mode not in MODES:
            raise ValueError("unknown execution mode")
        if not isinstance(control.candidate, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", control.candidate):
            raise ValueError("invalid candidate name")
        if not isinstance(control.source, str) or not isinstance(control.recapture, bool):
            raise ValueError("source must be text and recapture must be boolean")
        if (control.candidate == "baseline") != (control.source == ""):
            raise ValueError("baseline requires empty source; a candidate requires Python source")
        return control

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()


@dataclass(frozen=True)
class Replacement:
    target: str
    owner: Any
    attribute: str
    function: Any


def prepare_replacements(control: Control, native_resources=None) -> list[Replacement]:
    if not control.source:
        return []
    module = ModuleType(f"glm_candidate_{control.generation}")
    # Compile the supplied text directly so rapid edits never reuse stale pyc.
    exec(compile(control.source, f"<{control.candidate}>", "exec"), module.__dict__)
    factory = getattr(module, "replacements", None)
    if not callable(factory):
        raise ValueError("candidate must define replacements() returning a dict")
    replacements = (
        factory(native_resources=native_resources)
        if "native_resources" in inspect.signature(factory).parameters
        else factory()
    )
    if not isinstance(replacements, dict) or not replacements:
        raise ValueError("candidate replacements must be a nonempty dict")
    result = []
    for target, function in replacements.items():
        if not isinstance(target, str) or target.count(":") != 1:
            raise ValueError("replacement target must be module:attribute")
        module_name, attribute_path = target.split(":")
        if not module_name.startswith(("vllm.", "vllm_ascend.")):
            raise ValueError("replacement must target vLLM or vLLM Ascend")
        owner = importlib.import_module(module_name)
        attributes = attribute_path.split(".")
        if not all(part.isidentifier() for part in attributes):
            raise ValueError("invalid replacement attribute")
        for attribute in attributes[:-1]:
            owner = getattr(owner, attribute)
        attribute = attributes[-1]
        original = inspect.getattr_static(owner, attribute)
        if not inspect.isfunction(original) or not inspect.isfunction(function):
            raise ValueError("only Python functions and ordinary methods can be replaced")
        result.append(Replacement(target, owner, attribute, function))
    return sorted(result, key=lambda replacement: replacement.target)


class PatchSession:
    """Stage without mutation, then replace functions while workers are paused."""

    def __init__(self, native_resources=None):
        self.native_resources = native_resources
        self.current: Control | None = None
        self.pending: tuple[Control, list[Replacement]] | None = None
        self.originals: list[tuple[Any, str, bool, Any]] = []
        self.graphs_dirty = False

    def prepare(self, value: Any) -> dict[str, Any]:
        control = Control.from_dict(value)
        replacements = prepare_replacements(control, self.native_resources)
        self.pending = (control, replacements)
        return {"generation": control.generation, "digest": control.digest, "targets": [r.target for r in replacements]}

    def apply(self, generation: str) -> bool:
        if self.pending is None or self.pending[0].generation != generation:
            raise ValueError("generation has not been prepared")
        control, replacements = self.pending
        previous = self.current
        changed = (previous.digest if previous else Control(generation).digest) != control.digest
        # Mode changes reuse the already captured graphs and installed Python.
        if changed:
            for owner, attribute, existed, original in reversed(self.originals):
                if existed:
                    setattr(owner, attribute, original)
                else:
                    delattr(owner, attribute)
            self.originals = []
            for replacement in replacements:
                owner, attribute = replacement.owner, replacement.attribute
                self.originals.append((owner, attribute, attribute in vars(owner), getattr(owner, attribute)))
                setattr(owner, attribute, replacement.function)
        self.graphs_dirty |= changed or control.recapture
        self.current = control
        self.pending = None
        return self.graphs_dirty
