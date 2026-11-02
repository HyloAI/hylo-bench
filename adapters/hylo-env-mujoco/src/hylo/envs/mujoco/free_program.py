"""Scene programs (`hylo.freescene/v0.1`): loading one Python file into a restricted namespace.

A scene program is a Python file with top-level declarations and functions:

    ROBOT = "panda" | "aloha" | "g1"     INSTRUCTION = "..."     MAX_STEPS = 900     LABELS = {...}   (optional)
    def build(scene): ...                 required: adds the task's things to scene.spec (free_model.SceneHandle)
    def success(sim) -> bool: ...         required: judges the state (free_world.SimView)
    def randomize(sim, rng): ...          optional: per-seed changes to the start (free_world.SimEdit)
    def demo(sim): ...                    optional: a scripted proof with invisible hands (checkers only)
    def targets(sim) -> {name: xyz}: ...  optional: points a robot hand must reach

`program_problems(source)` lists what keeps a source from loading (imports other than math / numpy / mujoco, names
or attributes starting with an underscore, file and code-loading calls, classes); `SceneProgram(path)` checks,
runs the top level with a small set of builtins and exposes the declarations. The code agents check programs the
same way before they reach an environment (hylo.codeagents.scene_free_check.validate_scene_program); this second
check guards files written by hand. Neither is an OS-level jail: run untrusted programs in a container.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import math
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

FORMAT = "hylo.freescene/v0.1"
ROBOTS = ("panda", "aloha", "g1")
DEFAULT_MAX_STEPS = 900

# The same guard runs twice on purpose: here, at load time, and in the code agents at build time
# (hylo.codeagents.scene_free_check.validate_scene_program). The boundary rules keep the bench and the harness from
# sharing the code, so keep the two allow-lists in step.
SCENE_IMPORTS = {"math", "numpy", "numpy.linalg", "numpy.random", "mujoco"}
DENIED_NAMES = {
    "open", "exec", "eval", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr", "input",
    "breakpoint", "help", "exit", "quit", "memoryview", "type", "object", "super", "dir", "__import__",
    "__builtins__", "importlib", "os", "sys", "subprocess", "classmethod", "staticmethod", "property",
}
DENIED_ATTRS = {
    # file and code loading / saving in numpy and mujoco
    "tofile", "fromfile", "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt",
    "fromregex", "memmap", "dump", "dumps", "ctypes", "ctypeslib", "lib", "f2py", "distutils", "testing", "system",
    "popen", "from_file", "to_file", "from_zip", "to_zip", "from_xml_path", "from_binary_path", "mj_saveLastXML",
    "mj_saveModel", "mj_loadXML", "mj_loadModel", "mj_printModel", "mj_printData", "mj_printSchema", "viewer",
    "rollout", "plugin_paths", "mj_loadPluginLibrary", "mj_loadAllPluginLibraries",
    # frames and code objects
    "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "f_globals", "f_locals", "f_builtins", "f_back",
    "f_code", "tb_frame", "tb_next", "co_code", "mro",
}
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "callable", "chr", "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "hash", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "ord", "pow",
    "print", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "AssertionError", "Exception", "IndexError", "KeyError", "LookupError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)


class SceneProgramError(ValueError):
    """The scene program is malformed, raised while building, or the model did not compile."""


def program_problems(source: str, filename: str = "scene.py") -> list[str]:
    """What keeps `source` from loading as a scene program (empty: none). The code agents validate programs the
    same way before they reach an environment."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]
    out: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            out += [f"line {line}: import {a.name} is not allowed (math, numpy, mujoco)" for a in node.names
                    if a.name not in SCENE_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in SCENE_IMPORTS:
                out.append(f"line {line}: from {node.module} import ... is not allowed (math, numpy, mujoco)")
        elif isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in DENIED_NAMES):
            out.append(f"line {line}: name {node.id!r} is not available")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in DENIED_ATTRS):
            out.append(f"line {line}: attribute .{node.attr} is not available")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await)):
            out.append(f"line {line}: class definitions and async code are not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.arg, ast.keyword)):
            name = getattr(node, "name", None) or getattr(node, "arg", None)
            if name and name.startswith("_"):
                out.append(f"line {line}: name {name!r} starts with an underscore")
    return out


class GuardedModule:
    """A read-only view of a module refusing private and denied attributes (numpy.linalg, numpy.random nested)."""

    def __init__(self, module: Any, name: str) -> None:
        object.__setattr__(self, "_m", module)
        object.__setattr__(self, "_n", name)

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_") or attr in DENIED_ATTRS:
            raise AttributeError(f"{self._n}.{attr} is not available in scene programs")
        value = getattr(self._m, attr)
        if self._n == "numpy" and attr in ("linalg", "random"):
            return GuardedModule(value, f"numpy.{attr}")
        return value

    def __setattr__(self, attr: str, value: Any) -> None:
        raise AttributeError(f"{self._n} is read-only")


def _modules() -> dict[str, Any]:
    import mujoco

    np_ = GuardedModule(np, "numpy")
    return {"math": GuardedModule(math, "math"), "numpy": np_, "numpy.linalg": np_.linalg, "numpy.random": np_.random,
            "mujoco": GuardedModule(mujoco, "mujoco")}


def _importer(mods: dict[str, Any]) -> Callable[..., Any]:
    def imp(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if not level and name not in SCENE_IMPORTS and (name.startswith(("numpy.", "mujoco.")) or name in ("numpy",)):
            return builtins.__import__(name, globals, locals, fromlist, level)   # numpy / mujoco internals
        if level or name not in SCENE_IMPORTS:
            raise ImportError(f"import {name!r} is not allowed in scene programs (math, numpy, mujoco)")
        if fromlist or "." not in name:
            return mods[name]
        return mods[name.split(".")[0]]
    return imp


class SceneProgram:
    """A loaded scene program: its declarations and functions (run in a restricted namespace)."""

    REQUIRED = ("ROBOT", "INSTRUCTION", "build", "success")

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "scene.py"
        self.path = p.resolve()
        self.source = self.path.read_text()
        self.digest = hashlib.sha256(self.source.encode()).hexdigest()[:16]
        problems = program_problems(self.source, self.path.name)
        if problems:
            raise SceneProgramError("the scene program was refused: " + "; ".join(problems[:8]))
        mods = _modules()
        safe = {n: getattr(builtins, n) for n in SAFE_BUILTINS}
        safe["__import__"] = _importer(mods)
        self.output: list[str] = []
        safe["print"] = lambda *a, sep=" ", end="\n", **_: self.output.append(sep.join(map(str, a)) + end)
        ns: dict[str, Any] = {"__builtins__": safe, "__name__": "scene_program", "np": mods["numpy"],
                              "math": mods["math"]}
        try:
            exec(compile(ast.parse(self.source), str(self.path), "exec"), ns)   # noqa: S102 - validated above
        except Exception as exc:  # noqa: BLE001 - the program's own error goes back to its author
            raise SceneProgramError(f"running the program's top level failed: {short_error(exc, self.path)}") from exc
        missing = [n for n in self.REQUIRED if n not in ns]
        if missing:
            raise SceneProgramError(f"the program must define {', '.join(missing)}")
        self.robot = str(ns["ROBOT"]).lower().strip()
        if self.robot not in ROBOTS:
            raise SceneProgramError(f"ROBOT must be one of {list(ROBOTS)}, got {ns['ROBOT']!r}")
        self.instruction = str(ns["INSTRUCTION"]).strip()
        if not self.instruction:
            raise SceneProgramError("INSTRUCTION is empty")
        self.max_steps = int(ns.get("MAX_STEPS", DEFAULT_MAX_STEPS))
        self.labels = {str(k): str(v) for k, v in dict(ns.get("LABELS") or {}).items()}
        for fn in ("build", "success", "randomize", "demo", "targets"):
            if fn in ns and not callable(ns[fn]):
                raise SceneProgramError(f"{fn} must be a function")
        self.build = ns["build"]
        self.success = ns["success"]
        self.randomize = ns.get("randomize")
        self.demo = ns.get("demo")
        self.targets = ns.get("targets")
        self.namespace = ns

    def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Call one of the program's functions; its errors come back as SceneProgramError with program lines."""
        try:
            return fn(*args)
        except SceneProgramError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SceneProgramError(f"{getattr(fn, '__name__', 'function')}() raised {short_error(exc, self.path)}") from exc


def short_error(exc: BaseException, path: Path | None = None) -> str:
    """`Type: message` with the program's own lines from the traceback (others summarised)."""
    frames = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in frames if path is not None and f.filename == str(path)]
    lines = [f"line {f.lineno} in {f.name}: {(f.line or '').strip()}" for f in mine[-4:]]
    msg = f"{type(exc).__name__}: {str(exc)[:600]}"
    return (" | ".join(lines) + " -> " if lines else "") + msg


__all__ = ["DEFAULT_MAX_STEPS", "DENIED_ATTRS", "DENIED_NAMES", "FORMAT", "GuardedModule", "ROBOTS", "SCENE_IMPORTS", "SceneProgram",
           "SceneProgramError", "program_problems", "short_error"]
