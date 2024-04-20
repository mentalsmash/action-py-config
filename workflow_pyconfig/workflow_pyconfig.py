###############################################################################
# Copyright 2020-2024 Andrea Sorbini
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
import yaml
import json
import sys
import os
import importlib
import importlib.util
from pathlib import Path
from typing import NamedTuple
from collections import namedtuple

def dict_to_tuple(key: str, val: dict) -> NamedTuple:
  fields = {}
  for k, v in val.items():
    if isinstance(v, dict):
      v = dict_to_tuple(k, v)
    k = k.replace("-", "_")
    fields[k] = v
  
  keys = list(fields.keys())
  if not keys:
    return tuple()

  val_cls = namedtuple(key, keys)
  return val_cls(**fields)

def merge_dicts(result: dict, defaults: dict) -> dict:
  merged = {}
  for k, v in defaults.items():
    r_v = result.get(k)
    if isinstance(v, dict):
      r_v = merge_dicts((r_v or {}), v)
    elif r_v is None:
      r_v = v
    merged[k] = r_v
  return merged

def load_json_input(name: str, val: str) -> tuple:
  if val:
    return dict_to_tuple(name, json.loads(val))
  else:
    return tuple()

def write_output(vars: dict[str, bool | str | int | None] | None = None):
  def _output(var: str, val: bool | str | int | None):
    assert val is None or isinstance(val, (bool,  str, int)), f"unsupported output value type: {var} = {val.__class__}"
    if val is None:
      val = ""
    elif isinstance(val, bool):
      # Normalize booleans to non-empty/empty strings
      # Use lowercase variable name for easier debugging
      val = var.lower() if val else ""
    elif not isinstance(val, str):
      val = str(val)
    print(f"OUTPUT [{var}]: {val}")
    if "\n" not in val:
      output.write(var)
      output.write("=")
      if val:
        output.write(val)
      output.write("\n")
    else:
      output.write(f"{var}<<EOF""\n")
      output.write(val)
      output.write("\n")
      output.write("EOF\n")
  github_output = Path(os.environ["GITHUB_OUTPUT"])
  with github_output.open("a") as output:
    for var, val in (vars or {}).items():
      _output(var, val)


def lookup_config_val(ctx: NamedTuple, selector: str) -> str:
  def _getattr(obj, k):
    if isinstance(obj, dict):
      return obj[k]
    else:
      return getattr(obj, k)
  def _lookup_recur(current: NamedTuple, selector_parts: list[str]) -> str:
    selected = _getattr(current, selector_parts[0])
    if len(selector_parts) == 1:
      return selected
    return _lookup_recur(selected, selector_parts[1:])
  selector_parts = selector.split(".")
  if not selector_parts:
    raise RuntimeError("a non-empty selector is required")
  return _lookup_recur(ctx, selector_parts)


def load_file_module(module: str, pyfile: Path):
  spec = importlib.util.spec_from_file_location(module, str(pyfile))
  pymodule = importlib.util.module_from_spec(spec)
  sys.modules[module] = pymodule
  spec.loader.exec_module(pymodule)
  return pymodule

def action(
    github: str,
    inputs: str,
    outputs: str,
    clone_dir: str = "src/repo",
    config_dir: str = ".workflow-pyconfig",
    settings: str = ".workflow-pyconfig/settings.yml",
    workflow: str | None = None):
  github = load_json_input("github", github)
  inputs = load_json_input("inputs", inputs)
  clone_dir = Path(clone_dir)

  pyconfig_dir = clone_dir / config_dir
  if pyconfig_dir.exists():
    sys.path.insert(0, pyconfig_dir)

  cfg_file = clone_dir / settings
  if cfg_file and cfg_file.exists():
    cfg_dict = yaml.safe_load(cfg_file.read_text())
  else:
    cfg_dict = {}

  settings_file = pyconfig_dir / "settings.py"
  settings_mod = load_file_module("settings", settings_file)

  derived_cfg = settings_mod.settings(
    clone_dir=clone_dir,
    cfg=dict_to_tuple("settings", cfg_dict),
    github=github)

  # cfg = merge_dicts(derived_cfg, cfg_dict)
  cfg_dict["dyn"] = derived_cfg
  cfg = dict_to_tuple("settings", cfg_dict)

  action_outputs = {
    "CLONE_DIR": str(clone_dir),
  }

  if outputs is not None:
    for line in outputs.splitlines():
      var, val_k = line.split("=")
      ctx_name = val_k[:val_k.index(".")]
      ctx_select = val_k[len(ctx_name)+2:]
      ctx = {
        "cfg": cfg,
        "env": os.environ,
        "github": github,
        "inputs": inputs,
      }[ctx_name]
      action_outputs[var] = lookup_config_val(ctx, ctx_select)

  if workflow:
    workflow_file = pyconfig_dir / f"workflows/{workflow}.py"
    workflow_mod = load_file_module(f"workflows.{workflow}", workflow_file)
    dyn_outputs = workflow_mod.configure(
      clone_dir=clone_dir,
      cfg=cfg,
      github=github,
      inputs=inputs)
    if dyn_outputs:
      action_outputs.update(dyn_outputs)

  if action_outputs:
    write_output(action_outputs)
