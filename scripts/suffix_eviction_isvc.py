#!/usr/bin/env python3
"""Switch the himanshu-llmd-test predictors between suffix-eviction A/B arms.

  treatment  suffix-* patch files mounted, CPU offload removed, flag on
  baseline   suffix-* patch files mounted, CPU offload removed, flag off
  restore    container spec from the --backup InferenceService YAML

The suffix-* files live on PVC himanshu-vllm-cpu-medium-patch next to the
CPU-offload overlays. suffix-block_pool.py carries both patches, so the
offload overlays keep working in every arm.

Usage:
  KUBECONFIG=... python3 scripts/suffix_eviction_isvc.py treatment --backup <isvc.yaml> \
      --env VLLM_SUFFIX_EVICTION_TIER_TOKENS=16384 \
      --env VLLM_SUFFIX_EVICTION_MAX_TIERS=16 \
      --env VLLM_SUFFIX_EVICTION_IDLE_TTL_S=60
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import yaml

SITE = "/usr/local/lib/python3.12/dist-packages/vllm"
VOLUME = "vllm-cpu-medium-patch"
SUFFIX_FILES = {
    "envs.py": "envs.py",
    "v1/core/block_pool.py": "block_pool.py",
    "v1/core/kv_cache_utils.py": "kv_cache_utils.py",
    "v1/core/single_type_kv_cache_manager.py": "single_type_kv_cache_manager.py",
    "v1/core/kv_cache_manager.py": "kv_cache_manager.py",
    "v1/core/sched/scheduler.py": "scheduler.py",
    "v1/worker/gpu/states.py": "gpu_states.py",
    "v1/worker/gpu/model_runner.py": "gpu_model_runner.py",
}
FLAG = "VLLM_SUFFIX_PRESERVING_EVICTION"


def kubectl(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(["kubectl", *args], input=stdin, check=True, capture_output=True, text=True).stdout


def arm_container(backup: dict, flag_on: bool, env: dict[str, str], extra_args: list[str]) -> dict:
    c = json.loads(json.dumps(backup))
    c["args"] = [a for a in c["args"] if not a.startswith("--kv-transfer-config")] + extra_args
    c["env"] = [e for e in c.get("env", []) if e["name"] != FLAG and e["name"] not in env]
    c["env"].append({"name": FLAG, "value": "1" if flag_on else "0"})
    c["env"] += [{"name": k, "value": v} for k, v in env.items()]
    targets = {f"{SITE}/{rel}": f"suffix-{name}" for rel, name in SUFFIX_FILES.items()}
    mounts = [m for m in c.get("volumeMounts", []) if m["mountPath"] not in targets]
    mounts += [{"mountPath": path, "name": VOLUME, "subPath": sub} for path, sub in targets.items()]
    c["volumeMounts"] = mounts
    return c


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", choices=("treatment", "baseline", "restore"))
    p.add_argument("--backup", required=True, help="InferenceService YAML saved before any change")
    p.add_argument("--namespace", default="nai-admin")
    p.add_argument("--name", default="himanshu-llmd-test")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VAL", help="extra container env (repeatable)")
    p.add_argument(
        "--arg", action="append", default=[], help="extra vLLM arg (repeatable), e.g. --arg=--enable-per-request-metrics"
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.backup) as f:
        backup = yaml.safe_load(f)["spec"]["predictor"]["containers"][0]
    env = dict(kv.split("=", 1) for kv in args.env)
    container = backup if args.arm == "restore" else arm_container(backup, args.arm == "treatment", env, args.arg)
    patch = [
        {"op": "replace", "path": f"/spec/predictor/containers/0/{key}", "value": container[key]}
        for key in ("args", "env", "volumeMounts")
    ]
    if args.dry_run:
        print(json.dumps(patch, indent=2))
        return 0
    print(kubectl("-n", args.namespace, "patch", "isvc", args.name, "--type=json", "-p", json.dumps(patch)).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
