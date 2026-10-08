#!/usr/bin/env python3
# Copyright 2026
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
"""Run live suffix-eviction arms on the himanshu-llmd-test predictors.

For each arm: switch the InferenceService with suffix_eviction_isvc.py, wait
for all replicas to be replaced and ready, run one spec per pod in parallel
with suffix_eviction_live.py (fresh server, no initial reset), optionally run
the gap sweep on the first pod, and copy the results to --out/<arm>/.

Usage:
  KUBECONFIG=... suffix_eviction_live_arms.py --backup isvc.yaml \
      --specs a.json,b.json,c.json --arms vllm-native-lru,vllm-prefix-pruning-4096 --out DIR
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TIER = "VLLM_SUFFIX_EVICTION_TIER_TOKENS"
STEP = "VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS"
REMOTE = "/tmp/live"


def arm_settings(arm: str) -> tuple[str, dict[str, str]]:
    """'vllm-native-lru' | 'fraction-F' | 'vllm-prefix-pruning-T' -> (isvc arm, env)."""
    if arm in ("vllm-native-lru", "stock"):
        return "baseline", {TIER: "0", STEP: "0"}
    kind, *vals = arm.split("-")
    if kind == "vllm" and vals[:2] == ["prefix", "pruning"]:
        kind, vals = "tiers", vals[2:]
    if kind == "fraction":
        return "treatment", {"VLLM_SUFFIX_EVICTION_PREFIX_FRACTION": vals[0], TIER: "0", STEP: "0"}
    if kind == "tiers":
        return "treatment", {TIER: vals[0], STEP: vals[1] if len(vals) > 1 else "0"}
    raise ValueError(arm)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Kube:
    def __init__(self, namespace: str, name: str):
        self.ns, self.name = namespace, name
        self.deploy = f"{name}-predictor"

    def run(self, *args: str, check: bool = True, timeout: float | None = None) -> str:
        r = subprocess.run(["kubectl", "-n", self.ns, *args], capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout

    def pods(self) -> list[dict]:
        out = json.loads(self.run("get", "pods", "-l", f"serving.kserve.io/inferenceservice={self.name}", "-o", "json"))
        return [p for p in out["items"] if p["metadata"]["labels"].get("component") == "predictor"]

    def wait_arm(self, flag: str, env: dict[str, str], replicas: int, since: float, timeout: float) -> list[dict]:
        """Wait until `replicas` ready, non-terminating pods created after
        `since` carry the arm's env."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            good = []
            for p in self.pods():
                meta = p["metadata"]
                if meta.get("deletionTimestamp"):
                    continue
                created = calendar.timegm(time.strptime(meta["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"))
                if created < since - 5:
                    continue
                c = p["spec"]["containers"][0]
                penv = {e["name"]: e.get("value") for e in c.get("env", [])}
                if penv.get("VLLM_SUFFIX_PRESERVING_EVICTION") != flag:
                    continue
                if any(penv.get(k) != v for k, v in env.items()):
                    continue
                conds = {x["type"]: x["status"] for x in p.get("status", {}).get("conditions", [])}
                if conds.get("Ready") == "True":
                    good.append(p)
            if len(good) >= replicas:
                return sorted(good, key=lambda p: p["metadata"]["name"])
            time.sleep(15)
        raise TimeoutError("arm rollout did not finish")

    def cp_to(self, pod: str, src: str, dst: str) -> None:
        self.run("cp", src, f"{pod}:{dst}", check=False)

    def cp_from(self, pod: str, src: str, dst: str) -> None:
        self.run("cp", f"{pod}:{src}", dst, check=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup", required=True)
    p.add_argument("--specs", required=True, help="comma-separated spec JSONs, one per pod")
    p.add_argument("--arms", required=True)
    p.add_argument("--gap-arms", default="vllm-native-lru,vllm-prefix-pruning-262144")
    p.add_argument("--gaps", default="0,256,1024,4096,8192,16384")
    p.add_argument("--gap-reps", type=int, default=3)
    p.add_argument("--final-arm", default="baseline", choices=("baseline", "none"))
    p.add_argument("--namespace", default="nai-admin")
    p.add_argument("--name", default="himanshu-llmd-test")
    p.add_argument("--rollout-timeout", type=float, default=2700)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    kube = Kube(a.namespace, a.name)
    specs = a.specs.split(",")
    gap_arms = set(a.gap_arms.split(",")) if a.gap_arms else set()
    for arm in a.arms.split(","):
        isvc_arm, env = arm_settings(arm)
        flag = "1" if isvc_arm == "treatment" else "0"
        log(f"arm {arm}: switching isvc ({isvc_arm}, {env})")
        since = time.time()
        cmd = [
            sys.executable,
            os.path.join(HERE, "suffix_eviction_isvc.py"),
            isvc_arm,
            "--backup",
            a.backup,
            "--namespace",
            a.namespace,
            "--name",
            a.name,
            "--arg=--enable-per-request-metrics",
        ]
        cmd += [f"--env={k}={v}" for k, v in env.items()]
        subprocess.run(cmd, check=True)
        pods = kube.wait_arm(flag, env, len(specs), since, a.rollout_timeout)
        names = [x["metadata"]["name"] for x in pods][: len(specs)]
        log(f"arm {arm}: ready on {names}")
        arm_dir = os.path.join(a.out, arm)
        os.makedirs(arm_dir, exist_ok=True)
        procs = []
        for pod, spec in zip(names, specs, strict=False):
            kube.run("exec", pod, "--", "mkdir", "-p", REMOTE)
            for f in (os.path.join(HERE, "suffix_eviction_live.py"), os.path.join(HERE, "suffix_eviction_workload.py"), spec):
                kube.cp_to(pod, f, f"{REMOTE}/{os.path.basename(f)}")
            tag = os.path.splitext(os.path.basename(spec))[0]
            remote_cmd = (
                f"cd {REMOTE} && timeout 3600 python3 suffix_eviction_live.py run --no-reset "
                f"--spec {os.path.basename(spec)} --label {arm} --out out-{arm}-{tag}"
            )
            logf = open(os.path.join(arm_dir, f"{tag}.log"), "w")
            procs.append(
                (
                    pod,
                    tag,
                    subprocess.Popen(
                        ["kubectl", "-n", a.namespace, "exec", pod, "--", "sh", "-c", remote_cmd],
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                    ),
                )
            )
        for pod, tag, proc in procs:
            rc = proc.wait()
            kube.cp_from(pod, f"{REMOTE}/out-{arm}-{tag}", os.path.join(arm_dir, tag))
            node = next(x["spec"]["nodeName"] for x in pods if x["metadata"]["name"] == pod)
            with open(os.path.join(arm_dir, tag, "pod.json"), "w") as f:
                json.dump({"pod": pod, "node": node, "exit": rc}, f)
            log(f"arm {arm}: {tag} on {pod} exit {rc}")
        if arm in gap_arms:
            pod = names[0]
            log(f"arm {arm}: gap sweep on {pod}")
            kube.run(
                "exec",
                pod,
                "--",
                "sh",
                "-c",
                f"cd {REMOTE} && timeout 3600 python3 suffix_eviction_live.py gap --gaps {a.gaps} "
                f"--reps {a.gap_reps} --filler-tokens 65536 --out out-{arm}-gap",
                timeout=4000,
            )
            kube.cp_from(pod, f"{REMOTE}/out-{arm}-gap", os.path.join(arm_dir, "gap"))
            log(f"arm {arm}: gap sweep done")
    if a.final_arm == "baseline":
        log("restoring baseline arm")
        subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "suffix_eviction_isvc.py"),
                "baseline",
                "--backup",
                a.backup,
                "--namespace",
                a.namespace,
                "--name",
                a.name,
                f"--env={TIER}=0",
                f"--env={STEP}=0",
            ],
            check=True,
        )
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
