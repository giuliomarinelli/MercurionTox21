"""Minimal worker CMD/start-stop smoke, using an owned isolated NATS broker only."""
import argparse
import json
import subprocess
import time
import uuid

NATS_IMAGE = "nats@sha256:77a161c7d888a1791d5a46b50cd87a2f8466ce40a33cbb51238d227ace01e35a"


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--target", choices=["test", "staging", "production"], required=True)
    args = parser.parse_args()
    prefix = "mercurion-t21-04-" + uuid.uuid4().hex[:10]
    network = broker = worker = None
    try:
        network = docker("network", "create", "--internal", prefix)
        broker = docker("run", "-d", "--name", prefix + "-nats", "--network", network,
                        "--network-alias", "nats", NATS_IMAGE)
        worker = docker("run", "-d", "--name", prefix + "-worker", "--network", network,
                        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
                        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--env", "NATS_URL=nats://nats:4222",
                        args.image)
        subject = "inference.tox21.smiles" if args.target == "production" else args.target + ".inference.tox21.smiles"
        code = f'''
import asyncio,json
from nats.aio.client import Client
async def request():
    nc=Client()
    try:
        await nc.connect('nats://nats:4222',connect_timeout=1,max_reconnect_attempts=0)
        reply=await nc.request({subject!r},json.dumps({{'smiles':'CCO'}}).encode(),timeout=1)
        result=json.loads(reply.data)
        assert set(result)=={{'SR-ATAD5','NR-AhR','SR-MMP','SR-p53'}}
        assert all(0<=item['probability']<=1 for item in result.values())
    finally:
        await nc.close()
asyncio.run(request())
'''
        deadline = time.monotonic() + 45
        while True:
            result = subprocess.run(["docker", "exec", worker, "python", "-B", "-c", code],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                break
            state = json.loads(docker("inspect", "--format", "{{json .State}}", worker))
            if not state["Running"] or time.monotonic() >= deadline:
                raise RuntimeError("Worker did not answer isolated inference request: " + docker("logs", worker))
            time.sleep(0.5)
        docker("stop", "--time", "15", worker)
        state = json.loads(docker("inspect", "--format", "{{json .State}}", worker))
        assert state["ExitCode"] == 0 and not state["OOMKilled"], state
        print(json.dumps({"image": args.image, "target": args.target, "nats_rpc": "passed",
                          "stop_exit_code": state["ExitCode"], "host_ports": "none"}))
    finally:
        for owned in (worker, broker):
            if owned:
                subprocess.run(["docker", "rm", "--force", owned], capture_output=True, check=False)
        if network:
            subprocess.run(["docker", "network", "rm", network], capture_output=True, check=False)


if __name__ == "__main__":
    main()
