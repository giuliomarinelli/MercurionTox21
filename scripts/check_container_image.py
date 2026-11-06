"""Inspect a built worker image and run offline inference in its restricted runtime."""
import argparse
import json
import subprocess


AUDIT = r'''
import importlib.util,json,os,pathlib
assert (os.getuid(),os.getgid()) == (10021,10021)
assert os.statvfs('/app').f_flag & os.ST_RDONLY
assert os.environ['MODEL_PACKAGE_DIR']=='/app/model'
for name in ['pandas','sklearn','matplotlib','mercurion.train','mercurion.calibration',
             'mercurion.export_model','mercurion.preprocessing','mercurion.early_stopping',
             'mercurion.focal_loss','tests','pip']:
    assert importlib.util.find_spec(name) is None,name
expected={'main.py','config.py','runtime.py','api','schemas','mercurion','model'}
assert {p.name for p in pathlib.Path('/app').iterdir()}==expected
assert {p.name for p in pathlib.Path('/app/model').iterdir()}=={'weights.pt','thresholds.json','manifest.json'}
assert not os.access('/app/model/weights.pt',os.W_OK)
pathlib.Path('/tmp/runtime-write-check').write_text('temporary writes allowed')
from runtime import load_inference_runtime
model,device,predict=load_inference_runtime(package_dir='/app/model')
a=predict('CCO',model,device); b=predict('CCO',model,device)
assert a==b and set(a)=={'SR-ATAD5','NR-AhR','SR-MMP','SR-p53'}
assert all(0<=item['probability']<=1 for item in a.values())
print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'device':device,'environment':os.environ['PY_ENV'],
                  'readonly':True,'training_absent':True,'repeatable_inference':True}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--target", choices=["test", "staging", "production"], required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    config = image["Config"]
    assert config["User"] == "10021:10021"
    assert config["Cmd"] == ["python", "-m", "main"]
    assert not config.get("ExposedPorts"), "Worker must not invent a listening HTTP port"
    labels = config["Labels"]
    assert labels["org.opencontainers.image.revision"] == args.revision
    assert labels["org.opencontainers.image.version"] == args.version
    assert labels["org.opencontainers.image.source"] == "https://github.com/giuliomarinelli/MercurionTox21"
    assert labels["org.mercurion.environment"] == args.target
    output = subprocess.check_output([
        "docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--entrypoint", "python", args.image, "-B", "-c", AUDIT,
    ], text=True)
    result = json.loads(output)
    assert result["environment"] == args.target
    print(json.dumps({"image": args.image, "image_id": image["Id"], "size_bytes": image["Size"], **result}))


if __name__ == "__main__":
    main()
