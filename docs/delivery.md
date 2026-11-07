# T21-07 — Candidate archives and security evidence

Channel: **local Docker archives**, no registry or deployment. Image identity,
transport integrity, functional tests and security eligibility are separate.

The image contains `/usr/share/mercurion/build-identity.json`, generated from
the same version/source SHA as its OCI labels. `runtime.read_build_identity()`
reads it without environment substitution; startup prints it. Source checkouts
return `None` rather than inventing a revision; malformed present metadata fails
startup. Runtime `VERSION` overrides do not rewrite the baked identity.

```powershell
docker run --rm --network none --read-only --entrypoint python IMAGE -B -c "import json; from runtime import read_build_identity; print(json.dumps(read_build_identity()))"
python -B scripts/container_evidence.py --image IMAGE --output C:/artifacts/new-candidate --cache C:/artifacts/trivy-cache
```

Use a new output directory. The exporter runs the offline runtime/model audit,
saves the image **by immutable local ID without tags**, and generates a CycloneDX
SBOM plus a complete Trivy report. The pinned scanner downloads its public DB;
it does not upload images. [Trivy archives](https://trivy.dev/docs/latest/target/container_image/),
[SBOM documentation](https://trivy.dev/docs/latest/supply-chain/sbom/).

The report must match the archive's image config digest. The
[policy](container-policy.json) inherits Web's critical-severity block, with no
exceptions or ignored/unfixed findings. All severity counts and OS status remain
visible. Missing/malformed scans fail closed. This gate does not certify license
compliance or complete native-library coverage; the SBOM supports further review.

The exporter executes `docker load` on the untagged archive and checks restored
labels, architecture and filesystem layers against its config. It never removes
source images/tags and cannot overwrite named images through an untagged archive.
`component.json` remains a **candidate**: source/version/target, archive SHA256,
portable config digest, Docker image ID, model manifest, scan/policy hashes and
load evidence. The SBOM, runtime audit and full vulnerability report are retained.

Docker Desktop can expose an OCI index as its local ID while other engines use
a config digest; neither is called a verified registry digest. Loading into the
current daemon proves archive acceptance/identity, not a cold restore on a new
machine. It starts no services. [Docker save](https://docs.docker.com/reference/cli/docker/image/save/).

Exit 0 means the implemented critical-vulnerability gate passed. Exit **2** means
export/evidence is valid but **security eligibility is denied**. Tool, scanner or
integrity failures also exit nonzero. Diagnostics are retained; no failure is
translated into approval. CI runs this after functional tests; a critical finding
makes verification and Required gate red. Only JSON/log diagnostics are uploaded,
**not image archives**. A later coordinated manifest must include real records
for both Web images; missing components are explicit, never substituted by Tox21.

## Historical block — 6 November 2026

The T21-04 production baseline scan found **18 critical package/advisory
occurrences**, not 18 unique CVE IDs: 13 OS-package occurrences and five Pillow
occurrences. Debian 11.7 was reported EOSL. Other severity findings also remain
visible. Existing base/dependency pins were retained, not silently upgraded.

Updating the OS/Python base and Pillow compatibility requires a separate baseline
change and native/model regression validation. No exception, filter or reduced
threshold is adopted to obtain a green result. This blocks approved delivery.

The [7 November baseline revision](security-baseline.md) replaces those OS/Python/
Pillow pins. Eligibility is established by the gate on the exact resulting image,
not by treating this historical report as the current scan or removing its findings.

The expanded suite has **107 self-contained cases**, **93 application cases**,
model/export 9 and synthetic training 5. The reference group stays separate.
Coordinated delivery remains blocked pending security remediation and actual Web
image records. Transport QA does not close T21-07 or prove deployment acceptance.
