# NATS client user/password authentication

8 November 2026. Implemented on `feature/infra-vm-nats-client-auth`, based on
`develop` at `2cbfcd467afa9e9643065c630f6b06f265f0ac5a`. No merge or VM operation.

| Input | Contract |
| --- | --- |
| `NATS_URL` | Broker URL with `nats://` or `tls://`, optional port, no userinfo |
| `NATS_USERNAME` | Optional non-empty strict string; requires `NATS_PASSWORD` |
| `NATS_PASSWORD` | Optional non-empty strict secret string; requires `NATS_USERNAME` |

Omit both credentials to retain legacy unauthenticated connections. A declared
empty/null/non-string value or incomplete pair fails Pydantic validation before
model loading or broker connection. Diagnostics identify fields without values.
Values remain literal, including spaces and `$` characters; dotenv interpolation
stays disabled. Development/test process-over-file precedence is unchanged;
staging/production still never read dotenv files.

The password uses Pydantic
[`SecretStr`](https://pydantic.dev/docs/validation/latest/api/pydantic/types/#pydantic.types.SecretStr),
masked in model repr and JSON. `nats_connect_credentials()` unwraps it only to
construct the SDK `user`/`password` options for `nc.connect`. It does not append
credentials to the URL or print them. These are ordinary process inputs, not
native `_FILE` support or an implemented VM secret-delivery mechanism.

## Compatibility boundary

**URLs containing credentials are now rejected**, including when the separate
variables are absent. Migrate an existing authenticated URL by putting its
username/password in the two dedicated variables and removing userinfo from
`NATS_URL`. Preserve the decoded credential values; the new variables are not
URL-encoded. Never copy real credentials into shell history or tracked files.
This explicit validation change avoids two conflicting credential sources and
aligns the broker contract with MercurionWeb.

Broker authentication is separate from the five service JWT policies. Their
defaults, precedence, PEM contract and request/result shapes are unchanged.
Existing reconnect and lifecycle policy is unchanged; failed credentials do not
trigger an application-level anonymous fallback. No NKeys, account hierarchy,
TLS setup or subject ACL policy is introduced.

## Verification

- **107 application tests passed on Windows and Linux**, using existing
  dependencies. Includes omission/pair validation in all environments, literal
  values, masked diagnostics, local-file precedence, actual worker connect
  arguments, failure cleanup and existing JWT/handler regressions.
- Isolated real NATS smoke: **10 authenticated RPCs** across all five subjects
  in staging/production, **5 anonymous legacy RPCs**, **8 explicit worker auth
  denials** for missing credentials, wrong password, unknown user and credentials
  belonging to the other environment. Readiness stays false on denial and
  partial connections are closed. Network errors never count as auth denials.
- Inference and PubChem are synthetic substitutes in this smoke; three RDKit
  operations use the real implementation. No model weights, PEM or external
  provider are accessed. Separate Nest transport checks do not turn this into
  an integrated Nest-to-Tox21 acceptance claim.

Repeat with a locally built runtime image, from this repository root:

```bash
python -B scripts/smoke_nats_auth.py tox21-production --report /path/outside/repository/nats-auth.json
```

The default Docker context is `desktop-linux`; CI uses explicit
`--docker-context default`. Both require a local Unix/named-pipe socket. No pull,
host ports or persistent volumes; non-root owned containers and internal network
are removed by run label. Reports are emitted only after successful cleanup.

Local evidence used cached dependency image
`sha256:e0e45c20f9b4a88d28ffe228bd3a420033386aa53fd981d349482940a197d568`
with `--source-overlay`: only selected current source files/directories mounted
read-only, not a new release-image build. NATS was pinned to
`sha256:77a161c7d888a1791d5a46b50cd87a2f8466ce40a33cbb51238d227ace01e35a`.
The workflow adds the same smoke against its newly built production image,
without an overlay; remote CI remains a separate exact-SHA result.

Broker users/configuration, secret delivery and VM acceptance remain pending.
Do not enable operational broker auth before both consumer changes are integrated
and matching per-consumer/environment credentials are provisioned.
