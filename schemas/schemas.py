from typing_extensions import Annotated, Self
from typing import Literal
from urllib.parse import urlsplit
from pydantic import (
    BaseModel, StringConstraints, ConfigDict, Field,
    ValidationInfo, field_validator, model_validator,
)

SmilesStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1024,
    ),
]

TokenStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=10,
        max_length=4096,
    ),
]


class AuthenticatedRequest(BaseModel):
    accessToken: TokenStr | None = None
    model_config = ConfigDict(extra="forbid")  # blocca campi extra nel payload

    @field_validator("accessToken", mode="before")
    @classmethod
    def validate_access_token(cls, value, info: ValidationInfo):
        if info.context and info.context.get("skip_auth"):
            return None
        return value

    @model_validator(mode="after")
    def require_access_token(self, info: ValidationInfo) -> Self:
        if not (info.context and info.context.get("skip_auth")) and self.accessToken is None:
            raise ValueError("accessToken is required")
        return self


class InferenceRequest(AuthenticatedRequest):
    smiles: SmilesStr


class RdkitRequest(AuthenticatedRequest):
    model_config = ConfigDict(extra="forbid", strict=True)


class MoleculePropertiesRequest(RdkitRequest):
    smiles: SmilesStr


class CanonicalSmilesOptions(BaseModel):
    isomeric: bool = True
    kekule: bool = False
    model_config = ConfigDict(extra="forbid", strict=True)


class CanonicalSmilesRequest(MoleculePropertiesRequest):
    opts: CanonicalSmilesOptions | None = Field(default_factory=CanonicalSmilesOptions)


class SameStructureRequest(RdkitRequest):
    a: SmilesStr
    b: SmilesStr


AuthPolicy = Literal['allow', 'require-jwt', 'disabled']


class Configuration(BaseModel):
    py_env: Literal["development", "test", "staging", "production"]
    nats_url: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
    version: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
    jwt_public_key_path: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
    model_package_dir: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)] | None = None
    auth_inference: AuthPolicy = 'disabled'
    auth_rdkit_properties: AuthPolicy = 'disabled'
    auth_rdkit_canonical_smiles: AuthPolicy = 'disabled'
    auth_rdkit_same_structure: AuthPolicy = 'disabled'
    auth_pubchem_iupac_name: AuthPolicy = 'disabled'

    def resolved_skip_auth(self, global_skip_auth: bool) -> dict[str, bool]:
        """Resolve once at bootstrap; disabled means inherit, never disable RPC."""
        policies = {
            'inference.tox21.smiles': self.auth_inference,
            'rdkit_api.get_molecule_properties': self.auth_rdkit_properties,
            'rdkit_api.to_canonical_smiles': self.auth_rdkit_canonical_smiles,
            'rdkit_api.are_same_structure': self.auth_rdkit_same_structure,
            'pcp_api.get_iupac_name_from_smiles': self.auth_pubchem_iupac_name,
        }
        return {subject: global_skip_auth if policy == 'disabled' else policy == 'allow'
                for subject, policy in policies.items()}

    @field_validator("nats_url")
    @classmethod
    def validate_nats_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (parsed.scheme not in ("nats", "tls") or not parsed.hostname
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("Expected a NATS broker URL")
        if parsed.port is not None and parsed.port == 0:
            raise ValueError("Invalid NATS port")
        return value
