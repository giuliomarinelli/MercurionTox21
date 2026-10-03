from typing_extensions import Annotated, Self
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


class Configuration(BaseModel):
    py_env: str
    nats_url: str
    version: str
    jwt_public_key_path: str
