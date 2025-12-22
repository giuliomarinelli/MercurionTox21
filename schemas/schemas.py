from typing_extensions import Annotated
from pydantic import BaseModel, StringConstraints, ConfigDict

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


class InferenceRequest(BaseModel):
    smiles: SmilesStr
    accessToken: TokenStr
    model_config = ConfigDict(extra="forbid")  # blocca campi extra nel payload
    
class Configuration(BaseModel):
    py_env: str
    nats_url: str
    version: str
    jwt_public_key_path: str