from pydantic import BaseModel, ValidationError

class InferenceRequest(BaseModel):
    smiles: str
    accessToken: str 
