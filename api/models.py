from pydantic import BaseModel
from typing import Dict, Optional, List


class MakeupFeatures(BaseModel):
    hair: bool = True
    lips: bool = True
    skin: bool = False


class MakeupColors(BaseModel):
    hair: str = "Auburn"
    lips: str = "Ruby Red"
    skin: str = "Warm"


class MakeupRequest(BaseModel):
    selected_features: MakeupFeatures
    selected_colors: MakeupColors
    edge_smoothness: int = 71
    color_strength: float = 0.8
    detail_factor: float = 0.3


class MakeupResponse(BaseModel):
    success: bool
    message: str
    image_url: Optional[str] = None