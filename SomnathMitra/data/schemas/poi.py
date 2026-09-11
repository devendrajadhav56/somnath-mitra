from typing import Optional

from pydantic import BaseModel


class GeoPoint(BaseModel):
    type: str = "Point"
    coordinates: list[float]  # [lon, lat]


class Poi(BaseModel):
    source_id: str
    name: str
    category: Optional[str] = None
    subtype: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    opening_hours: Optional[str] = None
    location: GeoPoint
    merged_from: list[str] = []
    schema_version: int = 1
