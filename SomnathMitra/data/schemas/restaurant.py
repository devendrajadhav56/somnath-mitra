from typing import Optional

from pydantic import BaseModel

from schemas.poi import GeoPoint


class Restaurant(BaseModel):
    source_id: str
    name: str
    category: Optional[str] = None
    category_raw: Optional[str] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None
    has_website: Optional[bool] = None
    place_id: Optional[str] = None
    cid: Optional[str] = None
    google_maps_url: Optional[str] = None
    location: GeoPoint
    scraped_at: Optional[str] = None
    merged_from: list[str] = []
    distance_from_temple_km: Optional[float] = None
    schema_version: int = 1
