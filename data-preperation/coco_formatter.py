from typing import List, Union

from pydantic import BaseModel

class Category(BaseModel):
    id: int
    name: str
    supercategory: Union[str, None] = None

class License(BaseModel):
    id: int = 1
    name: str = 'Default License'
    url: str = ''

class Image(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    license: int

class Bbox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int

# class DetectionAnnotation(Annotation):
#     bbox: Bbox
#     area: int
#     is_crowd: int = 0

class DetectionAnnotation(Annotation):
    bbox: List[int]
    area: int
    iscrowd: int = 0

class Info(BaseModel):
    description: str
    version: str = '1.0.0'
    contributor: str = 'Emre Kara'
    date_created: str

class Dataset(BaseModel):
    info: Info
    licenses: List[License] = [License()]
    images: List[Image]
    annotations: List[DetectionAnnotation]
    categories: List[Category]

