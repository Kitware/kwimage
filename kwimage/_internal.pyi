from typing import Union
from _typeshed import Incomplete

KWIMAGE_DISABLE_WARNINGS: Incomplete
KWIMAGE_DISABLE_TRANSFORM_WARNINGS: Incomplete
KWIMAGE_DISABLE_IMPORT_WARNINGS: Incomplete
KWIMAGE_DISABLE_TORCHVISION_NMS: Incomplete
KWIMAGE_DISABLE_C_EXTENSIONS: Incomplete


def schedule_deprecation(modname: str,
                         name: str = '?',
                         type: str = '?',
                         migration: str = '',
                         deprecate: Union[str, None] = None,
                         error: Union[str, None] = None,
                         remove: Union[str, None] = None) -> None:
    ...
