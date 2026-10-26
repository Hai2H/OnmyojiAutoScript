"""Compatibility facade used by the API and tests."""

from .catalog import Catalog
from .paths import WorkspacePaths


class WorkbenchStore(Catalog):
    @classmethod
    def open(cls, paths: WorkspacePaths | None = None) -> 'WorkbenchStore':
        return cls.connect(paths or WorkspacePaths())
