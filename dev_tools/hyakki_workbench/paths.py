"""Filesystem layout used by the workbench."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkspacePaths:
    """All mutable workbench paths, with project-root defaults."""

    project_root: Path = PROJECT_ROOT
    bean_capture: Path | None = None
    boss_capture: Path | None = None
    state_root: Path | None = None
    runs_root: Path | None = None
    weights_root: Path | None = None

    def __post_init__(self) -> None:
        root = Path(self.project_root).resolve()
        object.__setattr__(self, 'project_root', root)
        object.__setattr__(
            self,
            'bean_capture',
            Path(
                self.bean_capture
                or os.environ.get('HYAKKI_BEAN_CAPTURE', root / 'data' / 'bean_capture')
            ).resolve(),
        )
        object.__setattr__(
            self,
            'boss_capture',
            Path(
                self.boss_capture
                or os.environ.get(
                    'HYAKKI_BOSS_CAPTURE',
                    root / 'log' / 'hyakkiyakou' / 'boss_capture',
                )
            ).resolve(),
        )
        object.__setattr__(
            self,
            'state_root',
            Path(
                self.state_root
                or os.environ.get(
                    'HYAKKI_WORKBENCH_DATA',
                    root / 'data' / 'hyakki_workbench',
                )
            ).resolve(),
        )
        object.__setattr__(
            self,
            'runs_root',
            Path(
                self.runs_root
                or os.environ.get(
                    'HYAKKI_WORKBENCH_RUNS',
                    root / 'runs' / 'hyakki_workbench',
                )
            ).resolve(),
        )
        object.__setattr__(
            self,
            'weights_root',
            Path(
                self.weights_root
                or os.environ.get(
                    'HYAKKI_WORKBENCH_WEIGHTS',
                    root / 'weights' / 'hyakkiyakou',
                )
            ).resolve(),
        )

    @property
    def database(self) -> Path:
        return self.state_root / 'workbench.sqlite3'

    @property
    def datasets_root(self) -> Path:
        return self.state_root / 'datasets'

    @property
    def exports_root(self) -> Path:
        return self.state_root / 'exports'

    @property
    def static_root(self) -> Path:
        return Path(__file__).resolve().parent / 'static'

    @property
    def train_python(self) -> Path:
        return self.project_root / '.venv-train' / 'Scripts' / 'python.exe'

    @property
    def runtime_python(self) -> Path:
        return self.project_root / '.venv' / 'Scripts' / 'python.exe'

    def ensure(self) -> None:
        """Create only workbench-owned output directories."""
        for folder in (
            self.state_root,
            self.datasets_root,
            self.exports_root,
            self.runs_root,
            self.weights_root,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def is_source_media(self, path: Path) -> bool:
        candidate = Path(path).resolve()
        for root in (self.bean_capture, self.boss_capture):
            try:
                candidate.relative_to(root)
                return candidate.is_file()
            except ValueError:
                continue
        return False
