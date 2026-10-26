"""FastAPI entry point for the local Hyakki dataset workbench."""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .paths import WorkspacePaths
from .store import WorkbenchStore
from .dataset import DatasetBuilder


class ReviewRequest(BaseModel):
    action: str = Field(pattern='^(correct|change|unknown|needs_label|skip|reopen)$')
    class_id: int | None = None
    note: str = ''


class XAnyConfigRequest(BaseModel):
    executable: str = ''


class DatasetBuildRequest(BaseModel):
    class_ids: list[int] | None = None


def create_app(paths: WorkspacePaths | None = None) -> FastAPI:
    workspace = paths or WorkspacePaths()
    workspace.ensure()
    scan_lock = threading.Lock()

    app = FastAPI(title='百鬼夜行训练工作台', version='0.1.0')

    @contextmanager
    def open_store() -> Iterator[WorkbenchStore]:
        connection = WorkbenchStore.open(workspace)
        try:
            yield connection
        finally:
            connection.close()

    @app.get('/', include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(workspace.static_root / 'index.html')

    @app.get('/static/{asset_path:path}', include_in_schema=False)
    def static_file(asset_path: str) -> FileResponse:
        target = (workspace.static_root / asset_path).resolve()
        try:
            target.relative_to(workspace.static_root.resolve())
        except ValueError as exception:
            raise HTTPException(status_code=404, detail='Not found') from exception
        if not target.is_file():
            raise HTTPException(status_code=404, detail='Not found')
        return FileResponse(target)

    @app.get('/api/status')
    def status() -> dict:
        with open_store() as catalog:
            return {
                'ok': True,
                'paths': {
                    'project_root': str(workspace.project_root),
                    'bean_capture': str(workspace.bean_capture),
                    'boss_capture': str(workspace.boss_capture),
                    'state_root': str(workspace.state_root),
                },
                **catalog.summary(),
            }

    @app.post('/api/scan')
    def scan() -> dict:
        if not scan_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail='扫描正在进行')
        try:
            with open_store() as catalog:
                return {'ok': True, **catalog.scan().as_dict()}
        finally:
            scan_lock.release()

    @app.get('/api/events')
    def events(
        status_value: str | None = Query(default=None, alias='status'),
        source: str | None = None,
        rarity: str | None = None,
        pending_high_only: bool = False,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        with open_store() as catalog:
            return catalog.list_events(
                status=status_value,
                source=source,
                rarity=rarity,
                pending_high_only=pending_high_only,
                limit=limit,
                offset=offset,
            )

    @app.get('/api/events/{event_id}')
    def event_detail(event_id: str) -> dict:
        with open_store() as catalog:
            event = catalog.get_event(event_id)
            if not event:
                raise HTTPException(status_code=404, detail='事件不存在')
            event['review_history'] = catalog.review_history(event_id)
            return event

    @app.post('/api/events/{event_id}/review')
    def review_event(event_id: str, request: ReviewRequest) -> dict:
        with open_store() as catalog:
            try:
                return catalog.review_event(
                    event_id,
                    action=request.action,
                    class_id=request.class_id,
                    note=request.note,
                )
            except KeyError as exception:
                raise HTTPException(status_code=404, detail='事件不存在') from exception
            except ValueError as exception:
                raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.post('/api/events/{event_id}/prepare-label')
    def prepare_label(event_id: str) -> dict:
        with open_store() as catalog:
            try:
                return catalog.prepare_label_package(event_id)
            except KeyError as exception:
                raise HTTPException(status_code=404, detail='事件不存在') from exception

    @app.post('/api/events/{event_id}/import-label')
    def import_label(event_id: str) -> dict:
        with open_store() as catalog:
            try:
                return catalog.import_label_package(event_id)
            except KeyError as exception:
                raise HTTPException(status_code=404, detail='事件不存在') from exception
            except ValueError as exception:
                raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.get('/api/classes')
    def classes(
        query: str = '',
        rarity: str | None = None,
        limit: int = Query(default=300, ge=1, le=500),
    ) -> dict:
        with open_store() as catalog:
            return {'items': catalog.list_classes(query=query, rarity=rarity, limit=limit)}

    @app.get('/api/dataset/readiness')
    def dataset_readiness() -> dict:
        with open_store() as catalog:
            return DatasetBuilder(catalog).readiness()

    @app.get('/api/datasets')
    def datasets() -> dict:
        with open_store() as catalog:
            return {'items': DatasetBuilder(catalog).list_datasets()}

    @app.post('/api/datasets/build')
    def build_dataset(request: DatasetBuildRequest) -> dict:
        with open_store() as catalog:
            try:
                return DatasetBuilder(catalog).build(request.class_ids)
            except ValueError as exception:
                raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.get('/api/xany')
    def xany_config() -> dict:
        with open_store() as catalog:
            executable = catalog.get_setting('xany_executable', '') or ''
            return {
                'executable': executable,
                'exists': bool(executable and Path(executable).is_file()),
                'url': 'https://github.com/CVHub520/X-AnyLabeling',
            }

    @app.post('/api/xany/config')
    def configure_xany(request: XAnyConfigRequest) -> dict:
        executable = request.executable.strip()
        if executable and not Path(executable).is_file():
            raise HTTPException(status_code=422, detail='X-AnyLabeling 可执行文件不存在')
        with open_store() as catalog:
            catalog.set_setting('xany_executable', executable)
        return {'ok': True, 'executable': executable}

    @app.post('/api/xany/launch')
    def launch_xany() -> dict:
        with open_store() as catalog:
            executable = catalog.get_setting('xany_executable', '') or ''
        if not executable or not Path(executable).is_file():
            raise HTTPException(status_code=422, detail='请先配置 X-AnyLabeling 可执行文件')
        try:
            subprocess.Popen([executable], cwd=str(Path(executable).parent))
        except OSError as exception:
            raise HTTPException(status_code=500, detail=f'启动失败：{exception}') from exception
        return {'ok': True}

    @app.get('/api/events/{event_id}/image/{image_index}')
    def event_image(event_id: str, image_index: int) -> FileResponse:
        with open_store() as catalog:
            event = catalog.get_event(event_id)
            if not event:
                raise HTTPException(status_code=404, detail='事件不存在')
            images = event.get('representatives') or []
            if image_index < 0 or image_index >= len(images):
                raise HTTPException(status_code=404, detail='代表帧不存在')
            path = Path(images[image_index]['image_path'])
            if not workspace.is_source_media(path):
                raise HTTPException(status_code=403, detail='不是采集目录中的图片')
            return FileResponse(path)

    return app


app = create_app()


def _browser_after_start(host: str, port: int) -> None:
    if host not in {'127.0.0.1', 'localhost'}:
        return
    if os.environ.get('HYAKKI_WORKBENCH_NO_BROWSER'):
        return
    threading.Timer(0.8, lambda: webbrowser.open(f'http://{host}:{port}')).start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not args.no_browser:
        _browser_after_start(args.host, args.port)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
