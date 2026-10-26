"""FastAPI entry point for the local Hyakki dataset workbench."""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .paths import WorkspacePaths
from .store import WorkbenchStore


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
