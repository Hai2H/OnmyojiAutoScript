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
from .deployment import InferenceServiceManager, ModelPublisher
from .training_manager import TrainingManager


class ReviewRequest(BaseModel):
    action: str = Field(pattern='^(correct|change|unknown|needs_label|skip|reopen)$')
    class_id: int | None = None
    note: str = ''


class XAnyConfigRequest(BaseModel):
    executable: str = ''


class DatasetBuildRequest(BaseModel):
    class_ids: list[int] | None = None


class TrainingStartRequest(BaseModel):
    dataset_version: str
    model: str = 'yolo11s.pt'
    epochs: int = 100
    imgsz: int = 960
    batch: int = 4
    patience: int = 20


class TrainingTestRequest(BaseModel):
    imgsz: int = 960
    batch: int = 4


class PublishRequest(BaseModel):
    mode: str = 'shadow'


class ActivateModelRequest(BaseModel):
    mode: str = 'shadow'


class RawFrameStatusRequest(BaseModel):
    action: str = Field(pattern='^(candidate|no_action|not_new|skip|reset)$')
    note: str = ''


def create_app(paths: WorkspacePaths | None = None) -> FastAPI:
    workspace = paths or WorkspacePaths()
    workspace.ensure()
    scan_lock = threading.Lock()
    training = TrainingManager(workspace)
    publisher = ModelPublisher(workspace)
    inference = InferenceServiceManager(workspace)

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

    @app.get('/api/capture')
    def capture() -> dict:
        with open_store() as catalog:
            return catalog.capture_summary()

    @app.get('/api/raw-frames')
    def raw_frames(
        round_id: str | None = None,
        filter_name: str = Query(default='all', alias='filter'),
        limit: int = Query(default=48, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        with open_store() as catalog:
            return catalog.list_raw_frames(
                round_id=round_id,
                filter_name=filter_name,
                limit=limit,
                offset=offset,
            )

    @app.get('/api/raw-frames/rounds')
    def raw_frame_rounds() -> dict:
        with open_store() as catalog:
            return {'items': catalog.raw_frame_rounds()}

    @app.get('/api/raw-frames/{frame_id}')
    def raw_frame_detail(frame_id: str) -> dict:
        with open_store() as catalog:
            frame = catalog.get_raw_frame(frame_id)
            if not frame:
                raise HTTPException(status_code=404, detail='原始帧不存在')
            return frame

    @app.get('/api/raw-frames/{frame_id}/image')
    def raw_frame_image(frame_id: str) -> FileResponse:
        with open_store() as catalog:
            frame = catalog.get_raw_frame(frame_id)
            if not frame:
                raise HTTPException(status_code=404, detail='原始帧不存在')
            path = Path(frame['image_path'])
            if not workspace.is_source_media(path):
                raise HTTPException(status_code=403, detail='不是采集目录中的图片')
            return FileResponse(path)

    @app.post('/api/raw-frames/{frame_id}/status')
    def raw_frame_status(frame_id: str, request: RawFrameStatusRequest) -> dict:
        with open_store() as catalog:
            try:
                return catalog.mark_raw_frame(frame_id, request.action, request.note)
            except KeyError as exception:
                raise HTTPException(status_code=404, detail='原始帧不存在') from exception
            except ValueError as exception:
                raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.post('/api/raw-frames/{frame_id}/prepare-label')
    def prepare_raw_frame_label(frame_id: str) -> dict:
        with open_store() as catalog:
            try:
                return catalog.prepare_raw_label_package(frame_id)
            except KeyError as exception:
                raise HTTPException(status_code=404, detail='原始帧不存在') from exception

    @app.post('/api/raw-frames/{frame_id}/open-label-folder')
    def open_raw_frame_label_folder(frame_id: str) -> dict:
        folder = workspace.state_root / 'scan_review' / frame_id.replace(':', '_').replace('/', '_')
        if not folder.is_dir():
            raise HTTPException(status_code=422, detail='请先生成标注目录')
        try:
            if os.name == 'nt':
                os.startfile(str(folder))
            else:
                subprocess.Popen(['xdg-open', str(folder)])
        except OSError as exception:
            raise HTTPException(status_code=500, detail=f'打开目录失败：{exception}') from exception
        return {'ok': True, 'folder': str(folder)}

    @app.post('/api/raw-frames/{frame_id}/import-label')
    def import_raw_frame_label(frame_id: str) -> dict:
        with open_store() as catalog:
            try:
                return catalog.import_raw_label_package(frame_id)
            except KeyError as exception:
                raise HTTPException(status_code=404, detail='原始帧不存在') from exception
            except ValueError as exception:
                raise HTTPException(status_code=422, detail=str(exception)) from exception

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

    @app.get('/api/training')
    def training_jobs(
        limit: int = Query(default=20, ge=1, le=100),
        log_tail: int = Query(default=0, ge=0, le=200000),
    ) -> dict:
        return training.list_jobs(limit=limit, log_tail=log_tail)

    @app.post('/api/training/start')
    def start_training(request: TrainingStartRequest) -> dict:
        try:
            return training.start_training(
                dataset_version=request.dataset_version,
                model=request.model,
                epochs=request.epochs,
                imgsz=request.imgsz,
                batch=request.batch,
                patience=request.patience,
            )
        except ValueError as exception:
            raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.get('/api/training/{job_id}')
    def training_job(
        job_id: str,
        log_tail: int = Query(default=60000, ge=0, le=200000),
    ) -> dict:
        try:
            return training.get_job(job_id, log_tail=log_tail)
        except KeyError as exception:
            raise HTTPException(status_code=404, detail='任务不存在') from exception

    @app.post('/api/training/{job_id}/test')
    def test_training(
        job_id: str,
        request: TrainingTestRequest,
    ) -> dict:
        try:
            return training.start_test(job_id, imgsz=request.imgsz, batch=request.batch)
        except KeyError as exception:
            raise HTTPException(status_code=404, detail='训练任务不存在') from exception
        except ValueError as exception:
            raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.post('/api/training/{job_id}/cancel')
    def cancel_training(job_id: str) -> dict:
        try:
            return training.cancel(job_id)
        except KeyError as exception:
            raise HTTPException(status_code=404, detail='训练任务不存在') from exception
        except ValueError as exception:
            raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.get('/api/deployment')
    def deployment() -> dict:
        return {
            'models': publisher.list_models(),
            'active': publisher.active(),
            'service': inference.health(),
            'gates': {'precision': .95, 'recall': .90, 'latency_ms_p95': 180},
        }

    @app.get('/api/models/{train_job_id}/candidate')
    def model_candidate(train_job_id: str) -> dict:
        try:
            return publisher.inspect_candidate(train_job_id)
        except KeyError as exception:
            raise HTTPException(status_code=404, detail='训练任务不存在') from exception

    @app.post('/api/models/{train_job_id}/publish')
    def publish_model(train_job_id: str, request: PublishRequest) -> dict:
        try:
            return publisher.publish(train_job_id, mode=request.mode)
        except KeyError as exception:
            raise HTTPException(status_code=404, detail='训练任务不存在') from exception
        except ValueError as exception:
            raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.post('/api/models/{version}/activate')
    def activate_model(version: str, request: ActivateModelRequest) -> dict:
        try:
            return publisher.set_active(version, request.mode)
        except ValueError as exception:
            raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.post('/api/inference/start')
    def start_inference() -> dict:
        try:
            return inference.start()
        except ValueError as exception:
            raise HTTPException(status_code=422, detail=str(exception)) from exception

    @app.post('/api/inference/stop')
    def stop_inference() -> dict:
        return inference.stop()

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
