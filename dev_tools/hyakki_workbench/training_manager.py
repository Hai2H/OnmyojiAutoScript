"""Persistent single-GPU training and test job management."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import WorkspacePaths
from .store import WorkbenchStore


JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    parent_job_id TEXT,
    dataset_version TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    pid INTEGER,
    log_path TEXT NOT NULL,
    run_dir TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_status
    ON training_jobs(status, created_at);
"""


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _job_payload(row, include_log: str = '') -> dict:
    payload = dict(row)
    try:
        payload['config'] = json.loads(payload.pop('config_json') or '{}')
    except json.JSONDecodeError:
        payload['config'] = {}
    try:
        payload['result'] = json.loads(payload.pop('result_json') or 'null')
    except json.JSONDecodeError:
        payload['result'] = None
    if include_log:
        log_path = Path(payload['log_path'])
        try:
            text = log_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            text = ''
        payload['log'] = text[-max(1000, min(int(include_log), 200000)):]
    return payload


class TrainingManager:
    def __init__(self, paths: WorkspacePaths):
        self.paths = paths
        self.paths.ensure()
        self._lock = threading.Lock()
        self._monitors: dict[str, threading.Thread] = {}
        self._init_database()
        self._recover_jobs()

    def _init_database(self) -> None:
        with WorkbenchStore.open(self.paths) as catalog:
            catalog.connection.executescript(JOB_SCHEMA)
            catalog.connection.commit()

    def _recover_jobs(self) -> None:
        with WorkbenchStore.open(self.paths) as catalog:
            rows = catalog.connection.execute(
                "SELECT * FROM training_jobs WHERE status IN ('queued', 'running')"
            ).fetchall()
        for row in rows:
            run_dir = Path(row['run_dir'])
            result_path = run_dir / 'result.json'
            if result_path.is_file():
                self._finish_job(row['id'], result_path)
                continue
            pid = row['pid']
            if pid and self._pid_alive(int(pid)):
                thread = threading.Thread(
                    target=self._monitor_pid,
                    args=(row['id'], int(pid), run_dir / 'result.json'),
                    daemon=True,
                )
                self._monitors[row['id']] = thread
                thread.start()
            else:
                self._set_failed(row['id'], '工作台重启时训练进程已不存在')

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            import psutil

            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    def _active_count(self, catalog: WorkbenchStore) -> int:
        return catalog.connection.execute(
            "SELECT COUNT(*) FROM training_jobs WHERE status IN ('queued', 'running')"
        ).fetchone()[0]

    def gpu_status(self) -> dict:
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=name,memory.total,memory.used,driver_version',
                    '--format=csv,noheader,nounits',
                ],
                cwd=self.paths.project_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().splitlines()[0]
                name, total, used, driver = [item.strip() for item in line.split(',')]
                return {
                    'available': True,
                    'name': name,
                    'memory_total_mb': int(total),
                    'memory_used_mb': int(used),
                    'driver': driver,
                }
        except Exception:
            pass
        return {'available': False}

    def start_training(
        self,
        *,
        dataset_version: str,
        model: str = 'yolo11s.pt',
        epochs: int = 100,
        imgsz: int = 960,
        batch: int = 4,
        patience: int = 20,
        workers: int = 0,
    ) -> dict:
        if model not in {'yolo11n.pt', 'yolo11s.pt'}:
            raise ValueError('V1 只允许 yolo11n.pt 或 yolo11s.pt')
        if not 1 <= int(epochs) <= 500:
            raise ValueError('epochs 必须在 1～500')
        if not 320 <= int(imgsz) <= 1280:
            raise ValueError('imgsz 必须在 320～1280')
        if not 1 <= int(batch) <= 32:
            raise ValueError('batch 必须在 1～32')
        if not 0 <= int(workers) <= 8:
            raise ValueError('workers 必须在 0～8')
        python = self.paths.train_python
        if not python.is_file():
            raise ValueError(f'训练环境不存在：{python}')

        with self._lock:
            with WorkbenchStore.open(self.paths) as catalog:
                if self._active_count(catalog):
                    raise ValueError('已有训练或测试任务正在运行')
                dataset = catalog.connection.execute(
                    'SELECT * FROM datasets WHERE version = ?',
                    (dataset_version,),
                ).fetchone()
                if not dataset:
                    raise ValueError('数据集版本不存在')
                data_yaml = Path(dataset['path']) / 'data.yaml'
                if not data_yaml.is_file():
                    raise ValueError('数据集缺少 data.yaml')
                job_id = self._job_id('train')
                run_dir = self.paths.runs_root / 'jobs' / job_id
                run_dir.mkdir(parents=True, exist_ok=False)
                log_path = run_dir / 'job.log'
                config = {
                    'model': model,
                    'epochs': int(epochs),
                    'imgsz': int(imgsz),
                    'batch': int(batch),
                    'patience': int(patience),
                    'workers': int(workers),
                    'device': '0',
                    'amp': True,
                }
                created = _now()
                catalog.connection.execute(
                    """
                    INSERT INTO training_jobs (
                        id, kind, dataset_version, dataset_path, model,
                        config_json, status, log_path, run_dir, created_at
                    ) VALUES (?, 'train', ?, ?, ?, ?, 'queued', ?, ?, ?)
                    """,
                    (
                        job_id,
                        dataset_version,
                        dataset['path'],
                        model,
                        json.dumps(config, ensure_ascii=False),
                        str(log_path),
                        str(run_dir),
                        created,
                    ),
                )
                catalog.connection.commit()

            command = [
                str(python),
                '-m',
                'dev_tools.hyakki_workbench.training_worker',
                'train',
                '--job-id', job_id,
                '--data', str(data_yaml),
                '--run-dir', str(run_dir),
                '--model', model,
                '--epochs', str(epochs),
                '--imgsz', str(imgsz),
                '--batch', str(batch),
                '--patience', str(patience),
                '--workers', str(workers),
            ]
            self._launch(job_id, command, run_dir)
        return self.get_job(job_id)

    def start_test(self, parent_job_id: str, *, imgsz: int = 960, batch: int = 4) -> dict:
        parent = self.get_job(parent_job_id)
        if parent['kind'] != 'train' or parent['status'] != 'completed':
            raise ValueError('只有已完成的训练任务才能进入最终测试')
        result = parent.get('result') or {}
        best = result.get('best_weight')
        if not best or not Path(best).is_file():
            raise ValueError('训练结果中没有可用 best.pt')
        data_yaml = Path(parent['dataset_path']) / 'data.yaml'
        if 'test:' not in data_yaml.read_text(encoding='utf-8'):
            raise ValueError('当前数据集没有独立 test 集')
        thresholds_path = Path(parent['run_dir']) / 'thresholds.json'
        if not thresholds_path.is_file():
            raise ValueError('当前训练结果缺少按类别校准阈值')
        python = self.paths.train_python

        with self._lock:
            with WorkbenchStore.open(self.paths) as catalog:
                if self._active_count(catalog):
                    raise ValueError('已有训练或测试任务正在运行')
                job_id = self._job_id('test')
                run_dir = self.paths.runs_root / 'jobs' / job_id
                run_dir.mkdir(parents=True, exist_ok=False)
                log_path = run_dir / 'job.log'
                config = {
                    'imgsz': int(imgsz),
                    'batch': int(batch),
                    'device': '0',
                    'thresholds': str(thresholds_path),
                }
                catalog.connection.execute(
                    """
                    INSERT INTO training_jobs (
                        id, kind, parent_job_id, dataset_version, dataset_path,
                        model, config_json, status, log_path, run_dir, created_at
                    ) VALUES (?, 'test', ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                    """,
                    (
                        job_id,
                        parent_job_id,
                        parent['dataset_version'],
                        parent['dataset_path'],
                        parent['model'],
                        json.dumps(config, ensure_ascii=False),
                        str(log_path),
                        str(run_dir),
                        _now(),
                    ),
                )
                catalog.connection.commit()
            command = [
                str(python),
                '-m',
                'dev_tools.hyakki_workbench.training_worker',
                'test',
                '--job-id', job_id,
                '--data', str(data_yaml),
                '--run-dir', str(run_dir),
                '--weight', str(best),
                '--thresholds', str(thresholds_path),
                '--imgsz', str(imgsz),
                '--batch', str(batch),
            ]
            self._launch(job_id, command, run_dir)
        return self.get_job(job_id)

    def _launch(self, job_id: str, command: list[str], run_dir: Path) -> None:
        log_path = run_dir / 'job.log'
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        log_handle = log_path.open('a', encoding='utf-8')
        try:
            process = subprocess.Popen(
                command,
                cwd=self.paths.project_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        except Exception as exception:
            log_handle.close()
            self._set_failed(job_id, f'启动失败：{exception}')
            raise ValueError(f'启动训练失败：{exception}') from exception
        with WorkbenchStore.open(self.paths) as catalog:
            catalog.connection.execute(
                "UPDATE training_jobs SET status='running', pid=?, started_at=? WHERE id=?",
                (process.pid, _now(), job_id),
            )
            catalog.connection.commit()
        thread = threading.Thread(
            target=self._monitor,
            args=(job_id, process, log_handle, run_dir / 'result.json'),
            daemon=True,
        )
        self._monitors[job_id] = thread
        thread.start()

    def _monitor(self, job_id: str, process, log_handle, result_path: Path) -> None:
        try:
            returncode = process.wait()
        finally:
            log_handle.close()
            self._monitors.pop(job_id, None)
        if result_path.is_file():
            self._finish_job(job_id, result_path, returncode=returncode)
        else:
            self._set_failed(job_id, f'训练进程退出码 {returncode}，未生成 result.json')

    def _monitor_pid(self, job_id: str, pid: int, result_path: Path) -> None:
        import time

        while self._pid_alive(pid):
            time.sleep(2)
        self._monitors.pop(job_id, None)
        if result_path.is_file():
            self._finish_job(job_id, result_path)
        else:
            self._set_failed(job_id, '训练进程结束，未生成 result.json')

    def _finish_job(self, job_id: str, result_path: Path, returncode: int | None = None) -> None:
        try:
            result = json.loads(result_path.read_text(encoding='utf-8'))
        except Exception as exception:
            self._set_failed(job_id, f'读取训练结果失败：{exception}')
            return
        ok = bool(result.get('ok'))
        status = 'completed' if ok else 'failed'
        error = None if ok else result.get('error') or f'进程退出码 {returncode}'
        with WorkbenchStore.open(self.paths) as catalog:
            catalog.connection.execute(
                """
                UPDATE training_jobs SET
                    status=?, result_json=?, error=?, finished_at=?
                WHERE id=?
                """,
                (status, json.dumps(result, ensure_ascii=False), error, _now(), job_id),
            )
            catalog.connection.commit()

    def _set_failed(self, job_id: str, error: str) -> None:
        with WorkbenchStore.open(self.paths) as catalog:
            catalog.connection.execute(
                "UPDATE training_jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                (error, _now(), job_id),
            )
            catalog.connection.commit()

    def cancel(self, job_id: str) -> dict:
        with WorkbenchStore.open(self.paths) as catalog:
            row = catalog.connection.execute(
                'SELECT * FROM training_jobs WHERE id=?', (job_id,)
            ).fetchone()
            if not row:
                raise KeyError(job_id)
            if row['status'] not in {'queued', 'running'}:
                raise ValueError('该任务已经结束')
            pid = row['pid']
            if pid:
                try:
                    import psutil

                    process = psutil.Process(int(pid))
                    children = process.children(recursive=True)
                    for child in children:
                        child.terminate()
                    process.terminate()
                    psutil.wait_procs([*children, process], timeout=5)
                except Exception:
                    pass
            catalog.connection.execute(
                "UPDATE training_jobs SET status='cancelled', finished_at=?, error=? WHERE id=?",
                (_now(), '用户取消', job_id),
            )
            catalog.connection.commit()
        return self.get_job(job_id)

    def _job_id(self, prefix: str) -> str:
        return f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

    def get_job(self, job_id: str, log_tail: int = 0) -> dict:
        with WorkbenchStore.open(self.paths) as catalog:
            row = catalog.connection.execute(
                'SELECT * FROM training_jobs WHERE id=?', (job_id,)
            ).fetchone()
        if not row:
            raise KeyError(job_id)
        return _job_payload(row, include_log=str(log_tail) if log_tail else '')

    def list_jobs(self, limit: int = 20, log_tail: int = 0) -> dict:
        with WorkbenchStore.open(self.paths) as catalog:
            rows = catalog.connection.execute(
                'SELECT * FROM training_jobs ORDER BY created_at DESC, id DESC LIMIT ?',
                (max(1, min(int(limit), 100)),),
            ).fetchall()
            active = self._active_count(catalog)
        return {
            'active': active,
            'gpu': self.gpu_status(),
            'items': [
                _job_payload(row, include_log=str(log_tail) if log_tail else '')
                for row in rows
            ],
        }
