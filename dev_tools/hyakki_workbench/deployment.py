"""Model publishing, activation and local inference service lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from .paths import WorkspacePaths
from .store import WorkbenchStore


PRECISION_GATE = .95
RECALL_GATE = .90
LATENCY_GATE_MS = 180.0
VALID_MODES = {'old', 'shadow', 'assist', 'primary'}


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


class ModelPublisher:
    def __init__(self, paths: WorkspacePaths):
        self.paths = paths

    def _job(self, job_id: str):
        with WorkbenchStore.open(self.paths) as catalog:
            return catalog.connection.execute(
                'SELECT * FROM training_jobs WHERE id=?', (job_id,)
            ).fetchone()

    def _test_for_train(self, job_id: str):
        with WorkbenchStore.open(self.paths) as catalog:
            return catalog.connection.execute(
                "SELECT * FROM training_jobs "
                "WHERE kind='test' AND parent_job_id=? AND status='completed' "
                'ORDER BY created_at DESC LIMIT 1',
                (job_id,),
            ).fetchone()

    def _gate_report(self, train_job, test_job) -> dict:
        train_result = _read_json(Path(train_job['run_dir']) / 'result.json')
        if not train_result:
            try:
                train_result = json.loads(train_job['result_json'] or '{}')
            except json.JSONDecodeError:
                train_result = {}
        try:
            test_result = json.loads(test_job['result_json'] or '{}')
        except json.JSONDecodeError:
            test_result = {}
        test = test_result.get('test') or {}
        classes = test.get('classes') or []
        failures = []
        if not classes:
            failures.append('最终测试没有逐类结果')
        latency_p95 = test.get('latency_ms_p95')
        if latency_p95 is None:
            failures.append('最终测试缺少 P95 延迟')
        elif float(latency_p95) > LATENCY_GATE_MS:
            failures.append(
                f'P95 延迟 {float(latency_p95):.1f}ms > {LATENCY_GATE_MS:.0f}ms'
            )
        for row in classes:
            support = int(row.get('tp', 0)) + int(row.get('fn', 0))
            if support <= 0:
                failures.append(f"类别 {row.get('class_index')} 没有独立 test 样本")
                continue
            if float(row.get('precision', 0)) < PRECISION_GATE:
                failures.append(
                    f"类别 {row.get('class_index')} Precision "
                    f"{float(row.get('precision', 0)):.3f} < {PRECISION_GATE:.2f}"
                )
            if float(row.get('recall', 0)) < RECALL_GATE:
                failures.append(
                    f"类别 {row.get('class_index')} Recall "
                    f"{float(row.get('recall', 0)):.3f} < {RECALL_GATE:.2f}"
                )
        return {
            'passed': not failures,
            'failures': failures,
            'test': test,
            'train': train_result,
        }

    def inspect_candidate(self, train_job_id: str) -> dict:
        train_job = self._job(train_job_id)
        if not train_job:
            raise KeyError(train_job_id)
        test_job = self._test_for_train(train_job_id)
        if not test_job:
            return {
                'train_job_id': train_job_id,
                'ready': False,
                'reason': '尚未完成独立 test',
            }
        gate = self._gate_report(train_job, test_job)
        return {
            'train_job_id': train_job_id,
            'test_job_id': test_job['id'],
            'ready': gate['passed'],
            'failures': gate['failures'],
            'test': gate['test'],
        }

    def publish(self, train_job_id: str, *, mode: str = 'shadow') -> dict:
        if mode not in VALID_MODES:
            raise ValueError('不支持的运行模式')
        train_job = self._job(train_job_id)
        if not train_job or train_job['kind'] != 'train' or train_job['status'] != 'completed':
            raise ValueError('训练任务尚未成功完成')
        test_job = self._test_for_train(train_job_id)
        if not test_job:
            raise ValueError('尚未完成最终 test')
        gate = self._gate_report(train_job, test_job)
        if not gate['passed']:
            raise ValueError('模型未通过上线验收：' + '；'.join(gate['failures']))
        result = gate['train']
        best = Path(result.get('best_weight') or '')
        if not best.is_file():
            raise ValueError('训练任务缺少 best.pt')
        thresholds = result.get('thresholds') or {}
        train_config = json.loads(train_job['config_json'] or '{}')
        imgsz = int(train_config.get('imgsz', 960))

        self.paths.weights_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        suffix = hashlib.sha256(
            json.dumps(gate['test'], sort_keys=True).encode()
            + best.read_bytes()
        ).hexdigest()[:8]
        version = f'model-{stamp}-{suffix}'
        temporary = self.paths.weights_root / f'.publishing-{uuid.uuid4().hex}'
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            export_command = [
                str(self.paths.train_python),
                '-m',
                'dev_tools.hyakki_workbench.training_worker',
                'export',
                '--job-id', f'export-{version}',
                '--run-dir', str(temporary),
                '--weight', str(best),
                '--imgsz', str(imgsz),
            ]
            exported = subprocess.run(
                export_command,
                cwd=self.paths.project_root,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            export_result = _read_json(temporary / 'result.json', {}) or {}
            if exported.returncode != 0 or not export_result.get('ok'):
                error = export_result.get('error') or exported.stderr[-2000:] or 'ONNX 导出失败'
                raise ValueError(error)
            onnx = Path(export_result['onnx'])
            target_onnx = temporary / 'model.onnx'
            if onnx != target_onnx:
                shutil.copy2(onnx, target_onnx)
            target_pt = temporary / 'model.pt'
            shutil.copy2(best, target_pt)
            (temporary / 'thresholds.json').write_text(
                json.dumps(thresholds, ensure_ascii=False, indent=1),
                encoding='utf-8',
            )
            with WorkbenchStore.open(self.paths) as catalog:
                dataset_row = catalog.connection.execute(
                    'SELECT manifest_json FROM datasets WHERE version=?',
                    (train_job['dataset_version'],),
                ).fetchone()
            dataset_manifest = json.loads(dataset_row['manifest_json']) if dataset_row else {}
            classes = dataset_manifest.get('classes') or []
            manifest = {
                'schema_version': 1,
                'model_version': version,
                'dataset_version': train_job['dataset_version'],
                'train_job_id': train_job_id,
                'test_job_id': test_job['id'],
                'created_at': _now(),
                'model_file': 'model.onnx',
                'pytorch_file': 'model.pt',
                'task': 'detect',
                'imgsz': imgsz,
                'classes': classes,
                'thresholds': thresholds,
                'metrics': {
                    'test': gate['test'],
                    'gates': {
                        'precision': PRECISION_GATE,
                        'recall': RECALL_GATE,
                        'latency_ms_p95': LATENCY_GATE_MS,
                    },
                },
                'service_url': 'http://127.0.0.1:8899',
            }
            (temporary / 'manifest.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1),
                encoding='utf-8',
            )
            target = self.paths.weights_root / 'versions' / version
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError('模型版本已存在')
            temporary.rename(target)
            self.set_active(version, mode)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {
            'version': version,
            'path': str(target),
            'manifest': manifest,
            'active': self.active(),
        }

    def list_models(self) -> list[dict]:
        root = self.paths.weights_root / 'versions'
        if not root.is_dir():
            return []
        active = self.active()
        rows = []
        for folder in sorted(root.iterdir(), reverse=True):
            manifest = _read_json(folder / 'manifest.json')
            if not isinstance(manifest, dict):
                continue
            rows.append({
                'version': manifest.get('model_version') or folder.name,
                'path': str(folder),
                'active': (manifest.get('model_version') or folder.name) == active.get('model_version'),
                'mode': active.get('mode') if (
                    manifest.get('model_version') or folder.name
                ) == active.get('model_version') else None,
                'dataset_version': manifest.get('dataset_version'),
                'classes': manifest.get('classes') or [],
                'metrics': manifest.get('metrics') or {},
                'created_at': manifest.get('created_at'),
            })
        return rows

    @property
    def active_path(self) -> Path:
        return self.paths.weights_root / 'active.json'

    def active(self) -> dict:
        return _read_json(self.active_path, {}) or {}

    def set_active(self, version: str, mode: str) -> dict:
        if mode not in VALID_MODES:
            raise ValueError('不支持的运行模式')
        folder = self.paths.weights_root / 'versions' / version
        manifest_path = folder / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError('模型版本不存在')
        manifest = _read_json(manifest_path, {}) or {}
        payload = {
            'model_version': version,
            'mode': mode,
            'service_url': manifest.get('service_url') or 'http://127.0.0.1:8899',
            'model_manifest': str(manifest_path.relative_to(self.paths.weights_root)),
            'updated_at': _now(),
        }
        temporary = self.active_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        temporary.replace(self.active_path)
        return payload


class InferenceServiceManager:
    def __init__(self, paths: WorkspacePaths):
        self.paths = paths
        self.pid_path = paths.state_root / 'inference.pid'
        self.log_path = paths.project_root / 'log' / 'workbench' / 'inference.log'
        self.publisher = ModelPublisher(paths)

    @property
    def python(self) -> Path:
        preferred = self.paths.project_root / '.venv-infer' / 'Scripts' / 'python.exe'
        return preferred if preferred.is_file() else self.paths.train_python

    def _pid(self) -> int | None:
        try:
            return int(self.pid_path.read_text(encoding='utf-8').strip())
        except (OSError, ValueError):
            return None

    def _alive(self, pid: int | None) -> bool:
        if not pid:
            return False
        try:
            import psutil

            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    def health(self) -> dict:
        active = self.publisher.active()
        endpoint = active.get('service_url') or 'http://127.0.0.1:8899'
        pid = self._pid()
        if self._alive(pid):
            try:
                with urllib.request.urlopen(f'{endpoint}/health', timeout=2) as response:
                    return {
                        'running': True,
                        'pid': pid,
                        'python': str(self.python),
                        **json.loads(response.read().decode('utf-8')),
                    }
            except Exception:
                pass
        return {
            'running': False,
            'pid': pid if self._alive(pid) else None,
            'python': str(self.python),
            'active': active,
        }

    def start(self) -> dict:
        active = self.publisher.active()
        if not active.get('model_manifest'):
            raise ValueError('尚未激活模型')
        if self.health().get('running'):
            return self.health()
        manifest = self.paths.weights_root / active['model_manifest']
        if not manifest.is_file():
            raise ValueError('激活模型清单不存在')
        if not self.python.is_file():
            raise ValueError(f'推理环境不存在：{self.python}')
        endpoint = active.get('service_url') or 'http://127.0.0.1:8899'
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        port = parsed.port or 8899
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.log_path.open('a', encoding='utf-8')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.Popen(
            [
                str(self.python),
                '-m',
                'dev_tools.hyakki_workbench.inference_service',
                '--manifest', str(manifest),
                '--host', parsed.hostname or '127.0.0.1',
                '--port', str(port),
            ],
            cwd=self.paths.project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        log_handle.close()
        self.pid_path.write_text(str(process.pid), encoding='utf-8')
        for _ in range(40):
            time.sleep(.25)
            health = self.health()
            if health.get('running'):
                return health
            if process.poll() is not None:
                break
        raise ValueError(f'推理服务启动失败，查看日志：{self.log_path}')

    def stop(self) -> dict:
        pid = self._pid()
        if self._alive(pid):
            try:
                import psutil

                process = psutil.Process(int(pid))
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                pass
        try:
            self.pid_path.unlink()
        except OSError:
            pass
        return self.health()
