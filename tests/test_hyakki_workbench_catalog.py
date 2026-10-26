import json
import tempfile
import unittest
from pathlib import Path

from dev_tools.hyakki_workbench import WorkspacePaths, WorkbenchStore
from dev_tools.hyakki_workbench.dataset import DatasetBuilder
from dev_tools.hyakki_workbench.deployment import ModelPublisher
from dev_tools.hyakki_workbench.training_manager import TrainingManager
from dev_tools.hyakki_workbench.training_worker import match_predictions, read_ground_truth
from PIL import Image


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (320, 180), color=(20, 30, 40)).save(path)


class WorkbenchCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def paths(self, bean_root: Path, boss_root: Path) -> WorkspacePaths:
        return WorkspacePaths(
            project_root=self.root,
            bean_capture=bean_root,
            boss_capture=boss_root,
            state_root=self.root / 'state',
            runs_root=self.root / 'runs',
            weights_root=self.root / 'weights',
        )

    def test_catalog_groups_bean_tracks_and_boss_candidates(self) -> None:
        bean_root = self.root / 'bean_capture'
        boss_root = self.root / 'boss_capture'
        round_root = bean_root / '2026-10-26_12-00-00_r001'
        (round_root / 'raw').mkdir(parents=True)
        (round_root / 'raw' / 'frame_000001.jpg').write_bytes(b'frame-1')
        (round_root / 'raw' / 'frame_000002.jpg').write_bytes(b'frame-2')
        rows = [
            {
                'frame': 'frame_000001',
                'captured_at': '2026-10-26 12:00:01',
                'phase': 'running',
                'tracks': [[7, 265, 0.88, 640, 360, 80, 120, 0.1]],
            },
            {
                'frame': 'frame_000002',
                'captured_at': '2026-10-26 12:00:02',
                'phase': 'running',
                'tracks': [[7, 265, 0.72, 660, 360, 82, 122, 0.1]],
            },
        ]
        (round_root / 'frames.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in rows) + '\n',
            encoding='utf-8',
        )

        boss_round = boss_root / '2026-10-26_12-01-00_001'
        boss_round.mkdir(parents=True)
        (boss_round / 'candidate_1.jpg').write_bytes(b'boss-1')
        (boss_round / 'candidate_2.jpg').write_bytes(b'boss-2')
        write_json(
            boss_round / 'meta.json',
            {
                'status': 'complete',
                'entries': [
                    {'index': 1, 'old_class_id': 222, 'old_name': '石长姬'},
                    {'index': 2, 'old_class_id': None},
                ],
            },
        )

        store = WorkbenchStore.open(self.paths(bean_root, boss_root))
        try:
            result = store.scan()
            self.assertEqual(result.bean_rounds, 1)
            self.assertEqual(result.boss_rounds, 1)
            self.assertEqual(result.events_upserted, 3)

            bean = store.get_event('bean:2026-10-26_12-00-00_r001:7')
            self.assertIsNotNone(bean)
            self.assertEqual(bean['old_label'], 'ssr_075')
            self.assertEqual(bean['frame_count'], 2)
            self.assertEqual(bean['confidence'], 0.88)
            self.assertIn('high_rarity', bean['suspicion'])

            boss = store.get_event('boss:2026-10-26_12-01-00_001:1')
            self.assertIsNotNone(boss)
            self.assertEqual(boss['rarity'], 'SSR')
            self.assertTrue(boss['image_url'].endswith('/image/0'))

            summary = store.summary()
            self.assertEqual(summary['events'], 3)
            self.assertEqual(summary['pending_high_rarity'], 2)
            self.assertEqual(summary['rounds'], {'bean': 1, 'boss': 1})
            capture = store.capture_summary()
            self.assertEqual(capture['bean']['rounds'], 1)
            self.assertEqual(capture['boss']['rounds'], 1)
            self.assertGreater(capture['total_bytes'], 0)

            reviewed = store.review_event('bean:2026-10-26_12-00-00_r001:7', action='correct')
            self.assertEqual(reviewed['status'], 'correct')
            self.assertEqual(reviewed['reviewed_label'], 'ssr_075')
            self.assertEqual(store.review_history(reviewed['id'])[0]['action'], 'correct')

            candidates = store.list_classes(query='百羽凤凰火')
            self.assertEqual(candidates[0]['label'], 'sp_051')
            changed = store.review_event(
                'boss:2026-10-26_12-01-00_001:2',
                action='change',
                class_id=candidates[0]['class_id'],
                note='旧模型未识别',
            )
            self.assertEqual(changed['status'], 'changed')
            self.assertEqual(changed['reviewed_label'], 'sp_051')

            package = store.prepare_label_package(reviewed['id'])
            self.assertTrue((Path(package['folder']) / 'classes.txt').is_file())
            self.assertEqual(len(package['files']), 2)
            self.assertTrue(Path(package['files'][0]['annotation']).is_file())

            boss_package = store.prepare_label_package(changed['id'])
            annotation_path = Path(boss_package['files'][0]['annotation'])
            annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
            annotation['shapes'] = [{
                'label': 'sp_051',
                'points': [[20, 30], [120, 150]],
                'shape_type': 'rectangle',
            }]
            annotation_path.write_text(json.dumps(annotation), encoding='utf-8')
            imported = store.import_label_package(changed['id'])
            self.assertEqual(imported['imported'], 1)
            self.assertEqual(imported['event']['reviewed_label'], 'sp_051')
            self.assertEqual(store.frame_labels(changed['id'])[0]['box']['x1'], 20.0)
        finally:
            store.close()

    def test_rescan_preserves_review_state(self) -> None:
        bean_root = self.root / 'bean_capture'
        round_root = bean_root / 'round'
        (round_root / 'raw').mkdir(parents=True)
        (round_root / 'raw' / 'frame_000001.jpg').write_bytes(b'frame')
        row = {
            'frame': 'frame_000001',
            'tracks': [[1, 265, 0.9, 640, 360, 80, 120, 0]],
        }
        (round_root / 'frames.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
        store = WorkbenchStore.open(self.paths(bean_root, self.root / 'boss_capture'))
        try:
            store.scan()
            store.connection.execute(
                "UPDATE events SET status='correct', reviewed_class_id=265, "
                "reviewed_label='ssr_075', reviewed_at='2026-10-26 12:00:00'"
            )
            store.connection.commit()
            store.scan()
            event = store.get_event('bean:round:1')
            self.assertEqual(event['status'], 'correct')
            self.assertEqual(event['reviewed_class_id'], 265)
        finally:
            store.close()

    def test_dataset_build_uses_reviewed_events_and_round_split(self) -> None:
        bean_root = self.root / 'bean_capture'
        event_ids = []
        for round_name in ('round_a', 'round_b'):
            round_root = bean_root / round_name
            write_image(round_root / 'raw' / 'frame_000001.jpg')
            tracks = [
                [track_id, 265, 0.92, 40 + track_id * 24, 90, 18, 40, 0]
                for track_id in range(1, 11)
            ]
            row = {
                'frame': 'frame_000001',
                'tracks': tracks,
            }
            (round_root / 'frames.jsonl').write_text(
                json.dumps(row) + '\n',
                encoding='utf-8',
            )
            event_ids.extend(f'bean:{round_name}:{track_id}' for track_id in range(1, 11))
        store = WorkbenchStore.open(self.paths(bean_root, self.root / 'boss_capture'))
        try:
            store.scan()
            for event_id in event_ids:
                store.review_event(event_id, action='correct')
            readiness = DatasetBuilder(store).readiness()
            self.assertEqual(readiness['round_groups'], 2)
            self.assertEqual(readiness['reviewed_events'], 20)
            self.assertEqual(readiness['classes'][0]['label'], 'ssr_075')

            result = DatasetBuilder(store).build([265])
            target = Path(result['path'])
            self.assertTrue((target / 'data.yaml').is_file())
            self.assertTrue((target / 'manifest.json').is_file())
            self.assertEqual(result['manifest']['stats']['positive_images'], 20)
            self.assertEqual(set(result['manifest']['stats']['per_split']), {'train', 'val'})
            self.assertEqual(len(DatasetBuilder(store).list_datasets()), 1)
            data_yaml = (target / 'data.yaml').read_text(encoding='utf-8')
            self.assertNotIn('test:', data_yaml)
        finally:
            store.close()

    def test_training_manager_starts_empty_and_persists_job_schema(self) -> None:
        paths = self.paths(self.root / 'bean_capture', self.root / 'boss_capture')
        manager = TrainingManager(paths)
        payload = manager.list_jobs()
        self.assertEqual(payload['items'], [])
        self.assertEqual(payload['active'], 0)
        with self.assertRaises(KeyError):
            manager.get_job('missing')

    def test_threshold_matching_counts_tp_fp_and_fn(self) -> None:
        predictions = [
            [0, 0.90, 0.1, 0.1, 0.5, 0.5],
            [0, 0.80, 0.6, 0.6, 0.9, 0.9],
        ]
        truths = [[0, 0.1, 0.1, 0.5, 0.5]]
        tp, fp, fn = match_predictions(predictions, truths, 0.5)
        self.assertEqual((tp, fp, fn), (1, 1, 0))
        tp, fp, fn = match_predictions(predictions, truths, 0.95)
        self.assertEqual((tp, fp, fn), (0, 0, 1))

    def test_yolo_ground_truth_path_is_read_from_sibling_labels_dir(self) -> None:
        image = self.root / 'dataset' / 'images' / 'val' / 'sample.jpg'
        write_image(image)
        label = self.root / 'dataset' / 'labels' / 'val' / 'sample.txt'
        label.parent.mkdir(parents=True)
        label.write_text('0 0.5 0.5 0.2 0.4\n', encoding='utf-8')
        truth = read_ground_truth(image, 1)
        self.assertEqual(truth[0][0], 0)
        for actual, expected in zip(truth[0][1:], [0.4, 0.3, 0.6, 0.7]):
            self.assertAlmostEqual(actual, expected)

    def test_model_publisher_activation_round_trip(self) -> None:
        paths = self.paths(self.root / 'bean_capture', self.root / 'boss_capture')
        version = paths.weights_root / 'versions' / 'model-test'
        version.mkdir(parents=True)
        (version / 'manifest.json').write_text(
            json.dumps({
                'model_version': 'model-test',
                'service_url': 'http://127.0.0.1:8899',
                'classes': [],
            }),
            encoding='utf-8',
        )
        publisher = ModelPublisher(paths)
        active = publisher.set_active('model-test', 'assist')
        self.assertEqual(active['mode'], 'assist')
        self.assertEqual(publisher.active()['model_version'], 'model-test')
        self.assertTrue(publisher.list_models()[0]['active'])

    def test_publish_gate_requires_every_class_precision_and_recall(self) -> None:
        run_dir = self.root / 'train'
        run_dir.mkdir()
        (run_dir / 'result.json').write_text(
            json.dumps({'best_weight': str(self.root / 'best.pt')}),
            encoding='utf-8',
        )
        train_job = {'run_dir': str(run_dir), 'result_json': '{}'}
        test_job = {
            'result_json': json.dumps({
                'test': {
                    'classes': [
                        {'class_index': 0, 'precision': .96, 'recall': .91, 'tp': 9, 'fp': 0, 'fn': 1},
                        {'class_index': 1, 'precision': .90, 'recall': .95, 'tp': 9, 'fp': 1, 'fn': 0},
                    ],
                },
            }),
        }
        publisher = ModelPublisher(self.paths(
            self.root / 'bean_capture',
            self.root / 'boss_capture',
        ))
        report = publisher._gate_report(train_job, test_job)
        self.assertFalse(report['passed'])
        self.assertTrue(any('类别 1 Precision' in item for item in report['failures']))


if __name__ == '__main__':
    unittest.main()
