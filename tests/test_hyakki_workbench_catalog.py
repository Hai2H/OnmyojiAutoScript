import json
import tempfile
import unittest
from pathlib import Path

from dev_tools.hyakki_workbench import WorkspacePaths, WorkbenchStore


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')


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


if __name__ == '__main__':
    unittest.main()
