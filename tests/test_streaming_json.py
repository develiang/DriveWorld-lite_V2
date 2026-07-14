import hashlib
import json

from driveworld.data.nuscenes_tables import NuScenesTables, iter_json_array
from scripts.cache_static_maps import build_manifest_index, iter_manifest_range


def test_iter_json_array_across_small_chunks(tmp_path):
    values = [{"token": str(index), "value": "x" * (index + 1)} for index in range(20)]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    assert list(iter_json_array(path, chunk_size=17)) == values


def test_manifest_index_reads_only_requested_nonempty_records(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_bytes(b'{"index":0}\n\n{"index":1}\n{"index":2}\n')
    offsets = tmp_path / "train.offsets.npy"

    digest, count = build_manifest_index(manifest, offsets)

    assert digest == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert count == 3
    assert list(iter_manifest_range(manifest, offsets, 1, 3)) == [
        (1, {"index": 1}),
        (2, {"index": 2}),
    ]


def test_nuscenes_tables_retains_only_keyframe_lidar_and_camera_ego_poses(tmp_path):
    table_root = tmp_path / "v1.0-test"
    table_root.mkdir()
    tables = {
        "scene": [
            {
                "token": "scene-token",
                "name": "scene-0001",
                "first_sample_token": "sample-token",
                "log_token": "log-token",
            }
        ],
        "log": [{"token": "log-token", "location": "test-map"}],
        "sample": [{"token": "sample-token"}],
        "sensor": [
            {"token": "camera-sensor", "channel": "CAM_FRONT"},
            {"token": "lidar-sensor", "channel": "LIDAR_TOP"},
        ],
        "calibrated_sensor": [
            {
                "token": "camera-calibration",
                "sensor_token": "camera-sensor",
                "rotation": [1, 0, 0, 0],
                "translation": [0, 0, 0],
                "camera_intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "token": "lidar-calibration",
                "sensor_token": "lidar-sensor",
                "rotation": [1, 0, 0, 0],
                "translation": [0, 0, 0],
            },
        ],
        "sample_data": [
            {
                "token": "camera-data",
                "sample_token": "sample-token",
                "ego_pose_token": "camera-pose",
                "calibrated_sensor_token": "camera-calibration",
                "timestamp": 1,
                "filename": "camera.jpg",
                "is_key_frame": True,
                "prev": "",
                "next": "",
            },
            {
                "token": "lidar-keyframe",
                "sample_token": "sample-token",
                "ego_pose_token": "lidar-keyframe-pose",
                "calibrated_sensor_token": "lidar-calibration",
                "timestamp": 1,
                "filename": "lidar.bin",
                "is_key_frame": True,
                "prev": "",
                "next": "lidar-sweep",
            },
            {
                "token": "lidar-sweep",
                "sample_token": "sample-token",
                "ego_pose_token": "lidar-sweep-pose",
                "calibrated_sensor_token": "lidar-calibration",
                "timestamp": 2,
                "filename": "lidar-sweep.bin",
                "is_key_frame": False,
                "prev": "lidar-keyframe",
                "next": "",
            },
        ],
        "ego_pose": [
            {
                "token": token,
                "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
            }
            for token in ["camera-pose", "lidar-keyframe-pose", "lidar-sweep-pose"]
        ],
    }
    for name, records in tables.items():
        (table_root / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")

    loaded = NuScenesTables(tmp_path, "v1.0-test")

    assert set(loaded.sample_data) == {"camera-data", "lidar-keyframe"}
    assert set(loaded.ego_poses) == {"camera-pose"}
    camera_records = loaded.camera_records("scene-0001", "CAM_FRONT")
    assert [record["token"] for record in camera_records] == ["camera-data"]
    assert loaded.magicdrive_camera_parameter(camera_records[0]).shape == (3, 7)
