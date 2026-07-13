import json

from driveworld.data.nuscenes_tables import iter_json_array


def test_iter_json_array_across_small_chunks(tmp_path):
    values = [{"token": str(index), "value": "x" * (index + 1)} for index in range(20)]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    assert list(iter_json_array(path, chunk_size=17)) == values
