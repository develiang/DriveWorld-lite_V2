from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


MAGICDRIVE_MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "road_divider",
    "lane_divider",
    "road_block",
)


class _NuScenesVectorMap:
    """Dependency-free reader for the polygon/line topology in map expansion JSON."""

    def __init__(self, path: Path, classes):
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = {
            record["token"]: np.asarray([record["x"], record["y"]], dtype=np.float64)
            for record in data["node"]
        }
        polygons = {record["token"]: record for record in data["polygon"]}
        lines = {record["token"]: record for record in data["line"]}
        self.features = {}
        for layer_name in classes:
            layer_features = []
            if layer_name in {"road_divider", "lane_divider"}:
                seen_geometry_tokens = set()
                for record in data[layer_name]:
                    geometry_token = record["line_token"]
                    if geometry_token in seen_geometry_tokens:
                        continue
                    seen_geometry_tokens.add(geometry_token)
                    line = lines[geometry_token]
                    coordinates = np.stack([nodes[token] for token in line["node_tokens"]])
                    layer_features.append(
                        ("line", coordinates, (), self._bounds(coordinates))
                    )
            else:
                polygon_tokens = []
                for record in data[layer_name]:
                    if layer_name == "drivable_area":
                        polygon_tokens.extend(record["polygon_tokens"])
                    else:
                        polygon_tokens.append(record["polygon_token"])
                seen_geometry_tokens = set()
                for token in polygon_tokens:
                    # Some v1.3 Singapore map layers contain hundreds of records
                    # pointing at the same placeholder polygon token (including
                    # ``None``). Masks are a union, so drawing identical geometry
                    # once is exactly equivalent and avoids pathological runtimes.
                    if token in seen_geometry_tokens:
                        continue
                    seen_geometry_tokens.add(token)
                    polygon = polygons[token]
                    exterior = np.stack(
                        [nodes[node] for node in polygon["exterior_node_tokens"]]
                    )
                    holes = tuple(
                        np.stack([nodes[node] for node in hole["node_tokens"]])
                        for hole in polygon["holes"]
                    )
                    layer_features.append(
                        ("polygon", exterior, holes, self._bounds(exterior))
                    )
            self.features[layer_name] = layer_features

    @staticmethod
    def _bounds(coordinates):
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        return (minimum[0], minimum[1], maximum[0], maximum[1])


class NuScenesStaticMapRenderer:
    """Render the exact eight-channel MagicDrive ego/lidar-centric BEV crop."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        xbound=(-50.0, 50.0, 0.5),
        ybound=(-50.0, 50.0, 0.5),
        classes=MAGICDRIVE_MAP_CLASSES,
    ):
        self.data_root = Path(data_root)
        self.xbound = tuple(float(value) for value in xbound)
        self.ybound = tuple(float(value) for value in ybound)
        self.classes = tuple(classes)
        if self.classes != MAGICDRIVE_MAP_CLASSES:
            raise ValueError(
                "Stage-3 requires the exact ordered MagicDrive eight map classes"
            )
        if len(self.xbound) != 3 or len(self.ybound) != 3:
            raise ValueError("xbound/ybound must be [min,max,resolution]")
        patch_h = self.ybound[1] - self.ybound[0]
        patch_w = self.xbound[1] - self.xbound[0]
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (
            round(patch_h / self.ybound[2]),
            round(patch_w / self.xbound[2]),
        )
        if self.canvas_size != (200, 200):
            raise ValueError(
                f"Stage-3 224/400 bucket expects a 200x200 BEV, got {self.canvas_size}"
            )
        self._maps = {}

    def _map(self, location: str):
        if location not in self._maps:
            expansion = self.data_root / "maps" / "expansion" / f"{location}.json"
            if not expansion.is_file():
                raise FileNotFoundError(
                    f"Missing nuScenes semantic map expansion: {expansion}. "
                    "The four maps/*.png files alone cannot produce MagicDrive's "
                    "eight semantic BEV channels."
                )
            self._maps[location] = _NuScenesVectorMap(expansion, self.classes)
        return self._maps[location]

    def render(self, location: str, map_pose) -> np.ndarray:
        pose = np.asarray(map_pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("map_pose must be finite [global_x,global_y,yaw_rad]")
        masks = np.stack(
            [
                self._geometry_mask(layer_name, self._map(location).features[layer_name], pose)
                for layer_name in self.classes
            ]
        )
        # Preserve MagicDrive LoadBEVSegmentation's lidar/canvas orientation.
        masks = masks.transpose(0, 2, 1)
        if masks.shape != (8, 200, 200):
            raise RuntimeError(f"Unexpected nuScenes map mask shape: {masks.shape}")
        return masks.astype(np.float32, copy=False)

    def _canvas_coordinates(self, coordinates, pose):
        values = np.asarray(coordinates, dtype=np.float64) - pose[None, :2]
        cosine, sine = math.cos(float(pose[2])), math.sin(float(pose[2]))
        values = np.column_stack(
            [
                cosine * values[:, 0] + sine * values[:, 1],
                -sine * values[:, 0] + cosine * values[:, 1],
            ]
        )
        values[:, 0] = (values[:, 0] + self.patch_size[1] / 2) * (
            self.canvas_size[1] / self.patch_size[1]
        )
        values[:, 1] = (values[:, 1] + self.patch_size[0] / 2) * (
            self.canvas_size[0] / self.patch_size[0]
        )
        return np.rint(values[:, :2]).astype(np.int32)

    def _geometry_mask(self, layer_name, features, pose):
        canvas = Image.new("L", (self.canvas_size[1], self.canvas_size[0]), 0)
        draw = ImageDraw.Draw(canvas)
        radius = math.hypot(self.patch_size[0] / 2, self.patch_size[1] / 2)
        patch_bounds = (
            pose[0] - radius,
            pose[1] - radius,
            pose[0] + radius,
            pose[1] + radius,
        )
        for kind, exterior, holes, bounds in features:
            if (
                bounds[2] < patch_bounds[0]
                or bounds[0] > patch_bounds[2]
                or bounds[3] < patch_bounds[1]
                or bounds[1] > patch_bounds[3]
            ):
                continue
            coordinates = self._canvas_coordinates(exterior, pose)
            if kind == "line":
                if len(coordinates) >= 2:
                    # Pillow also accepts a flat coordinate sequence.  Avoid
                    # constructing one Python tuple per map vertex: on dense
                    # nuScenes layers that conversion costs more than drawing.
                    draw.line(coordinates.reshape(-1).tolist(), fill=1, width=2)
            else:
                if len(coordinates) >= 3:
                    draw.polygon(coordinates.reshape(-1).tolist(), fill=1)
                for hole in holes:
                    hole_coordinates = self._canvas_coordinates(hole, pose)
                    if len(hole_coordinates) >= 3:
                        draw.polygon(hole_coordinates.reshape(-1).tolist(), fill=0)
        return np.asarray(canvas, dtype=np.uint8)
