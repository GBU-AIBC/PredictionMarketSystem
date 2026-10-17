from __future__ import annotations

from dataclasses import dataclass, field

from pms.core.models import EvalRecord


@dataclass
class NetcalCalibrator:
    min_samples: int = 100
    _samples: dict[str, list[EvalRecord]] = field(default_factory=dict)
    _ingested_decision_ids: dict[str, set[str]] = field(default_factory=dict)

    def add_samples(self, model_id: str, records: list[EvalRecord]) -> None:
        samples = self._samples.setdefault(model_id, [])
        ingested = self._ingested_decision_ids.setdefault(model_id, set())
        for record in records:
            if record.decision_id in ingested:
                continue
            ingested.add(record.decision_id)
            samples.append(record)

    def sample_count(self, model_id: str) -> int:
        return len(self._samples.get(model_id, []))

    def calibrate(self, prob: float, *, model_id: str) -> float:
        records = self._samples.get(model_id, [])
        if len(records) < self.min_samples:
            return prob
        points = sorted(
            (
                record.calibration_prob_estimate
                if record.calibration_prob_estimate is not None
                else record.prob_estimate,
                record.resolved_outcome,
            )
            for record in records
        )
        blocks = _pava(points)
        for upper_bound, value in blocks:
            if prob <= upper_bound:
                return _clip_probability(value)
        return _clip_probability(blocks[-1][1])


def _pava(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    blocks: list[tuple[float, float, int, float]] = []
    for x_value, y_value in points:
        blocks.append((x_value, y_value, 1, x_value))
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
            left_x, left_y, left_count, _ = blocks.pop(-2)
            right_x, right_y, right_count, right_upper = blocks.pop(-1)
            total_count = left_count + right_count
            mean_y = (
                left_y * left_count + right_y * right_count
            ) / total_count
            blocks.append((left_x, mean_y, total_count, right_upper))
    return [(upper_bound, value) for _, value, _, upper_bound in blocks]


def _clip_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)
