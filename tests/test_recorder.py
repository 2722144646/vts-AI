import asyncio

from dsh_vts_avatar.performance.recorder import record_expression


class FakeVTS:
    def __init__(self):
        self.calls = 0

    async def inject(self, params):
        return None

    async def current_model(self):
        return {"modelID": "fake", "modelName": "Fake"}

    async def list_input_params(self):
        self.calls += 1
        active = self.calls > 20
        return {
            "defaultParameters": [
                {"name": "MouthSmile", "value": 0.75 if active else 0.10},
                {"name": "FaceAngleY", "value": -5.0 if active else 0.0},
            ],
            "customParameters": [],
        }

    async def list_live2d_params(self):
        return {"parameters": [{"name": "ParamMouthForm", "value": 0.5 if self.calls > 20 else 0.0}]}


def test_record_expression_returns_deltas():
    summary = asyncio.run(record_expression(FakeVTS(), "test", duration_s=1.0, baseline_s=0.3, hz=30))
    assert summary["name"] == "test"
    assert summary["input_deltas"]["MouthSmile"]["delta"] > 0.4
    assert summary["input_deltas"]["FaceAngleY"]["delta"] < -1.0
    assert summary["live2d_deltas"]["ParamMouthForm"]["delta"] > 0.2
