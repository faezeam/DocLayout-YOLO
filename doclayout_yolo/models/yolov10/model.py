from doclayout_yolo.engine.model import Model
from doclayout_yolo.models import yolo
from doclayout_yolo.nn.tasks import OBBModel, YOLOv10DetectionModel
from .val import YOLOv10DetectionValidator
from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer

from huggingface_hub import PyTorchModelHubMixin

class YOLOv10(Model, PyTorchModelHubMixin, repo_url="https://github.com/opendatalab/DocLayout-YOLO", pipeline_tag="object-detection", license="agpl-3.0"):

    def __init__(self, model="yolov10n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
            # OBB reuses the generic yolo.obb classes rather than YOLOv10-specific ones:
            # the v10 trainer/validator/predictor exist to serve v10Detect's dual
            # one2one/one2many head, which has no rotated counterpart.
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }