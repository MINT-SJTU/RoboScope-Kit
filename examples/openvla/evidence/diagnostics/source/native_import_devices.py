"""Inspect TensorFlow visibility across the upstream evaluator import, without weights."""
import json
import sys
from pathlib import Path
import tensorflow as tf
before=[str(x) for x in tf.config.get_visible_devices("GPU")]
from experiments.robot.libero import run_libero_eval
result={"python":sys.version,"tensorflow":tf.__version__,"before_upstream_import":before,"after_upstream_import":[str(x) for x in tf.config.get_visible_devices("GPU")],"evaluator_file":run_libero_eval.__file__}
out=Path(__file__).resolve().parent.parent/"preprocess-probe/native-import-devices.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
