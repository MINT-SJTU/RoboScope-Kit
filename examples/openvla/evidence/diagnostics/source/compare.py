"""Compare saved paired traces without inferring missing steps."""
import json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

report = []
for suite in ["spatial", "object", "goal", "10"]:
    for nr in [ROOT / f"{suite}-native", ROOT / f"{suite}-native-extra"]:
        kr = ROOT / f"{suite}-kit"
        if not (nr / "episodes.jsonl").exists() or not (kr / "result.json").exists():
            continue
        outcomes = {r["episode_id"]: r for r in rows(nr / "episodes.jsonl")}
        kout = {}
        for file in (kr / "queue/completed").glob("*.json"):
            for r in json.loads(file.read_text())["records"]:
                assert r["record_id"] not in kout
                kout[r["record_id"]] = {**r, "worker_id": json.loads(file.read_text())["worker_id"]}
        native = rows(nr / "trace/steps.jsonl")
        kit = rows(kr / "policy-trace/steps.jsonl")
        for episode, outcome in outcomes.items():
            if episode not in kout:
                continue
            ns = {r["step"]: r for r in native if r["episode_id"] == episode}
            ks = {r["step"]: r for r in kit if r["episode_id"] == episode}
            assert len(ns) == sum(r["episode_id"] == episode for r in native)
            assert len(ks) == sum(r["episode_id"] == episode for r in kit)
            assert len(ns) == outcome["policy_steps"]
            assert len(ks) == kout[episode]["steps"]
            first = {}
            details = []
            for step in sorted(ns.keys() & ks.keys()):
                a, b = ns[step], ks[step]
                diff = {}
                for key in ["raw_primary", "canonical_primary", "resized_primary", "model_image"]:
                    if a["images"][key]["sha256"] != b["images"][key]["sha256"]:
                        first.setdefault(key, step)
                        diff[key] = True
                if a["prompt"] != b["prompt"]:
                    first.setdefault("prompt", step)
                for key in ["model_action", "environment_action"]:
                    av, bv = np.asarray(a[key]), np.asarray(b[key])
                    delta = float(np.max(np.abs(av-bv)))
                    if delta:
                        first.setdefault(key + "_exact", step)
                    if delta > 1e-6:
                        first.setdefault(key + "_gt_1e-6", step)
                    diff[key + "_max_abs"] = delta
                with np.load(nr / "trace" / a["files"] / "processor-inputs.npz") as ai, np.load(kr / "policy-trace" / b["files"] / "processor-inputs.npz") as bi:
                    assert set(ai.files) == set(bi.files)
                    for key in ai.files:
                        if not np.array_equal(ai[key], bi[key]):
                            first.setdefault("processor_" + key, step)
                            diff["processor_" + key] = True
                        if key == "pixel_values":
                            av = torch.from_numpy(ai[key]).to(torch.bfloat16).view(torch.int16).numpy()
                            bv = torch.from_numpy(bi[key]).to(torch.bfloat16).view(torch.int16).numpy()
                            if not np.array_equal(av, bv):
                                first.setdefault("processor_pixel_values_bfloat16_derived", step)
                                diff["processor_pixel_values_bfloat16_derived"] = True
                details.append({"step": step, **diff})
            item = {"suite": suite, "episode_id": episode, "task": outcome["task"]["name"],
                    "native_success": outcome["success"], "kit_success": kout[episode]["success"],
                    "native_steps": outcome["policy_steps"], "kit_steps": kout[episode]["steps"],
                    "first_divergence": first, "step_comparison": details,
                    "different_model_action_with_equal_saved_model_inputs": [d["step"] for d in details if d["model_action_max_abs"] > 1e-6 and not d.get("processor_pixel_values_bfloat16_derived") and not d.get("processor_input_ids") and not d.get("processor_attention_mask")],
                    "note": "Exact float differences and >1e-6 differences are descriptive, not an acceptance tolerance."}
            historical = next(r for r in json.loads((ROOT / "historical-disagreements.json").read_text())
                              if r["suite"] == "libero_" + suite and r["task_id"] == outcome["task_id"]
                              and r["initial_state_index"] == outcome["initial_state_index"])
            item["historical"] = historical
            item["native_source_batch"] = nr.name
            item["kit_worker_id"] = kout[episode].get("worker_id")
            assert set(ns) == set(range(outcome["policy_steps"]))
            assert set(ks) == set(range(kout[episode]["steps"]))
            worker_root = kr / "worker-trace" / episode
            if (worker_root / "initial.json").exists():
                initial = json.loads((worker_root / "initial.json").read_text())
                item["initial_state_arrays_equal"] = initial["initial_state_sha256"] == outcome["initial_state"]["sha256"]
                item["initial_images_equal"] = initial["initial_image_sha256"] == outcome["initial_primary_image"]["sha256"]
                item["kit_reset_count"] = initial["reset_count_in_task_runtime"]
                item["native_reset_count"] = outcome.get("reset_count_in_task_runtime")
                actual_rows = [r for r in rows(worker_root / "steps.jsonl") if r["policy_step"] >= 0]
                actual = {r["policy_step"]: r for r in actual_rows}
                assert len(actual) == len(actual_rows)
                assert set(actual) == set(ks)
                assert all(np.array_equal(actual[t]["action"], ks[t]["environment_action"]) for t in ks)
                item["worker_actions_match_policy_trace"] = True
                for stage in ["initial", "after-settling"]:
                    native_file = nr / "trace" / episode / (stage + ".npz")
                    if native_file.exists():
                        with np.load(native_file) as na, np.load(worker_root / (stage + ".npz")) as ka:
                            item[stage + "_state_comparison"] = {key: {"equal": bool(np.array_equal(na[key], ka[key])),
                                "max_abs": float(np.max(np.abs(na[key]-ka[key]))) if na[key].size else 0.0}
                                for key in na.files if key in ka.files}
            report.append(item)
            milestones = [first[k] for k in ["raw_primary", "model_image", "processor_pixel_values_bfloat16_derived", "model_action_gt_1e-6"] if k in first]
            candidates = sorted(set([0, min(max(ns), max(ks))] + [max(0, v-1) for v in milestones] + milestones))
            # Contact sheet: raw observations before each policy action.
            sheet = Image.new("RGB", ((len(candidates)+1)*256, 560), "white")
            draw = ImageDraw.Draw(sheet)
            for col, step in enumerate(candidates):
                for row, mapping, root, label in [(0,ns,nr/"trace","Native"),(1,ks,kr/"policy-trace","Kit")]:
                    if step not in mapping: continue
                    with Image.open(root / mapping[step]["files"] / "canonical_primary.png") as im:
                        sheet.paste(im, (col*256,row*280+24))
                    draw.text((col*256+4,row*280+4), f"{label} step {step}", fill="black")
            for row, mapping, root, label in [(0,ns,nr/"trace","Native"),(1,ks,kr/"policy-trace","Kit")]:
                last = max(mapping)
                with Image.open(root / mapping[last]["files"] / "canonical_primary.png") as im:
                    sheet.paste(im, (len(candidates)*256,row*280+24))
                draw.text((len(candidates)*256+4,row*280+4), f"{label} last input: step {last}", fill="black")
            sheet.save(ROOT / f"{suite}-{episode}-comparison.png")
(ROOT / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps([{k:v for k,v in x.items() if k != "step_comparison"} for x in report],indent=2))
