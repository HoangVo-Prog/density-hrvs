"""SAM3 visualization must be the actual verifier/answer input, with image axes."""
from PIL import Image
import numpy as np
from algorithm.models.sam3_visual_prompt import build_sam3_region_context
from algorithm.models.spatial_context import spatial_verification_prompt
from algorithm.search.pipeline import evaluate_node
from algorithm.search.answering import answer_top_region
from test_question_region import state_with, node, QUESTION


def test_maps_clips_and_labels_the_same_detections_in_both_images():
    src = Image.new("RGB", (200, 100), (25, 35, 45)); before = src.tobytes()
    detections = [
        {"bbox": [-3, 2, 20, 15], "proposal_role": "target", "reference_id": "target_1", "prompt": "car", "sam3_score": .1},
        {"bbox": [25, 3, 40, 20], "proposal_role": "reference", "reference_id": "reference_1", "prompt": "bus", "sam3_score": .8},
        {"bbox": [99, 0, 110, 10], "proposal_role": "target", "prompt": "invalid"},
    ]
    ctx = build_sam3_region_context(src, (50, 20, 100, 60), detections=detections, max_side=100)
    boxes = ctx["metadata"]["sam3_boxes"]
    assert len(boxes) == 2 and {b["id"] for b in boxes} == {"T1.1", "R1.1"}
    assert boxes[0]["local_bbox_xyxy"] == [0, 2, 20, 15]
    assert boxes[0]["global_bbox_xyxy"] == [50, 22, 70, 35]
    assert boxes[0]["overview_bbox_xyxy"] == [25, 11, 35, 18]
    assert ctx["detail"].getpixel((0, 2)) == (255, 80, 210)
    assert ctx["overview"].getpixel((25, 11)) == (255, 80, 210)
    assert src.tobytes() == before
    prompt = spatial_verification_prompt("Which car?", ctx["metadata"])
    assert "displayed image" in prompt and "object's" in prompt
    assert "Answer Yes or No only" in prompt and '"prompt":"car"' in prompt
    assert "sam3_score" not in prompt and "normalized" not in prompt


def test_box_images_are_sent_to_verifier_export_and_answer():
    seen = []
    def detector(pixels, prompt):
        return [{"bbox": [2, 3, 10, 12], "sam3_score": .9}]
    def verifier(im, q, *, spatial_context, on_request):
        on_request(); seen.append(im.tobytes())
        assert q == QUESTION and spatial_context["metadata"]["sam3_boxes"]
        assert im.getpixel((2, 3)) == (255, 80, 210)
        return {"answer": "Yes", "score": .9, "model_id": "mock"}
    state = state_with(detector, verifier)
    state.settings.update(region_spatial_context=True, answer_view="spatial_context")
    evaluate_node(state, node())
    assert np.array_equal(state.observed_clusters[0]["image"], np.asarray(Image.frombytes("RGB", (20,20), seen[0])))
    def answer(im, q, *, spatial_context, on_request):
        on_request()
        assert im.tobytes() == seen[0]
        assert "displayed-image axes" in spatial_context["metadata"]["direction_convention"]
        return "blue"
    state.local_scorer.answer_spatial = answer
    result = answer_top_region(state, {"ranked_regions": state.observed_regions})
    assert result["answer"] == "blue"
    assert "NEVER" in result["prompt"] and "facing direction" in result["prompt"]
