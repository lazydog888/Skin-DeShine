import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "Skin-DeShine.json"


def load_package():
    spec = importlib.util.spec_from_file_location(
        "skin_deshine_release",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    by_type = {}
    for node in data["nodes"]:
        by_type.setdefault(node["type"], []).append(node)

    deoil = by_type["LIN_DeOilSkin"][0]
    face_mask = by_type["LIN_FaceSkinMask"][0]

    assert [item["name"] for item in deoil["inputs"]] == ["image", "skin_mask", "manual_mask"]
    assert [item["name"] for item in deoil["outputs"]] == ["image", "shine_mask", "mask_preview"]
    assert deoil["widgets_values_named"]["processing_area"] == "inside_mask"
    assert deoil["widgets_values_named"]["strength"] == 1
    assert deoil["widgets_values_named"]["core_repair"] == 0.95
    assert deoil["widgets_values_named"]["color_repair"] == 1

    face_inputs = [item["name"] for item in face_mask["inputs"]]
    assert face_inputs == ["image", "manual_mask"]
    assert "analysis_models" not in face_inputs
    assert [item["name"] for item in face_mask["outputs"]] == ["skin_mask", "preview"]
    assert face_mask["widgets_values_named"]["algorithm"] == "mediapipe"
    assert face_mask["widgets_values_named"]["forehead_expand"] == 5.5

    for node_type in (
        "MaskPreview",
        "MaskToImage",
        "ImageCompositeMasked",
        "PreviewImage",
        "iToolsCompareImage",
    ):
        assert node_type in by_type, node_type

    expected_links = [
        [1, 1, 0, 2, 0, "IMAGE"],
        [3, 1, 0, 4, 0, "IMAGE"],
        [4, 4, 0, 2, 1, "MASK"],
        [5, 2, 0, 5, 0, "IMAGE"],
        [6, 1, 0, 5, 1, "IMAGE"],
        [7, 4, 0, 6, 0, "MASK"],
        [8, 2, 1, 7, 0, "MASK"],
        [9, 2, 0, 3, 0, "IMAGE"],
        [21, 6, 0, 14, 2, "MASK"],
        [22, 1, 0, 14, 0, "IMAGE"],
        [23, 6, 0, 15, 0, "MASK"],
        [24, 15, 0, 14, 1, "IMAGE"],
        [25, 14, 0, 16, 0, "IMAGE"],
    ]
    assert data["links"] == expected_links

    package = load_package()
    mappings = package.NODE_CLASS_MAPPINGS
    displays = package.NODE_DISPLAY_NAME_MAPPINGS
    expected_mappings = {
        "LIN_DeOilSkin",
        "LIN_FaceSkinMask",
        "LIN_SkinDeShineMaskLayer",
        "LIN_SkinDeShineMaskEditor",
    }
    assert expected_mappings.issubset(mappings)
    assert expected_mappings.issubset(displays)

    mask_inputs = mappings["LIN_FaceSkinMask"].INPUT_TYPES()
    assert "analysis_models" not in mask_inputs.get("required", {})
    assert "analysis_models" not in mask_inputs.get("optional", {})
    assert "manual_mask" in mask_inputs["optional"]
    assert "forehead_expand" in mask_inputs["required"]
    assert "hybrid" in mask_inputs["required"]["algorithm"][0]

    deoil_cls = mappings["LIN_DeOilSkin"]
    deoil_inputs = deoil_cls.INPUT_TYPES()
    assert deoil_inputs["required"]["processing_area"][0] == ["inside_mask", "outside_mask", "full_image"]
    assert "skin_mask" not in deoil_inputs["required"]
    assert "skin_mask" in deoil_inputs["optional"]
    assert deoil_cls.RETURN_TYPES == ("IMAGE", "MASK", "IMAGE")
    assert deoil_cls.RETURN_NAMES == ("image", "shine_mask", "mask_preview")

    print("workflow and node mapping schema check passed")


if __name__ == "__main__":
    main()
