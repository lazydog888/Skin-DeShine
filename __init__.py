from .deoil import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .skin_mask import LIN_FaceSkinMask, LIN_SkinDeShineMaskLayer, LIN_SkinDeShineMaskEditor

NODE_CLASS_MAPPINGS["LIN_FaceSkinMask"] = LIN_FaceSkinMask
NODE_CLASS_MAPPINGS["LIN_SkinDeShineMaskLayer"] = LIN_SkinDeShineMaskLayer
NODE_CLASS_MAPPINGS["LIN_SkinDeShineMaskEditor"] = LIN_SkinDeShineMaskEditor
NODE_DISPLAY_NAME_MAPPINGS["LIN_DeOilSkin"] = "Skin DeShine"
NODE_DISPLAY_NAME_MAPPINGS["LIN_FaceSkinMask"] = "Skin DeShine Mask"
NODE_DISPLAY_NAME_MAPPINGS["LIN_SkinDeShineMaskLayer"] = "Skin DeShine Mask Composite"
NODE_DISPLAY_NAME_MAPPINGS["LIN_SkinDeShineMaskEditor"] = "Skin DeShine Mask Layer"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
