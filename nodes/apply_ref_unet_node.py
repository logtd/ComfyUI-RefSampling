
from ..utils.module_utils import setup_ref_unet

class ApplyRefUNetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "attn": (["FULL"],),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self, model, attn):
        ref_controller = setup_ref_unet(model, attn)
        if ref_controller is not None:
            model.model_options['ref_controller'] = ref_controller
        return (model, )