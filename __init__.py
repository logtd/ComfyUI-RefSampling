from .nodes.apply_ref_unet_node import ApplyRefUNetNode
from .nodes.ref_apply_node import RefApplyNode

NODE_CLASS_MAPPINGS = {
    "ApplyRefUNetNode": ApplyRefUNetNode,
    "RefApplyNode": RefApplyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyRefUNetNode": "Apply Ref UNet",
    "RefApplyNode": "Apply Reference",
}
