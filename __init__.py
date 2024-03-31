from .nodes.apply_ref_unet_node import ApplyRefUNetNode
from .nodes.apply_ref_content_node import ApplyRefContentNode
from .nodes.apply_ref_style_node import ApplyRefStyleNode

NODE_CLASS_MAPPINGS = {
    "ApplyRefUNetNode": ApplyRefUNetNode,
    "ApplyRefContentNode": ApplyRefContentNode,
    "ApplyRefStyleNode": ApplyRefStyleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyRefUNetNode": "Apply Ref UNet",
    "ApplyRefContentNode": "Apply Ref Content",
    "ApplyRefStyleNode": "Apply Ref Style",
}
