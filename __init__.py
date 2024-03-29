from .nodes.apply_ref_unet_node import ApplyRefUNetNode
from .nodes.ref_sampler_node import RefSamplerNode, RefSamplerCustomNode

NODE_CLASS_MAPPINGS = {
    "ApplyRefUNetNode": ApplyRefUNetNode,
    "RefSamplerNode": RefSamplerNode,
    "RefSamplerCustomNode": RefSamplerCustomNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyRefUNetNode": "Apply Ref UNet",
    "RefSamplerNode": "Ref Sampler",
    "RefSamplerCustomNode": "Ref Sampler Custom",
}
