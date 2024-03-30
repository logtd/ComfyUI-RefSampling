from ..utils.ref_sampling_utils import prepare_ref_latents
from ..utils.ref_config import RefConfig


class RefApplyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "ref_latents": ("LATENT",),
            "style_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "attention_bound": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self, model, ref_latents, style_fidelity, attention_bound):

        ref_latents = prepare_ref_latents(model, ref_latents)
        sampling = model.model.model_sampling
        ref_config = RefConfig(sampling, ref_latents,
                               style_fidelity, attention_bound)

        transformer_options = model.model_options.get(
            'transformer_options', {})
        model.model_options['transformer_options'] = transformer_options
        transformer_options['ref_config'] = ref_config

        return (model, )
