import torch

from .ref_controller import RefController, RefMode
from ..utils.ref_content_config import RefContentConfig
from ..utils.ref_style_config import RefStyleConfig
from ..utils.noise_utils import add_noise

def get_unet_wrapper(cls, ref_controller: RefController):
    class RefUNet(cls):
        is_ref = True

        def _ref_content_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            ref_config: RefContentConfig = transformer_options.get('ref_content_config', None)
            if ref_config is None:
                return RefMode.OFF
            ref_latent = ref_config.ref_latent
            ref_latent = torch.cat([ref_latent]*2)
            sigma = ref_config.sampling.sigma(timesteps)
            start_sigma = ref_config.sampling.percent_to_sigma(ref_config.content_start_percent)
            end_sigma = ref_config.sampling.percent_to_sigma(ref_config.content_end_percent)
            if not (start_sigma >= sigma[0] >= end_sigma):
                return RefMode.OFF

            ref_latent_noised = add_noise(ref_latent, torch.randn_like(ref_latent), sigma[0]).to(x.device).to(x.dtype)

            ref_controller.set_cfg(len(transformer_options['cond_or_uncond']) > 1) # TODO
            ref_controller.set_attention_bound(ref_config.content_attention_bound)
            ref_controller.set_content_fidelity(ref_config.content_fideltiy)

            ref_controller.set_normal_mode(RefMode.WRITE)
            super().forward(ref_latent_noised, 
                        timesteps=timesteps,
                        context=context,
                        y=y,
                        control=None,
                        transformer_options=transformer_options,
                        **kwargs)
            ref_controller.set_normal_mode(RefMode.OFF)

            return RefMode.READ
        
        def _ref_style_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            ref_config: RefStyleConfig = transformer_options.get('ref_style_config', None)
            if ref_config is None:
                return RefMode.OFF, None
            ref_latent = ref_config.style_latent
            ref_latent = torch.cat([ref_latent]*2)
            sigma = ref_config.sampling.sigma(timesteps)
            start_sigma = ref_config.sampling.percent_to_sigma(ref_config.style_start_percent)
            end_sigma = ref_config.sampling.percent_to_sigma(ref_config.style_end_percent)
            if not (start_sigma >= sigma[0] >= end_sigma):
                return RefMode.OFF, None

            ref_latent_noised = add_noise(ref_latent, torch.randn_like(ref_latent), sigma[0]).to(x.device).to(x.dtype)

            ref_controller.set_kv_mode(RefMode.WRITE, ref_config.style_count)
            super().forward(ref_latent_noised, 
                        timesteps=timesteps,
                        context=ref_config.style_prompt.to(ref_latent_noised.device).half(),
                        y=y,
                        control=None,
                        transformer_options=transformer_options,
                        **kwargs)
            ref_controller.set_kv_mode(RefMode.OFF)

            return RefMode.READ, ref_config.style_count
            

        def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            ref_controller.set_mode(RefMode.OFF)
            ref_controller.clear_modules()
            try:
                ref_controller.set_mode(RefMode.WRITE)
                normal_read = self._ref_content_forward(
                    x, 
                    timesteps=timesteps,
                    context=context,
                    y=y,
                    control=control,
                    transformer_options=transformer_options,
                    **kwargs
                )
                kv_read, kv_count = self._ref_style_forward(
                    x, 
                    timesteps=timesteps,
                    context=context,
                    y=y,
                    control=control,
                    transformer_options=transformer_options,
                    **kwargs
                )
                ref_controller.set_mode(RefMode.READ)
                ref_controller.set_normal_mode(normal_read)
                ref_controller.set_kv_mode(kv_read, kv_count)
                output = super().forward(x, 
                                timesteps=timesteps,
                                context=context,
                                y=y,
                                control=control,
                                transformer_options=transformer_options,
                                **kwargs)
                ref_controller.set_mode(RefMode.OFF)
                return output
            finally:
                ref_controller.clear_modules()
        
    return RefUNet