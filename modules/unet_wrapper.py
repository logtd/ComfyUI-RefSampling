import torch

from .ref_controller import RefController, RefMode
from ..utils.ref_config import RefConfig
from ..utils.noise_utils import add_noise

def get_unet_wrapper(cls, ref_controller: RefController):
    class RefUNet(cls):
        is_ref = True
        def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            if 'ref_config' not in transformer_options:
                return super().forward(x, 
                                timesteps=timesteps,
                                context=context,
                                y=y,
                                control=control,
                                transformer_options=transformer_options,
                                **kwargs)
            
            ref_config: RefConfig = transformer_options['ref_config']
            ref_latent = ref_config.ref_latent
            ref_latent = torch.cat([ref_latent]*2)

            sigma = ref_config.sampling.sigma(timesteps)

            ref_latent_noised = add_noise(ref_latent, torch.randn_like(ref_latent), sigma[0]).to(x.device).to(x.dtype)

            ref_controller.set_cfg(True)  #TODO
            ref_controller.set_attention_bound(ref_config.attention_bound)
            ref_controller.set_style_fidelity(ref_config.style_fidelity)

            ref_controller.clear_modules()
            try:
                ref_controller.set_mode(RefMode.WRITE)
                super().forward(ref_latent_noised, 
                                timesteps=timesteps,
                                context=context,
                                y=y,
                                control=None,
                                transformer_options=transformer_options,
                                **kwargs)
                
                ref_controller.set_mode(RefMode.READ)
                output = super().forward(x, 
                                timesteps=timesteps,
                                context=context,
                                y=y,
                                control=control,
                                transformer_options=transformer_options,
                                **kwargs)
                ref_controller.set_mode(RefMode.OFF)
                return output
            # except Exception as e:
            #     print('issue with ref')
            #     raise e
            finally:
                ref_controller.clear_modules()
        
    return RefUNet