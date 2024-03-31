
class RefStyleConfig:
    def __init__(self, 
                 sampling, 
                 style_latent, 
                 style_transfer,
                 style_prompt,
                 style_count,
                 style_start_percent,
                 style_end_percent
                 ):
        self.sampling = sampling
        self.style_latent = style_latent
        self.style_transfer = style_transfer
        self.style_count = style_count
        self.style_prompt = style_prompt
        self.style_start_percent = style_start_percent
        self.style_end_percent = style_end_percent