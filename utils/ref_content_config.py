
class RefContentConfig:
    def __init__(self, 
                 sampling, 
                 ref_latent, 
                 content_transfer,
                 content_fideltiy, 
                 content_attention_bound,
                 content_start_percent,
                 content_end_percent,
                 ):
        self.sampling = sampling
        self.ref_latent = ref_latent
        self.content_fideltiy = content_fideltiy
        self.content_attention_bound = content_attention_bound
        self.content_transfer = content_transfer
        self.content_start_percent = content_start_percent
        self.content_end_percent = content_end_percent