import enum


class RefMode(enum.Enum):
    OFF = 1
    WRITE = 2
    READ = 3


class RefController:
    modules = []
    temp_modules = []
    default_video_length = 16
    default_full_length = 16

    def add_module(self, module):
        self.modules.append(module)

    def set_mode(self, mode: RefMode):
        for module in self.modules:
            module.ref_mode = mode

    def clear_modules(self):
        for module in self.modules:
            module.ref_bank = None
        self.set_mode(RefMode.OFF)

    def set_cfg(self, cfg: bool):
        for module in self.modules:
            module.cfg = cfg

    def set_attention_bound(self, attention_bound: float):
        for module in self.modules:
            module.attention_bound = attention_bound

    def set_style_fidelity(self, style_fidelity: float): 
        for module in self.modules:
            module.style_fidelity = style_fidelity

    def add_temporal_transformer(self, module):
        self.temp_modules.append(module)

    def get_default_temporal_values(self):
        if len(self.temp_modules) > 0:
            self.default_full_length = self.temp_modules[0].full_length
            self.default_video_length = self.temp_modules[0].video_length

    def set_temporal_values_to(self, full_length, video_length):
        for module in self.temp_modules:
            module.full_length = full_length
            module.video_length = video_length

    def set_temporal_values_to_default(self):
        self.set_temporal_values_to(self.default_full_length, self.default_video_length)