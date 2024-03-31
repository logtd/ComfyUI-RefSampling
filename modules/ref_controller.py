import enum

from .block_type import BlockType


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
        if mode == RefMode.OFF:
            self.set_kv_mode(mode)
            self.set_normal_mode(mode)

    def set_normal_mode(self, mode: RefMode):
        for module in self.modules:
            module.normal_mode = mode

    def set_kv_mode(self, mode: RefMode, count = None):
        for module in self.modules:
            module.kv_mode = RefMode.OFF
        if mode == RefMode.OFF or count is None:
            count = len(self.modules)
        output_modules = list(filter(lambda m: m.block_type == BlockType.OUTPUT, self.modules))
        output_modules = sorted(output_modules, key=lambda m: -m.block_idx)[:count]
        for module in output_modules:
            module.kv_mode = mode

    def clear_modules(self):
        for module in self.modules:
            module.clean()
        self.set_mode(RefMode.OFF)
        self.set_kv_mode(RefMode.OFF, len(self.modules))
        self.set_normal_mode(RefMode.OFF)

    def set_cfg(self, cfg: bool):
        for module in self.modules:
            module.is_cfg = cfg

    def set_attention_bound(self, attention_bound: float):
        for module in self.modules:
            module.attention_bound = attention_bound

    def set_content_fidelity(self, content_fidelity: float): 
        for module in self.modules:
            module.content_fidelity = content_fidelity
