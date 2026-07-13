from __future__ import annotations

class EMA:
    def __init__(self, model, decay: float = 0.9999, warmup: bool = False):
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def update(self, model) -> None:
        self.num_updates += 1
        decay = self.decay
        if self.warmup:
            # Fast early tracking, asymptotically approaching the configured cap.
            decay = min(decay, 1 - (1 + self.num_updates) ** (-2 / 3))
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if value.is_floating_point():
                shadow.lerp_(value.detach(), 1 - decay)
            else:
                shadow.copy_(value)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state) -> None:
        self.decay = state["decay"]
        self.warmup = bool(state.get("warmup", False))
        self.num_updates = int(state.get("num_updates", 0))
        loaded = state["shadow"]
        self.shadow = {
            name: loaded[name].to(device=value.device, dtype=value.dtype)
            for name, value in self.shadow.items()
        }
