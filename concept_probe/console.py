class ConsoleLogger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)

    def info(self, message: str) -> None:
        if self.enabled:
            print(message, flush=True)

    def warn(self, message: str) -> None:
        if self.enabled:
            print(f"WARNING: {message}", flush=True)
