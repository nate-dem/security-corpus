"""Reuse one unchanged vLLM model across sequential, bounded scoring packets."""


class ResidentModel:
    def __init__(self):
        self.model = None
        self.arguments = None

    def __call__(self, **kwargs):
        if self.model is None:
            from vllm import LLM

            self.model = LLM(**kwargs)
            self.arguments = kwargs
        elif kwargs != self.arguments:
            raise ValueError("Cannot reuse a model with different inference parameters")
        return self.model
