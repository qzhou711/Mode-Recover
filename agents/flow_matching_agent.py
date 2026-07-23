from agents.ddpm_agent import DiffusionAgent


class FlowMatchingAgent(DiffusionAgent):
    """Flow Matching policy using the established d3il agent lifecycle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_model_name = "eval_best_flow.pth"
        self.last_model_name = "last_flow.pth"
