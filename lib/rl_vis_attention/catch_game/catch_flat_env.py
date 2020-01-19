from rl_vis_attention.catch_game.catch import CatchEnv

class CatchFlatEnv(CatchEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_state(self, unflatten=False):
        state = super().get_state()
        if unflatten:
            return state
        return state.reshape(-1)
