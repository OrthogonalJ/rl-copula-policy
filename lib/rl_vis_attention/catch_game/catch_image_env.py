from rl_vis_attention.catch_game.catch import CatchEnv

class CatchImageEnv(CatchEnv):
    """
    Wrapper that returns the state as a 3D tensor with appropriate channels.
    Currently only uses 1 channel.
    """

    def get_state(self, use_simple_shape=False):
        state = super().get_state()
        if use_simple_shape:
            return state
        return state.reshape(self.get_state_shape())
    
    def get_state_shape(self):
        shape = super().get_state_shape()
        return shape + [1]