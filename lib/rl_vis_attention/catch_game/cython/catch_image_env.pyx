from rl_vis_attention.catch_game.cython.catch cimport CatchEnv

cdef class CatchImageEnv(CatchEnv):
    """
    Wrapper that returns the state as a 3D tensor with appropriate channels.
    Currently only uses 1 channel.
    """

    cpdef object get_state(self, use_simple_shape=False):
        state = CatchEnv.get_state(self)
        #state = super(CatchImageEnv).get_state()
        if use_simple_shape:
            return state
        return state.reshape(self.get_state_shape())
    
    cpdef object get_state_shape(self):
        shape = CatchEnv.get_state_shape(self)
        #shape = super(CatchImageEnv).get_state_shape()
        return shape + [1]