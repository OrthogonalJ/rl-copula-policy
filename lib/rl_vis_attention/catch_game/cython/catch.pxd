cdef class CatchEnv:
    cdef int _field_size
    cdef object _field
    cdef int [:, :] _field_view
    cdef int _catcher_width
    cdef int _catcher_height
    cdef float _throw_rate
    cdef int _max_dropped_balls
    cdef float _max_balls
    cdef object _balls
    # {y: {x: {balls...} } }
    cdef object _balls_at_pos
    # [x, y]
    cdef object _last_catcher_pos 
    cdef int _time_step
    cdef float _next_throw_time
    # Total balls droped in current episode
    cdef int _total_dropped_balls
    # Number of balls that have passed the bottom of the field this episode
    cdef int _balls_finished
    cdef int _balls_thrown

    cpdef object get_state_shape(self)
    cpdef object get_action_shape(self)
    cdef object _reset(self)
    cpdef object reset(self)
    cpdef tuple get_legal_actions(self)
    cpdef object get_state(self)
    cpdef tuple step(self, action)
    cpdef bint episode_done(self)
    cdef tuple _count_ball_contacts(self)
    cdef _move_balls(self)
    cdef _generate_new_ball(self)
    cdef _move_catcher_to(self, int x)
    cdef _move_catcher(self, int direction)
    cdef _draw_catcher(self)
    cdef _draw_rect(self, int top_left_x, int top_left_y, int width, int height, int fill)
    