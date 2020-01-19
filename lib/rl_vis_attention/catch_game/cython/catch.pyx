import copy, functools, os, datetime
from collections import defaultdict
import numpy as np

cdef class Ball:
    cdef (int, int) _position
    cdef (int, int) _direction
    cdef int _step_size

    def __cinit__(self, position, direction=(0, 1)):
        """
        Args:
            positions: array with structure [x, y]
            direction: array with structure [deltaX, deltaY]
        """
        self._position = tuple(position)
        self._direction = tuple(direction)
        self._step_size = 1
        #self._position = np.asarray(position)
        #self._direction = np.asarray(direction)

    cdef (int, int) next_position(self):
        next_pos_y = self._position[0] + self._step_size * self._direction[0]
        next_pos_x = self._position[0] + self._step_size * self._direction[1]
        return next_pos_y, next_pos_x
        #return self._position + step_size * self._direction

    cdef set_position(self, (int, int) position):
        self._position[0] = position[0]
        self._position[1] = position[1]
        #self._position = np.asarray(position)

    cdef set_direction(self, (int, int) direction):
        self._direction[0] = direction[0]
        self._direction[1] = direction[1]
        #self._direction = np.asarray(direction)

    cdef (int, int) get_position(self):
        return self._position

cdef class CatchEnv:
    SYMBOLS = {
        'BALL': 255,
        'CATCHER': 127,
        'EMPTY': 0
    }

    #cdef int _field_size
    #cdef object _field
    #cdef int [:, :] _field_view
    #cdef int _catcher_width
    #cdef int _catcher_height
    #cdef float _throw_rate
    #cdef int _max_dropped_balls
    #cdef float _max_balls
    #cdef object _balls
    # {y: {x: {balls...} } }
    #cdef object _balls_at_pos
    # [x, y]
    #cdef object _last_catcher_pos 
    #cdef int _time_step
    #cdef float _next_throw_time
    # Total balls droped in current episode
    #cdef int _total_dropped_balls
    # Number of balls that have passed the bottom of the field this episode
    #cdef int _balls_finished
    #cdef int _balls_thrown

    def __cinit__(self, field_size=10, catcher_width=1, throw_rate=1/10, max_dropped_balls=3, max_balls=None):
        """
        Args:
            field_size(int): Size of the field (in pixels)
            catch_width(int): Size of the catcher (in pixels)
            throw_rate(float): The arrival rate of balls
            max_dropped_balls(int): Maximum number of the balls the player can drop before the game/episode ends
            max_balls(int): Maximum number of balls to throw each game/episode
        """
        self._field_size = field_size
        self._field = np.full((self._field_size, self._field_size), CatchEnv.SYMBOLS['EMPTY'], dtype=np.int32)
        self._field_view = self._field
        self._catcher_width = catcher_width
        self._catcher_height = 1
        self._throw_rate = throw_rate
        self._max_dropped_balls = max_dropped_balls
        self._max_balls = float(max_balls) if max_balls is not None else float('inf')
        self._balls = set()
        self._balls_at_pos = defaultdict(functools.partial(defaultdict, set))

        self._reset()

    def seed(self, value):
        pass

    cpdef object get_state_shape(self):
        return [self._field_size, self._field_size]
    
    cpdef object get_action_shape(self):
        return (3,)

    cdef object _reset(self):
        self._balls.clear()
        self._balls_at_pos.clear()

        self._time_step = 0
        self._next_throw_time = -1
        self._total_dropped_balls = 0
        self._balls_finished = 0
        self._balls_thrown = 0

        # Clear field
        self._field.fill(CatchEnv.SYMBOLS['EMPTY'])
        
        # Place catcher in the centre
        x_centre = int((self._field_size + 1) / 2) - 1
        default_catcher_x = x_centre - int((self._catcher_width + 1) / 2) + 1
        self._move_catcher_to(x=default_catcher_x)
        
    cpdef object reset(self):
        self._reset()
        return self.get_state()

    cpdef object get_state(self):
        return copy.deepcopy(self._field)
    
    cpdef tuple get_legal_actions(self):
        return (0, 1, 2)
    
    cpdef tuple step(self, action):
        """
        Args:
            actions(int): 0 (left), 1 (don't move), 2 (right)
        Returns:
            (reward, done, state, debug_info_dict)
        """
        self._time_step += 1

        cdef int catcher_move_direction = action - 1 # using -1, 0, 1 internally
        self._move_catcher(catcher_move_direction)

        # Process balls
        self._move_balls()
        self._generate_new_ball()

        # Hacky way of showing catcher over ball
        self._draw_catcher()

        # Check bottom contacts
        dropped_balls, caught_balls = self._count_ball_contacts()
        self._total_dropped_balls += dropped_balls
        self._balls_finished += (dropped_balls + caught_balls)
        
        cdef bint episode_done = self.episode_done()
        #self._episode_done = (self._total_dropped_balls > self._max_dropped_balls) 


        # Reward for keeping the episode up
        # if episode_done:
        #     reward = -1
        # elif dropped_balls > 0:
        #     reward = -1 * dropped_balls
        # else:
        #     reward = 1

        # Reward for catching balls
        # if episode_done:
        #     reward = -1
        # elif dropped_balls > 0:
        #     reward = -1 * dropped_balls
        # elif caught_balls > 0:
        #     reward = caught_balls
        # else:
        #     reward = 0
        
        # Reward for catching balls
        cdef int reward = int(caught_balls > 0)

        info = {'dropped_balls': dropped_balls, 'caught_balls': caught_balls, 'dropped_balls_this_episode': self._total_dropped_balls}
        return self.get_state(), reward, episode_done, info
    
    cpdef bint episode_done(self):
        return (self._total_dropped_balls > self._max_dropped_balls) or (float(self._balls_finished) >= self._max_balls)

    cdef tuple _count_ball_contacts(self):
        """ 
        Handle caught and dropped balls
        Returns: number of dropped balls this timestep
        """
        cdef int dropped_balls = 0
        cdef int caught_balls = 0
        cdef int catcher_x_start = self._last_catcher_pos[0]
        cdef int catcher_x_end = self._last_catcher_pos[0] + self._catcher_width - 1

        for x_idx, x_balls in self._balls_at_pos[self._field_size - 1].items():
            if x_idx >= catcher_x_start and x_idx <= catcher_x_end:
                caught_balls += len(x_balls)
            else:
                dropped_balls += len(x_balls)
        
        return dropped_balls, caught_balls

    cdef _move_balls(self):
        balls_to_remove = set()
        cdef (int, int) old_pos
        cdef (int, int) new_pos
        cdef Ball ball
        for ball in self._balls:
            old_pos = ball.get_position()
            new_pos = ball.next_position()
            
            self._balls_at_pos[old_pos[1]][old_pos[0]].remove(ball)
            # clear drawing at old pos
            self._field_view[old_pos[1], old_pos[0]] = CatchEnv.SYMBOLS['EMPTY']

            if new_pos[1] < self._field_size:
                self._balls_at_pos[new_pos[1]][new_pos[0]].add(ball)
                ball.set_position(new_pos)
                # draw ball at new pos
                self._field_view[new_pos[1], new_pos[0]] = CatchEnv.SYMBOLS['BALL']
            else:
                # remove ball if its moved past the bottom of the field
                balls_to_remove.add(ball)
        
        if len(balls_to_remove) > 0:
            self._balls.difference_update(balls_to_remove)

    cdef _generate_new_ball(self):
        """ Checks if a ball should be thrown and throws it if so.
        Time between balls is distributed by Exp(lamdba=1/_throw_rate)
        """
        cdef int ball_x, ball_y
        if self._balls_thrown < self._max_balls:
            if self._next_throw_time == -1:
                self._next_throw_time = self._time_step + np.random.exponential(1/self._throw_rate)

            if self._time_step >= self._next_throw_time:
                self._balls_thrown += 1
                ball_x = np.argmax(np.random.multinomial(1, [1 / self._field_size] * self._field_size, size=1)).astype(int)
                ball_y = 0
                ball = Ball([ball_x, ball_y])
                self._balls.add(ball)
                self._balls_at_pos[ball_y][ball_x].add(ball)
                self._next_throw_time = -1
                # Draw new ball
                self._field_view[ball_y, ball_x] = CatchEnv.SYMBOLS['BALL']
    
    cdef _move_catcher_to(self, int x):
        cdef int y = self._field_size - 1

        # clip x movement
        if x < 0:
            x = 0
        if x > (self._field_size - self._catcher_width):
            x = self._field_size - self._catcher_width
        
        # remove catcher image at last position
        if not self._last_catcher_pos is None:
            self._draw_rect(self._last_catcher_pos[0], self._last_catcher_pos[1], self._catcher_width, 
                    self._catcher_height, CatchEnv.SYMBOLS['EMPTY'])
        
        self._last_catcher_pos = [x, y]
        self._draw_catcher()
    
    cdef _move_catcher(self, int direction):
        """ 
        Args:
            direction(int): One of -1 (left), 0 (don't move), 1 (right)
        """
        if direction != 0:
            self._move_catcher_to(x=self._last_catcher_pos[0] + direction)

    cdef _draw_catcher(self):
        self._draw_rect(self._last_catcher_pos[0], self._last_catcher_pos[1], self._catcher_width, 
                self._catcher_height, CatchEnv.SYMBOLS['CATCHER'])
    
    cdef _draw_rect(self, int top_left_x, int top_left_y, int width, int height, int fill):
        cdef int x_end = top_left_x + width
        cdef int y_end = top_left_y + height
        #self._field[top_left_y:y_end, top_left_x:x_end] = fill

        cdef int x_idx, y_idx
        for y_idx in range(top_left_y, y_end):
            for x_idx in range(top_left_x, x_end):
                self._field_view[y_idx, x_idx] = fill
