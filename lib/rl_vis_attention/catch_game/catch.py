import copy, functools, os, datetime
from collections import defaultdict
import numpy as np

class Ball:
    def __init__(self, position, direction=[0, 1], fall_rate=1/2):
        """
        Args:
            positions: array with structure [x, y]
            direction: array with structure [deltaX, deltaY]
        """
        self._position = np.asarray(position)
        self._direction = np.asarray(direction)
        self._fall_rate = fall_rate
        self._momentum = 0.0

    def next_position(self, step_size=1):
        self._momentum += self._fall_rate
        if self._momentum >= 1.0:
            self._position = self._position + step_size * self._direction
            self._momentum = 0.0
        return self._position

    def set_position(self, position):
        self._position = np.asarray(position)

    def set_direction(self, direction):
        self._direction = np.asarray(direction)


class CatchEnv:
    SYMBOLS = {
        'BALL': 255,
        'CATCHER': 255,
        'EMPTY': 0
    }

    def __init__(self, field_size=10, catcher_width=1, throw_rate=1/10, max_dropped_balls=3, max_balls=None, fall_rate=1.0):
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
        self._catcher_width = catcher_width
        self._catcher_height = 1
        self._throw_rate = throw_rate
        self._max_dropped_balls = max_dropped_balls
        self._max_balls = float(max_balls) if max_balls is not None else float('inf')
        self._fall_rate = fall_rate
        self._balls = set()
        # {y: {x: {balls...} } }
        self._balls_at_pos = defaultdict(functools.partial(defaultdict, set))

        self._last_catcher_pos = None # [x, y]
        self._time_step = None
        self._next_throw_time = None
        # Total balls droped in current episode
        self._total_dropped_balls = None
        # Number of balls that have passed the bottom of the field this episode
        self._balls_finished = None
        self._balls_thrown = None

        # self._episode_done = None
        self.reset(return_state=False)

    def seed(self, value):
        pass

    def get_state_shape(self):
        return [self._field_size, self._field_size]
    
    def get_action_shape(self):
        return (3,)

    def reset(self, return_state=True):
        self._balls.clear()
        self._balls_at_pos.clear()

        self._time_step = 0
        self._next_throw_time = None
        self._total_dropped_balls = 0
        self._balls_finished = 0
        self._balls_thrown = 0
        # self._episode_done = False

        # Clear field
        self._field.fill(CatchEnv.SYMBOLS['EMPTY'])
        
        # Place catcher in the centre
        # x_centre = int((self._field_size + 1) / 2) - 1
        # default_catcher_x = x_centre - int((self._catcher_width + 1) / 2) + 1
        # self._move_catcher_to(x=default_catcher_x)
        # Place catcher in random position
        catcher_x = np.argmax(np.random.multinomial(1,
                [1 / self._field_size] * self._field_size, size=1))
        self._move_catcher_to(x=catcher_x)
         
        if return_state:
            return self.get_state()

    def get_state(self):
        return copy.deepcopy(self._field)
    
    def get_legal_actions(self):
        return (0, 1, 2)
    
    def step(self, action):
        """
        Args:
            actions(int): 0 (left), 1 (don't move), 2 (right)
        Returns:
            (reward, done, state, debug_info_dict)
        """
        self._time_step += 1

        catcher_move_direction = action - 1 # using -1, 0, 1 internally
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
        
        episode_done = self.episode_done()
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
        reward = int(caught_balls > 0)

        info = {'dropped_balls': dropped_balls, 'caught_balls': caught_balls, 'dropped_balls_this_episode': self._total_dropped_balls}
        return self.get_state(), reward, episode_done, info
    
    def episode_done(self):
        return (self._total_dropped_balls > self._max_dropped_balls) or (float(self._balls_finished) >= self._max_balls)

    def _count_ball_contacts(self):
        """ 
        Handle caught and dropped balls
        Returns: number of dropped balls this timestep
        """
        dropped_balls = 0
        caught_balls = 0
        catcher_x_start = self._last_catcher_pos[0]
        catcher_x_end = self._last_catcher_pos[0] + self._catcher_width - 1

        for x_idx, x_balls in self._balls_at_pos[self._field_size - 1].items():
            if x_idx >= catcher_x_start and x_idx <= catcher_x_end:
                caught_balls += len(x_balls)
            else:
                dropped_balls += len(x_balls)
        
        return dropped_balls, caught_balls

    def _move_balls(self):
        balls_to_remove = set()

        for ball in self._balls:
            old_pos = ball._position
            new_pos = ball.next_position()
            
            self._balls_at_pos[old_pos[1]][old_pos[0]].remove(ball)
            # clear drawing at old pos
            self._field[old_pos[1], old_pos[0]] = CatchEnv.SYMBOLS['EMPTY']

            if new_pos[1] < self._field_size:
                self._balls_at_pos[new_pos[1]][new_pos[0]].add(ball)
                # Handling position update in Ball class now
                # ball.set_position(new_pos)

                # draw ball at new pos
                self._field[new_pos[1], new_pos[0]] = CatchEnv.SYMBOLS['BALL']
            else:
                # remove ball if its moved past the bottom of the field
                balls_to_remove.add(ball)
        
        if len(balls_to_remove) > 0:
            self._balls.difference_update(balls_to_remove)

    def _generate_new_ball(self):
        """ Checks if a ball should be thrown and throws it if so.
        Time between balls is distributed by Exp(lamdba=1/_throw_rate)
        """
        if self._balls_thrown < self._max_balls:
            if self._next_throw_time is None:
                self._next_throw_time = self._time_step + np.random.exponential(1/self._throw_rate)

            if self._time_step >= self._next_throw_time:
                self._balls_thrown += 1
                # ball_x = 0 # FIXME: undo this
                ball_x = np.argmax(np.random.multinomial(1, [1 / self._field_size] * self._field_size, size=1))
                ball_y = 0
                ball = self._make_ball([ball_x, ball_y])
                #ball = Ball([ball_x, ball_y], fall_rate=self._fall_rate)
                self._balls.add(ball)
                self._balls_at_pos[ball_y][ball_x].add(ball)
                self._next_throw_time = None
                # Draw new ball
                self._field[ball_y, ball_x] = CatchEnv.SYMBOLS['BALL']
    
    def _make_ball(self, position):
        return Ball(position, fall_rate=self._fall_rate)

    def _move_catcher_to(self, x=None, y=None):
        y = self._field_size - 1 if y is None else y
        assert y >= 0 and y < self._field_size

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
    
    def _move_catcher(self, direction):
        """ 
        Args:
            direction(int): One of -1 (left), 0 (don't move), 1 (right)
        """
        if direction != 0:
            self._move_catcher_to(x=self._last_catcher_pos[0] + direction)

    def _draw_catcher(self):
        self._draw_rect(self._last_catcher_pos[0], self._last_catcher_pos[1], self._catcher_width, 
                self._catcher_height, CatchEnv.SYMBOLS['CATCHER'])
    
    def _draw_rect(self, top_left_x, top_left_y, width, height, fill):
        x_end = top_left_x + width
        y_end = top_left_y + height
        self._field[top_left_y:y_end, top_left_x:x_end] = fill
