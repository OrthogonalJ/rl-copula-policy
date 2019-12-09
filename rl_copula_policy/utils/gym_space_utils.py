from gym.spaces import Dict, Box, Tuple

def convert_to_flat_tuple_space(space):
    return [space] if not isinstance(space, Tuple) else list(space)