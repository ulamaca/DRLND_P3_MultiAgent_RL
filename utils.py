import random

# Misc
def random_color(choice=False):
    "select the color code from "
    lib=['r', 'b', 'g', 'k', 'y', 'c', 'm']
    if isinstance(choice, int):
        if choice<=6:
            return lib[choice]
        else:
            raise ValueError("choice value should be less than/equal to 6")
    else:
        return random.choice(lib)