import numpy as np
import random

full_list = list(range(0, 129))

# Randomly select 20 unique integers from the list
selected = random.sample(full_list, 20)

# Create a list with the remaining integers
remaining = [x for x in full_list if x not in selected]