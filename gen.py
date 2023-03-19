import numpy as np
from PIL import Image

# create a blank image
new_image = Image.open("4.png").convert('L')

# add some noise to the image
# np.random.seed(42)
# noise = np.random.randint(low=0, high=256, size=(28, 28)).astype(np.uint8)
array = np.maximum(new_image, 0)
flat_list = array.reshape(-1).tolist()
print(", ".join(str(x) for x in flat_list))
new_image = Image.fromarray(array)

# save the image
new_image.save("new_digit.png")