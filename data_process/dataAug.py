import random
from PIL import Image
from PIL import ImageOps

def data_augmentation(image, augment_methods=["rotate", "flip", "scale", "crop"]):
    augmented_images = []
    for method in augment_methods:
        if method == "rotate":
            angle = random.choice([10, 20, 30])
            augmented_images.append(image.rotate(angle, expand=True))
            augmented_images.append(image.rotate(-angle, expand=True))
        elif method == "flip":
            augmented_images.append(ImageOps.flip(image))
            augmented_images.append(ImageOps.mirror(image))
        elif method == "scale":
            scale = random.choice([0.9, 1.1])
            new_size = (int(image.width * scale), int(image.height * scale))
            augmented_images.append(image.resize(new_size, Image.Resampling.LANCZOS))
        elif method == "crop":
            left = random.randint(0, image.width // 10)
            upper = random.randint(0, image.height // 10)
            right = random.randint(image.width * 9 // 10, image.width)
            lower = random.randint(image.height * 9 // 10, image.height)
            augmented_images.append(image.crop((left, upper, right, lower)))
    return random.choices(augmented_images, k=1)

