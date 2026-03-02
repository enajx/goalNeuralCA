from PIL import Image


def print_image_info(image_path: str):
    """Print the size and number of channels of an image."""
    img = Image.open(image_path)
    width, height = img.size
    num_channels = len(img.getbands())
    has_alpha = "A" in img.getbands()

    print(f"Size: {width}x{height}")
    print(f"Channels: {num_channels} ({img.mode})")
    print(f"Has alpha: {has_alpha}")


if __name__ == "__main__":
    # Example usage
    image_path = "datasets/dendrites/dendrites.png"
    print_image_info(image_path)
