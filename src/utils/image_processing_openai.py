import argparse
import os
from openai import OpenAI


def generate_image(
    prompt: str,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard",
    output_path: str = None,
):
    """
    Generate an image from a text prompt using OpenAI's image generation API.

    Args:
        prompt: Text description of the image to generate.
        model: Model to use ("dall-e-2" or "dall-e-3").
        size: Output size. For dall-e-2: "256x256", "512x512", "1024x1024".
              For dall-e-3: "1024x1024", "1792x1024", "1024x1792".
        quality: Image quality ("standard" or "hd"). Only for dall-e-3.
        output_path: Path to save the generated image. If None, saves as generated_{timestamp}.png.

    Returns:
        Path to the saved generated image.
    """
    client = OpenAI()

    kwargs = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }

    if model == "dall-e-3":
        kwargs["quality"] = quality

    response = client.images.generate(**kwargs)

    image_url = response.data[0].url

    # Download and save the image
    import requests

    img_data = requests.get(image_url).content

    if output_path is None:
        import time
        timestamp = int(time.time())
        output_path = f"generated_{timestamp}.png"

    with open(output_path, "wb") as f:
        f.write(img_data)

    print(f"Generated image saved to: {output_path}")
    return output_path


def edit_image(
    image_path: str,
    prompt: str,
    model: str = "dall-e-2",
    mask_path: str = None,
    size: str = "512x512",
    output_path: str = None,
):
    """
    Edit an image using OpenAI's image edit API.

    Args:
        image_path: Path to the input image (must be PNG, square, < 4MB).
        prompt: Text description of the desired edit.
        model: Model to use (default: "dall-e-2").
        mask_path: Optional path to mask image (transparent areas will be edited).
        size: Output size ("256x256", "512x512", or "1024x1024").
        output_path: Path to save the edited image. If None, saves as {original}_edited.png.

    Returns:
        Path to the saved edited image.
    """
    client = OpenAI()

    kwargs = {
        "model": model,
        "image": open(image_path, "rb"),
        "prompt": prompt,
        "size": size,
        "n": 1,
    }

    if mask_path:
        kwargs["mask"] = open(mask_path, "rb")

    response = client.images.edit(**kwargs)

    image_url = response.data[0].url

    # Download and save the image
    import requests

    img_data = requests.get(image_url).content

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_edited.png"

    with open(output_path, "wb") as f:
        f.write(img_data)

    print(f"Edited image saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or edit images using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a new image from text:
  python image_processing_openai.py --prompt "A sunset over mountains"

  # Generate with specific model and quality:
  python image_processing_openai.py --prompt "A cat" --model dall-e-3 --quality hd

  # Edit an existing image:
  python image_processing_openai.py --image_path input.png --prompt "Add a hat"

Requirements:
  - OPENAI_API_KEY environment variable must be set
  - For editing: Input image must be PNG, square, and less than 4MB
  - Mask image (if provided) must be PNG with transparent areas marking regions to edit
""",
    )
    parser.add_argument(
        "--image_path", help="Path to input image for editing (PNG, square, <4MB). If not provided, generates a new image."
    )
    parser.add_argument("--prompt", required=True, help="Text description of the image to generate or edit")
    parser.add_argument("--mask", help="Path to mask image for editing (transparent areas will be edited)")
    parser.add_argument(
        "--model",
        default="dall-e-3",
        choices=["dall-e-2", "dall-e-3"],
        help="Model to use (default: dall-e-3)",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        help="Output image size (default: 1024x1024). dall-e-2: 256x256, 512x512, 1024x1024. dall-e-3: 1024x1024, 1792x1024, 1024x1792",
    )
    parser.add_argument(
        "--quality",
        default="standard",
        choices=["standard", "hd"],
        help="Image quality for dall-e-3 (default: standard)",
    )
    parser.add_argument(
        "--output", help="Output path for image (default: generated_{timestamp}.png or {input}_edited.png)"
    )

    args = parser.parse_args()

    if args.image_path:
        # Edit mode
        edit_image(
            image_path=args.image_path,
            prompt=args.prompt,
            model=args.model,
            mask_path=args.mask,
            size=args.size,
            output_path=args.output,
        )
    else:
        # Generate mode
        generate_image(
            prompt=args.prompt,
            model=args.model,
            size=args.size,
            quality=args.quality,
            output_path=args.output,
        )
