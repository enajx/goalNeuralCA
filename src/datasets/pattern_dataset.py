import torch
import numpy as np
import random
from torch.utils.data import Dataset
import requests
import io
import PIL
from PIL import Image
import os
import hashlib
import pickle


def _get_cache_dir():
    """Get or create the pattern cache directory."""
    cache_dir = os.path.expanduser("~/.cache/goalNCA/patterns")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_cache_key(url, size):
    """Generate a cache key for the given URL and size."""
    key_string = f"{url}_{size}"
    return hashlib.md5(key_string.encode()).hexdigest()


def load_image(url, size=64):
    """Load image from URL with disk caching."""
    cache_dir = _get_cache_dir()
    cache_key = _get_cache_key(url, size)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_tensor = pickle.load(f)
            return cached_tensor
        except Exception as e:
            print(f"Warning: Failed to load from cache {cache_file}: {e}")
            # Continue to download if cache loading fails

    # Download and process image
    try:
        r = requests.get(url)
        img = PIL.Image.open(io.BytesIO(r.content))
        img.thumbnail((size, size), PIL.Image.LANCZOS)
        img = np.float32(img) / 255.0
        img[..., :3] *= img[..., 3:]
        img_tensor = torch.tensor(img).permute(2, 0, 1)

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(img_tensor, f)
        except Exception as e:
            print(f"Warning: Failed to save to cache {cache_file}: {e}")

        return img_tensor
    except Exception as e:
        print(f"Failed to load image from URL: {url}\nError: {e}")
        raise


def load_pattern(pattern, size, code=None, style="google"):
    if code is None:
        # Handle multi-character patterns by converting to hex codes
        code_points = [hex(ord(char))[2:].lower() for char in pattern]
        code = "_".join(code_points)
    if style == "google":
        url = (
            "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"
            % code
        )
    elif style == "apple":
        raise NotImplementedError("Apple pattern style not implemented yet")
    elif style == "joypixels":
        raise NotImplementedError("Twitter pattern style not implemented yet")

    return load_image(url, size)


def clear_pattern_cache():
    """Clear the pattern cache directory."""
    cache_dir = _get_cache_dir()
    if os.path.exists(cache_dir):
        import shutil

        shutil.rmtree(cache_dir)
        print(f"Pattern cache cleared: {cache_dir}")
    else:
        print("Pattern cache directory does not exist")


def emoji_to_numpy(pattern, size):
    img_tensor = load_pattern(pattern, size)
    img_np = img_tensor.numpy()
    return img_np


def load_png_from_path(path, size, embedding_dim, dtype, device):
    """Load PNG image from file path and process it like pattern loading."""
    img = PIL.Image.open(path)
    img.thumbnail((size, size), PIL.Image.LANCZOS)
    img = np.float32(img) / 255.0

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] == 3:
        alpha = np.ones((*img.shape[:2], 1), dtype=np.float32)
        img = np.concatenate([img, alpha], axis=-1)

    img[..., :3] *= img[..., 3:]
    img_tensor = torch.tensor(img, dtype=dtype, device=device).permute(2, 0, 1)

    if embedding_dim == 3 and img_tensor.shape[0] == 4:
        img_tensor = img_tensor[:3, :, :]
    elif embedding_dim == 4 and img_tensor.shape[0] == 3:
        alpha = torch.ones(1, size, size, dtype=dtype, device=device)
        img_tensor = torch.cat([img_tensor, alpha], dim=0)

    return img_tensor


def load_patterns(target_patterns, size, embedding_dim, dtype, device):
    """
    Load patterns from various sources: folder path, list of PNG paths, pattern unicode, or tensor.

    Args:
        target_patterns: Can be:
            - String ending in '/': folder path containing PNGs
            - List with items ending in '.png': list of PNG file paths
            - List of pattern unicode characters
            - torch.Tensor: pre-loaded patterns with shape [num_patterns, channels, height, width]
    """
    pattern_tensors = []
    pattern_identifiers = []

    # Handle pre-loaded tensor (e.g., from saved targets.pt)
    if isinstance(target_patterns, torch.Tensor):
        num_patterns = target_patterns.shape[0]
        for i in range(num_patterns):
            pattern_tensors.append(target_patterns[i].to(device=device, dtype=dtype))
            pattern_identifiers.append(f"pattern_{i}")
        return pattern_tensors, pattern_identifiers

    if isinstance(target_patterns, str):
        if target_patterns.endswith("/"):
            png_files = sorted([f for f in os.listdir(target_patterns) if f.endswith(".png")])
            if png_files:
                for png_file in png_files:
                    full_path = os.path.join(target_patterns, png_file)
                    img_tensor = load_png_from_path(full_path, size, embedding_dim, dtype, device)
                    pattern_tensors.append(img_tensor)
                    pattern_identifiers.append(os.path.splitext(png_file)[0])
            else:
                subfolders = sorted([d for d in os.listdir(target_patterns) if os.path.isdir(os.path.join(target_patterns, d))])
                for subfolder in subfolders:
                    subfolder_path = os.path.join(target_patterns, subfolder)
                    subfolder_pngs = [f for f in os.listdir(subfolder_path) if f.endswith(".png")]
                    if subfolder_pngs:
                        png_file = subfolder_pngs[0]
                        full_path = os.path.join(subfolder_path, png_file)
                        img_tensor = load_png_from_path(full_path, size, embedding_dim, dtype, device)
                        pattern_tensors.append(img_tensor)
                        pattern_identifiers.append(subfolder)
        else:
            raise ValueError("String target_patterns must end with '/' to indicate folder path")
    elif isinstance(target_patterns, list):
        if any(p.endswith(".png") for p in target_patterns):
            for png_path in target_patterns:
                img_tensor = load_png_from_path(png_path, size, embedding_dim, dtype, device)
                pattern_tensors.append(img_tensor)
                pattern_identifiers.append(os.path.splitext(os.path.basename(png_path))[0])
        else:
            for pattern in target_patterns:
                pattern_np = emoji_to_numpy(pattern, size)
                img_tensor = torch.tensor(pattern_np, dtype=dtype, device=device)
                if img_tensor.shape[0] != embedding_dim:
                    if embedding_dim == 3 and img_tensor.shape[0] == 4:
                        img_tensor = img_tensor[:3, :, :]
                    elif embedding_dim == 4 and img_tensor.shape[0] == 3:
                        alpha = torch.ones(1, size, size, dtype=dtype, device=device)
                        img_tensor = torch.cat([img_tensor, alpha], dim=0)
                    else:
                        raise ValueError(
                            f"Cannot convert {img_tensor.shape[0]} channels to {embedding_dim} channels"
                        )
                pattern_tensors.append(img_tensor)
                pattern_identifiers.append(pattern)
    else:
        raise ValueError(
            "target_patterns must be a string (folder path) or list (PNG paths or pattern unicode)"
        )

    return pattern_tensors, pattern_identifiers


class GoalPatternsDataset(Dataset):
    def __init__(
        self,
        size,
        seed_type,
        embedding_dim,
        extra_channels,
        one_hot_encoder,
        external_encoder_dim,
        device,
        dtype,
        target_patterns,
        seed_positions,
        no_task_encoder=False,
        space_size=None,
        boundary_condition="circular",
    ):
        """
        Args:
            size: Size of the images (size x size)
            embedding_dim: Number of embedding dimensions (3 for RGB)
            extra_channels: Number of extra channels for the NCA
            device: Device to store tensors on
            dtype: Data type for tensors
            target_patterns: Folder path, list of PNG paths, or list of pattern unicode
            no_task_encoder: If True, task encoders will be None (for unconditional growth)
            space_size: Optional canvas size to embed patterns into (space_size x space_size)
            boundary_condition: Boundary condition for embedding (only relevant if space_size is set)
        """
        self.size = size
        self.embedding_dim = embedding_dim
        self.extra_channels = extra_channels
        self.one_hot_encoder = one_hot_encoder
        self.external_encoder_dim = external_encoder_dim
        self.device = device
        self.dtype = dtype
        self.no_task_encoder = no_task_encoder
        self.space_size = space_size if space_size is not None else size
        self.boundary_condition = boundary_condition

        pattern_tensors, pattern_identifiers = load_patterns(
            target_patterns, size, embedding_dim, dtype, device
        )
        self.pattern_identifiers = pattern_identifiers
        self.num_patterns = len(pattern_identifiers)

        self.initial_seeds = []
        self.targets = []
        for i in range(self.num_patterns):
            seed_pos = seed_positions[i] if seed_positions is not None else [0.5, 0.5]
            seed_pos_x = int(seed_pos[0] * self.space_size)
            seed_pos_y = int(seed_pos[1] * self.space_size)

            # Create target with pattern embedded at seed position
            pattern_tensor = pattern_tensors[i]
            if self.space_size != self.size:
                pattern_tensor = self._embed_pattern_in_space(
                    pattern_tensor, seed_pos_x, seed_pos_y
                )
            self.targets.append(pattern_tensor)

            # Create initial seed at seed position
            seed = torch.zeros(
                self.embedding_dim + self.extra_channels,
                self.space_size,
                self.space_size,
                device=device,
                dtype=dtype,
            )
            if seed_type == "single_cell_ones_all":
                seed[:, seed_pos_x, seed_pos_y] = 1.0
            elif seed_type == "single_cell_RGB_ones_OG":
                seed[3:, seed_pos_x, seed_pos_y] = 1.0
            elif seed_type == "single_cell_RGB_ones_OG_reversed":
                seed[:3, seed_pos_x, seed_pos_y] = 1.0
            elif seed_type == "double_cell_RGB_ones_all":
                seed[
                    :,
                    seed_pos_x - 1 : seed_pos_x + 1,
                    seed_pos_y - 1 : seed_pos_y + 1,
                ] = 1.0
            elif seed_type == "double_cell_RGB_ones_OG":
                seed[
                    3:,
                    seed_pos_x - 1 : seed_pos_x + 1,
                    seed_pos_y - 1 : seed_pos_y + 1,
                ] = 1.0
            elif seed_type == "double_cell_RGB_ones_OG_reversed":
                seed[
                    :3,
                    seed_pos_x - 1 : seed_pos_x + 1,
                    seed_pos_y - 1 : seed_pos_y + 1,
                ] = 1.0
            elif seed_type == "single_cell_RGB_random":
                seed[:3, seed_pos_x, seed_pos_y] = torch.rand(4, device=device, dtype=dtype) * 2 - 1
            elif seed_type == "single_cell_random":
                seed[:, seed_pos_x, seed_pos_y] = (
                    torch.rand(self.embedding_dim + self.extra_channels, device=device, dtype=dtype)
                    * 2
                    - 1
                )
            elif seed_type == "all_cells_random":
                seed = (
                    torch.rand(
                        self.embedding_dim + self.extra_channels,
                        self.space_size,
                        self.space_size,
                        device=device,
                        dtype=dtype,
                    )
                    * 2
                    - 1
                )
            elif seed_type == "all_cells_ones":
                seed = torch.ones(
                    self.embedding_dim + self.extra_channels,
                    self.space_size,
                    self.space_size,
                    device=device,
                    dtype=dtype,
                )
            elif seed_type == "zeros":
                seed = torch.zeros(
                    self.embedding_dim + self.extra_channels,
                    self.space_size,
                    self.space_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                raise ValueError(f"Invalid seed type: {seed_type}")
            self.initial_seeds.append(seed)

        # Create external encodings
        self.task_encoders = []
        for i in range(self.num_patterns):
            if self.no_task_encoder:
                # No task encoder - append None for each pattern
                self.task_encoders.append(None)
            elif self.one_hot_encoder:
                one_hot = torch.zeros(self.external_encoder_dim, device=device, dtype=dtype)
                one_hot[i] = 1.0
                self.task_encoders.append(one_hot)
            else:
                # create random external encoding U(-1, 1) for each pattern
                external_encoding = (
                    torch.rand(self.external_encoder_dim, device=device, dtype=dtype) * 2 - 1
                )
                self.task_encoders.append(external_encoding)

    def train(self):
        """Set the dataset to training mode (no effect for this dataset, but needed for compatibility)."""
        return self

    def eval(self):
        """Set the dataset to evaluation mode (no effect for this dataset, but needed for compatibility)."""
        return self

    def __len__(self):
        return self.num_patterns

    def __getitem__(self, idx):
        return self.initial_seeds[idx], self.task_encoders[idx], self.targets[idx]

    def _embed_pattern_in_space(self, pattern_tensor, center_x, center_y):
        """
        Embed the pattern tensor into a larger space at the specified center position.

        Args:
            pattern_tensor: Tensor of shape [C, size, size]
            center_x: X coordinate of center position in space coordinates
            center_y: Y coordinate of center position in space coordinates

        Returns:
            Tensor of shape [C, space_size, space_size] with pattern embedded
        """
        # If space size equals pattern size, return as is
        if self.space_size == self.size:
            return pattern_tensor

        space_tensor = torch.zeros(
            self.embedding_dim,
            self.space_size,
            self.space_size,
            dtype=self.dtype,
            device=self.device,
        )

        half_pattern = self.size // 2

        if self.boundary_condition == "circular":
            # Handle circular boundary conditions with proper wrapping
            for pattern_y in range(self.size):
                for pattern_x in range(self.size):
                    space_x = (center_x - half_pattern + pattern_x) % self.space_size
                    space_y = (center_y - half_pattern + pattern_y) % self.space_size
                    space_tensor[:, space_y, space_x] = pattern_tensor[:, pattern_y, pattern_x]
        else:
            # Calculate pattern placement bounds
            start_x = max(0, center_x - half_pattern)
            end_x = min(self.space_size, center_x - half_pattern + self.size)
            start_y = max(0, center_y - half_pattern)
            end_y = min(self.space_size, center_y - half_pattern + self.size)

            # Calculate corresponding pattern bounds
            pattern_start_x = max(0, half_pattern - center_x)
            pattern_end_x = pattern_start_x + (end_x - start_x)
            pattern_start_y = max(0, half_pattern - center_y)
            pattern_end_y = pattern_start_y + (end_y - start_y)

            # Place pattern in space
            space_tensor[:, start_y:end_y, start_x:end_x] = pattern_tensor[
                :, pattern_start_y:pattern_end_y, pattern_start_x:pattern_end_x
            ]

        return space_tensor


class GoalPatternsTransformDataset(Dataset):
    def __init__(
        self,
        pattern_size,
        space_size,
        embedding_dim,
        extra_channels,
        device,
        dtype,
        target_patterns,
        transformation_amount,
        transformation_type,
        boundary_condition,
        num_samples_per_transformation,
        domain_noise,
        batch_size,
    ):
        """
        Dataset for training NCAs to perform incremental image transformations.

        Args:
            pattern_size: Size of the pattern itself (pattern_size x pattern_size)
            space_size: Size of the space/canvas (space_size x space_size) where pattern is embedded
            embedding_dim: Number of embedding dimensions (3 for RGB)
            extra_channels: Number of extra channels for the NCA
            device: Device to store tensors on
            dtype: Data type for tensors
            target_patterns: Folder path, list of PNG paths, or list of pattern unicode
            transformation_amount: Amount of transformation in degrees for rotation, pixels for translation
            transformation_type: Type of transformation - "rotation" or "translation"
            boundary_condition: Boundary condition type - "zeros", "circular", "reflect", "replicate"
            num_samples_per_transformation: Number of random starting positions for translation task
            batch_size: Batch size
        """
        self.pattern_size = pattern_size
        self.space_size = space_size
        self.embedding_dim = embedding_dim
        self.extra_channels = extra_channels
        self.device = device
        self.dtype = dtype
        self.transformation_amount = transformation_amount
        self.transformation_type = transformation_type
        self.boundary_condition = boundary_condition
        self.num_samples_per_transformation = num_samples_per_transformation
        self.domain_noise = domain_noise
        self.training = True
        self.batch_size = batch_size

        pattern_tensors, pattern_identifiers = load_patterns(
            target_patterns, pattern_size, embedding_dim, dtype, device
        )
        self.pattern_identifiers = pattern_identifiers
        self.pattern_tensors = pattern_tensors
        self.num_patterns = len(pattern_identifiers)

        # Generate all samples
        self.samples = []
        self._generate_samples()

        print(f"GoalPatternsTransformDataset created with {len(self.samples)} samples")
        print(f"Using {self.num_patterns} patterns, {transformation_type} transformations")
        print(f"Boundary condition: {self.boundary_condition}")
        if hasattr(self, "boundary_condition") and self.transformation_type == "translation":
            if self.boundary_condition == "circular":
                print(f"Pattern positioning: full space utilization (periodic boundaries)")
            else:
                margin = int(1.5 * self.pattern_size)
                print(
                    f"Pattern positioning: {margin}px margin (1.5x pattern_size) from edges to avoid boundary artifacts"
                )
        print(f"Expected samples: {self.num_patterns} × {transformation_type} transformations\n")

    def _rotate_image_tensor(self, img_tensor, angle_degrees):
        """
        Rotate an image tensor by the given angle in degrees.

        Args:
            img_tensor: Tensor of shape [C, H, W]
            angle_degrees: Rotation angle in degrees

        Returns:
            Rotated tensor with same shape, padded with zeros/transparent pixels
        """
        # Convert to PIL Image for rotation
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]

        # Handle RGBA vs RGB
        if img_np.shape[2] == 4:
            # RGBA
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="RGBA")
        else:
            # RGB - add alpha channel
            img_rgba = np.ones((img_np.shape[0], img_np.shape[1], 4))
            img_rgba[:, :, :3] = img_np
            img_pil = Image.fromarray((img_rgba * 255).astype(np.uint8), mode="RGBA")

        # Rotate with transparent background
        rotated_pil = img_pil.rotate(angle_degrees, fillcolor=(0, 0, 0, 0))

        # Convert back to tensor
        rotated_np = np.array(rotated_pil).astype(np.float32) / 255.0

        # Ensure we have the right number of channels to match embedding_dim
        if rotated_np.shape[2] == 4 and self.embedding_dim == 3:
            # Remove alpha channel if we want RGB
            rotated_np = rotated_np[:, :, :3]
        elif rotated_np.shape[2] == 3 and self.embedding_dim == 4:
            # Add alpha channel if we want RGBA
            alpha = np.ones((rotated_np.shape[0], rotated_np.shape[1], 1))
            rotated_np = np.concatenate([rotated_np, alpha], axis=2)

        # Convert back to tensor format [C, H, W]
        rotated_tensor = torch.tensor(
            rotated_np.transpose(2, 0, 1), dtype=self.dtype, device=self.device
        )

        return rotated_tensor

    def _embed_pattern_in_space(self, pattern_tensor, center_x, center_y):
        """
        Embed the pattern tensor into a larger space at the specified center position.

        Args:
            pattern_tensor: Tensor of shape [C, pattern_size, pattern_size]
            center_x: X coordinate of center position in space coordinates
            center_y: Y coordinate of center position in space coordinates

        Returns:
            Tensor of shape [C, space_size, space_size] with pattern embedded
        """
        # Create empty space
        space_tensor = torch.zeros(
            self.embedding_dim,
            self.space_size,
            self.space_size,
            dtype=self.dtype,
            device=self.device,
        )

        half_pattern = self.pattern_size // 2

        if self.boundary_condition == "circular":
            # Handle circular boundary conditions with proper wrapping
            # For circular boundaries, we need to wrap around the space
            for pattern_y in range(self.pattern_size):
                for pattern_x in range(self.pattern_size):
                    # Calculate position in space with wrapping
                    space_x = (center_x - half_pattern + pattern_x) % self.space_size
                    space_y = (center_y - half_pattern + pattern_y) % self.space_size

                    # Copy pattern pixel to wrapped position
                    space_tensor[:, space_y, space_x] = pattern_tensor[:, pattern_y, pattern_x]
        else:
            # Calculate pattern placement bounds
            start_x = max(0, center_x - half_pattern)
            end_x = min(self.space_size, center_x - half_pattern + self.pattern_size)
            start_y = max(0, center_y - half_pattern)
            end_y = min(self.space_size, center_y - half_pattern + self.pattern_size)

            # Calculate corresponding pattern bounds
            pattern_start_x = max(0, half_pattern - center_x)
            pattern_end_x = pattern_start_x + (end_x - start_x)
            pattern_start_y = max(0, half_pattern - center_y)
            pattern_end_y = pattern_start_y + (end_y - start_y)

            # Place pattern in space
            space_tensor[:, start_y:end_y, start_x:end_x] = pattern_tensor[
                :, pattern_start_y:pattern_end_y, pattern_start_x:pattern_end_x
            ]

        return space_tensor

    def _translate_image_tensor(self, img_tensor, dx, dy):
        """
        Translate an image tensor by dx, dy pixels with periodic boundary conditions.

        Args:
            img_tensor: Tensor of shape [C, H, W]
            dx: Horizontal translation (positive = right)
            dy: Vertical translation (positive = down)

        Returns:
            Translated tensor with same shape using periodic boundary conditions
        """
        # Use torch.roll for periodic boundary conditions
        # Roll along height dimension (dim=-2) for dy
        # Roll along width dimension (dim=-1) for dx
        translated_tensor = torch.roll(img_tensor, shifts=(dy, dx), dims=(-2, -1))
        return translated_tensor

    def _generate_samples(self):
        """Generate all samples for the dataset."""
        if self.transformation_type == "rotation":
            # Rotation transformations
            transformation_types = [
                (
                    "clockwise",
                    torch.tensor([1.0, 0.0], dtype=self.dtype, device=self.device),
                    self.transformation_amount,
                ),
                (
                    "anticlockwise",
                    torch.tensor([0.0, 1.0], dtype=self.dtype, device=self.device),
                    -self.transformation_amount,
                ),
                ("no_change", torch.tensor([0.0, 0.0], dtype=self.dtype, device=self.device), 0),
            ]

            for pattern_idx, pattern_tensor in enumerate(self.pattern_tensors):
                for angle in range(360):  # 0 to 359 degrees
                    # Create input image at current angle
                    input_image = self._rotate_image_tensor(pattern_tensor, angle)

                    # Pad input image with extra channels if needed
                    if self.extra_channels > 0:
                        extra_padding = torch.zeros(
                            self.extra_channels,
                            self.pattern_size,
                            self.pattern_size,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        input_image = torch.cat([input_image, extra_padding], dim=0)

                    for transform_name, task_encoder, angle_delta in transformation_types:
                        # Create target image
                        target_angle = (angle + angle_delta) % 360
                        target_image = self._rotate_image_tensor(pattern_tensor, target_angle)

                        # Store sample
                        sample = {
                            "input_image": input_image,
                            "task_encoder": task_encoder,
                            "target_image": target_image,
                            "pattern_idx": pattern_idx,
                            "input_angle": angle,
                            "target_angle": target_angle,
                            "transformation": transform_name,
                        }
                        self.samples.append(sample)

        elif self.transformation_type == "translation":
            # Translation transformations
            transformation_types = [
                (
                    "up",
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
                    (0, -self.transformation_amount),
                ),
                (
                    "right",
                    torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
                    (self.transformation_amount, 0),
                ),
                (
                    "down",
                    torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=self.dtype, device=self.device),
                    (0, self.transformation_amount),
                ),
                (
                    "left",
                    torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device),
                    (-self.transformation_amount, 0),
                ),
                (
                    "no_movement",
                    torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
                    (0, 0),
                ),
            ]

            # Generate samples with random starting positions
            num_samples_per_pattern = self.num_samples_per_transformation

            for pattern_idx, pattern_tensor in enumerate(self.pattern_tensors):
                for t_idx, (transform_name, task_encoder, (dx, dy)) in enumerate(
                    transformation_types
                ):
                    for _ in range(num_samples_per_pattern):
                        # Calculate positioning constraints based on boundary conditions
                        if self.boundary_condition == "circular":
                            min_pos = 0
                            max_pos = self.space_size - 1
                        else:
                            margin = 3
                            min_pos = self.pattern_size // 2 + margin
                            max_pos = self.space_size - self.pattern_size // 2 - 1 - margin

                        if min_pos > max_pos:
                            if self.boundary_condition == "circular":
                                required_space = 1
                            else:
                                required_space = self.pattern_size + 2 * 3
                            raise ValueError(
                                f"Cannot fit pattern with safety margin in space. "
                                f"Need space_size >= {required_space} for boundary_condition='{self.boundary_condition}'. "
                                f"Current: space_size={self.space_size}, pattern_size={self.pattern_size}, "
                                f"boundary_condition={self.boundary_condition}"
                            )

                        # position always in the center
                        center_x = self.space_size // 2
                        center_y = self.space_size // 2

                        input_image = self._embed_pattern_in_space(
                            pattern_tensor, center_x, center_y
                        )

                        if self.extra_channels > 0:
                            extra_padding = torch.zeros(
                                self.extra_channels,
                                self.space_size,
                                self.space_size,
                                dtype=self.dtype,
                                device=self.device,
                            )
                            input_image = torch.cat([input_image, extra_padding], dim=0)

                        if self.boundary_condition == "circular":
                            target_center_x = (center_x + dx) % self.space_size
                            target_center_y = (center_y + dy) % self.space_size
                        else:
                            target_center_x = center_x + dx
                            target_center_y = center_y + dy

                        target_image = self._embed_pattern_in_space(
                            pattern_tensor, target_center_x, target_center_y
                        )
                        # target_image = self._translate_image_tensor(input_image, dx, dy)

                        sample = {
                            "input_image": input_image,
                            "task_encoder": task_encoder,
                            "target_image": target_image,
                            "pattern_idx": pattern_idx,
                            "center_x": center_x,
                            "center_y": center_y,
                            "target_center_x": target_center_x,
                            "target_center_y": target_center_y,
                            "transformation": transform_name,
                            "dx": dx,
                            "dy": dy,
                        }
                        self.samples.append(sample)
        else:
            raise ValueError(f"Unknown transformation_type: {self.transformation_type}")

    def train(self):
        """Set the dataset to training mode (applies domain noise)."""
        self.training = True
        return self

    def eval(self):
        """Set the dataset to evaluation mode (no domain noise)."""
        self.training = False
        return self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.domain_noise > 0 and self.training:
            input_image = sample["input_image"] + self.domain_noise * torch.randn_like(
                sample["input_image"]
            )
            input_image = torch.clamp(input_image, 0, 1)
        else:
            input_image = sample["input_image"]
        return input_image, sample["task_encoder"], sample["target_image"]

    def get_sample_info(self, idx):
        """Get additional information about a sample for debugging/visualization."""
        sample = self.samples[idx]
        info = {
            "pattern": self.pattern_identifiers[sample["pattern_idx"]],
            "transformation": sample["transformation"],
        }

        if self.transformation_type == "rotation":
            info.update(
                {
                    "input_angle": sample["input_angle"],
                    "target_angle": sample["target_angle"],
                }
            )
        elif self.transformation_type == "translation":
            info.update(
                {
                    "center_x": sample["center_x"],
                    "center_y": sample["center_y"],
                    "target_center_x": sample["target_center_x"],
                    "target_center_y": sample["target_center_y"],
                    "dx": sample["dx"],
                    "dy": sample["dy"],
                }
            )
        return info


class GoalPatternsMorphingDataset(Dataset):
    def __init__(
        self,
        size,
        embedding_dim,
        extra_channels,
        one_hot_encoder,
        external_encoder_dim,
        device,
        dtype,
        target_patterns,
        domain_noise,
    ):
        """
        Dataset for pattern morphing that generates all possible combinations including identity cases.

        For N patterns, this creates N×N samples where each pattern can transform to every other pattern
        including itself. Identity cases use the proper one-hot task encoding for the target pattern,
        ensuring the model learns to reproduce the pattern when given the correct task signal.

        Args:
            size: Size of the images (size x size)
            embedding_dim: Number of embedding dimensions (3 for RGB)
            extra_channels: Number of extra channels for the NCA
            one_hot_encoder: Whether to use one-hot encoding for task encoders
            external_encoder_dim: Dimension of external encoder (should be N for one-hot)
            device: Device to store tensors on
            dtype: Data type for tensors
            target_patterns: Folder path, list of PNG paths, or list of pattern unicode
        """
        self.size = size
        self.embedding_dim = embedding_dim
        self.extra_channels = extra_channels
        self.one_hot_encoder = one_hot_encoder
        self.external_encoder_dim = external_encoder_dim
        self.device = device
        self.dtype = dtype
        self.domain_noise = domain_noise
        self.training = False

        pattern_tensors, pattern_identifiers = load_patterns(
            target_patterns, size, embedding_dim, dtype, device
        )
        self.pattern_identifiers = pattern_identifiers
        self.pattern_tensors = pattern_tensors
        self.num_patterns = len(pattern_identifiers)

        # Generate all possible combinations
        self.samples = []
        self._generate_all_combinations()

        print(f"GoalPatternsMorphingDataset created with {len(self.samples)} samples")
        print(
            f"Using {self.num_patterns} patterns with {self.num_patterns}×{self.num_patterns} combinations (including identity cases)"
        )
        print(f"Selected patterns: {self.pattern_identifiers}")

    def _generate_all_combinations(self):
        """Generate all possible pattern-to-pattern combinations, including identity cases with proper task encoding."""
        for source_idx in range(self.num_patterns):
            for target_idx in range(self.num_patterns):
                # Create initial seed from source pattern
                source_pattern = self.pattern_tensors[source_idx]
                seed = torch.zeros(
                    self.embedding_dim + self.extra_channels,
                    self.size,
                    self.size,
                    device=self.device,
                    dtype=self.dtype,
                )

                # Start with the source pattern for the embedding dimensions
                seed[: self.embedding_dim, :, :] = source_pattern

                # Initialize extra channels with zeros if needed
                if self.extra_channels > 0:
                    seed[self.embedding_dim :, :, :] = 0.0

                # Create task encoder (one-hot encoding for target)
                # This includes proper encoding for identity cases
                if self.one_hot_encoder:
                    one_hot = torch.zeros(
                        self.external_encoder_dim, device=self.device, dtype=self.dtype
                    )
                    one_hot[target_idx] = 1.0
                    task_encoder = one_hot
                else:
                    # For non-one-hot, create consistent encoding for each target
                    # Use a deterministic approach based on target index
                    task_encoder = torch.zeros(
                        self.external_encoder_dim, device=self.device, dtype=self.dtype
                    )
                    if self.external_encoder_dim > target_idx:
                        task_encoder[target_idx] = 1.0
                    else:
                        # If external_encoder_dim is smaller, use modulo
                        task_encoder[target_idx % self.external_encoder_dim] = 1.0

                # Get target pattern
                target_pattern = self.pattern_tensors[target_idx]

                # Store sample with metadata
                sample = {
                    "initial_seed": seed,
                    "task_encoder": task_encoder,
                    "target": target_pattern,
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "source_pattern": self.pattern_identifiers[source_idx],
                    "target_pattern": self.pattern_identifiers[target_idx],
                }
                self.samples.append(sample)

    def train(self):
        """Set the dataset to training mode (applies domain noise)."""
        self.training = True
        return self

    def eval(self):
        """Set the dataset to evaluation mode (no domain noise)."""
        self.training = False
        return self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.domain_noise > 0 and self.training:
            initial_seed = sample["initial_seed"] + self.domain_noise * torch.randn_like(
                sample["initial_seed"]
            )
            # clip to 0-1
            initial_seed = torch.clamp(initial_seed, 0, 1)
        else:
            initial_seed = sample["initial_seed"]
        return initial_seed, sample["task_encoder"], sample["target"]

    def get_sample_info(self, idx):
        """Get additional information about a sample for debugging/visualization."""
        sample = self.samples[idx]
        return {
            "source_pattern": sample["source_pattern"],
            "target_pattern": sample["target_pattern"],
            "source_idx": sample["source_idx"],
            "target_idx": sample["target_idx"],
            "transformation": f"{sample['source_pattern']} → {sample['target_pattern']}",
        }

    def get_combinations_matrix(self):
        """
        Return a matrix showing all source→target combinations.
        All combinations including identity cases are now included in the dataset.
        Useful for debugging and visualization.
        """
        matrix = []
        for source_idx in range(self.num_patterns):
            row = []
            for target_idx in range(self.num_patterns):
                # Calculate sample index using the standard N×N indexing
                sample_idx = source_idx * self.num_patterns + target_idx
                info = self.get_sample_info(sample_idx)
                row.append(info["transformation"])
            matrix.append(row)
        return matrix


class GoalPatternsTrajectoryDataset(Dataset):
    def __init__(
        self,
        pattern_size,
        space_size,
        embedding_dim,
        extra_channels,
        device,
        dtype,
        target_patterns,
        nca_steps,
        boundary_condition,
        num_samples_per_transformation,
        domain_noise,
    ):
        """
        Dataset for training NCAs to perform multi-step trajectories.

        Args:
            pattern_size: Size of the pattern itself (pattern_size x pattern_size)
            space_size: Size of the space/canvas (space_size x space_size) where pattern is embedded
            embedding_dim: Number of embedding dimensions (3 for RGB)
            extra_channels: Number of extra channels for the NCA
            device: Device to store tensors on
            dtype: Data type for tensors
            target_patterns: Folder path, list of PNG paths, or list of pattern unicode
            nca_steps: Int for fixed trajectory length, or list [min_length, max_length] for variable trajectory length sampling
            boundary_condition: Boundary condition type - "zeros", "circular", "reflect", "replicate"
            num_samples_per_transformation: Number of trajectory samples per pattern
            domain_noise: Amount of domain noise to add during training
        """
        self.pattern_size = pattern_size
        self.space_size = space_size
        self.embedding_dim = embedding_dim
        self.extra_channels = extra_channels
        self.device = device
        self.dtype = dtype
        self.boundary_condition = boundary_condition
        self.num_samples_per_transformation = num_samples_per_transformation
        self.domain_noise = domain_noise
        self.training = True

        # Handle nca_steps as either int (fixed length) or [min_length, max_length] (variable length)
        if isinstance(nca_steps, int):
            # Fixed trajectory length
            self.nca_steps_min = nca_steps
            self.nca_steps_max = nca_steps
        elif isinstance(nca_steps, (list, tuple)) and len(nca_steps) == 2:
            # Variable trajectory length
            self.nca_steps_min, self.nca_steps_max = nca_steps
        else:
            raise ValueError(
                "nca_steps must be either an int (fixed length) or a list/tuple of [min_length, max_length] (variable length)"
            )

        pattern_tensors, pattern_identifiers = load_patterns(
            target_patterns, pattern_size, embedding_dim, dtype, device
        )
        self.pattern_identifiers = pattern_identifiers
        self.pattern_tensors = pattern_tensors
        self.num_patterns = len(pattern_identifiers)

        # Don't pre-generate trajectories - generate them on-demand for better generalization
        # Dataset size is num_patterns × num_samples_per_transformation
        self.dataset_size = self.num_patterns * self.num_samples_per_transformation

        print(f"GoalPatternsTrajectoryDataset created with {self.dataset_size} samples")
        if self.nca_steps_min == self.nca_steps_max:
            print(
                f"Using {self.num_patterns} patterns with fixed trajectory length {self.nca_steps_min}"
            )
        else:
            print(
                f"Using {self.num_patterns} patterns with trajectory lengths [{self.nca_steps_min}, {self.nca_steps_max}]"
            )
        print(f"Boundary condition: {self.boundary_condition}")

    def _generate_trajectory(self):
        """Generate a random trajectory sequence.

        Returns:
            List of (direction_name, task_encoder, (dx, dy)) tuples
        """
        # Sample random trajectory length
        trajectory_length = random.randint(self.nca_steps_min, self.nca_steps_max)

        # Define all possible transformations (1 pixel movements)
        transformation_types = [
            (
                "up",
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
                (0, -1),
            ),
            (
                "right",
                torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
                (1, 0),
            ),
            (
                "down",
                torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=self.dtype, device=self.device),
                (0, 1),
            ),
            (
                "left",
                torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device),
                (-1, 0),
            ),
            (
                "no_movement",
                torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
                (0, 0),
            ),
        ]

        # Generate random sequence of directions
        trajectory = []
        for _ in range(trajectory_length):
            # Randomly select one of the 5 possible transformations
            direction_name, task_encoder, (dx, dy) = random.choice(transformation_types)
            trajectory.append((direction_name, task_encoder, (dx, dy)))

        return trajectory

    def _execute_trajectory(self, pattern_tensor, trajectory):
        """Execute a trajectory sequence starting from center position.

        Args:
            pattern_tensor: Tensor of shape [C, pattern_size, pattern_size]
            trajectory: List of (direction_name, task_encoder, (dx, dy)) tuples

        Returns:
            List of target images (tensors of shape [C, space_size, space_size])
        """
        # Start from center position
        current_x = self.space_size // 2
        current_y = self.space_size // 2

        target_images = []

        for direction_name, task_encoder, (dx, dy) in trajectory:
            # Calculate new position after applying transformation
            new_x = current_x + dx
            new_y = current_y + dy

            # Handle boundary conditions
            if self.boundary_condition == "circular":
                # Use periodic boundary conditions
                new_x = new_x % self.space_size
                new_y = new_y % self.space_size
            else:
                # For non-circular boundary conditions, clamp to valid range
                # Use same logic as existing dataset - ensure pattern fits in space
                margin = 3  # Same margin as existing dataset
                min_pos = self.pattern_size // 2 + margin
                max_pos = self.space_size - self.pattern_size // 2 - 1 - margin

                new_x = max(min_pos, min(max_pos, new_x))
                new_y = max(min_pos, min(max_pos, new_y))

            # Create target image with pattern at new position
            target_image = self._embed_pattern_in_space(pattern_tensor, new_x, new_y)
            target_images.append(target_image)

            # Update current position for next step
            current_x = new_x
            current_y = new_y

        return target_images

    def _embed_pattern_in_space(self, pattern_tensor, center_x, center_y):
        """
        Embed the pattern tensor into a larger space at the specified center position.
        Reuses the logic from GoalPatternsTransformDataset.
        """
        # Create empty space
        space_tensor = torch.zeros(
            self.embedding_dim,
            self.space_size,
            self.space_size,
            dtype=self.dtype,
            device=self.device,
        )

        half_pattern = self.pattern_size // 2

        if self.boundary_condition == "circular":
            # Handle circular boundary conditions with proper wrapping
            for pattern_y in range(self.pattern_size):
                for pattern_x in range(self.pattern_size):
                    # Calculate position in space with wrapping
                    space_x = (center_x - half_pattern + pattern_x) % self.space_size
                    space_y = (center_y - half_pattern + pattern_y) % self.space_size

                    # Copy pattern pixel to wrapped position
                    space_tensor[:, space_y, space_x] = pattern_tensor[:, pattern_y, pattern_x]
        else:
            # Handle non-circular boundary conditions with clipping
            # Calculate pattern placement bounds
            start_x = max(0, center_x - half_pattern)
            end_x = min(self.space_size, center_x - half_pattern + self.pattern_size)
            start_y = max(0, center_y - half_pattern)
            end_y = min(self.space_size, center_y - half_pattern + self.pattern_size)

            # Calculate corresponding pattern bounds
            pattern_start_x = max(0, half_pattern - center_x)
            pattern_end_x = pattern_start_x + (end_x - start_x)
            pattern_start_y = max(0, half_pattern - center_y)
            pattern_end_y = pattern_start_y + (end_y - start_y)

            # Place pattern in space
            space_tensor[:, start_y:end_y, start_x:end_x] = pattern_tensor[
                :, pattern_start_y:pattern_end_y, pattern_start_x:pattern_end_x
            ]

        return space_tensor

    def train(self):
        """Set the dataset to training mode (applies domain noise)."""
        self.training = True
        return self

    def eval(self):
        """Set the dataset to evaluation mode (no domain noise)."""
        self.training = False
        return self

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Get a trajectory sample.

        Returns:
            input_image: [channels, H, W] - Pattern at center position
            task_encoders: [trajectory_length, 4] - Sequence of directional commands
            target_images: [trajectory_length, 3, H, W] - Sequence of target positions
        """
        # Determine which pattern this sample corresponds to
        pattern_idx = idx % self.num_patterns
        pattern_tensor = self.pattern_tensors[pattern_idx]

        # Generate a fresh random trajectory for this sample
        trajectory = self._generate_trajectory()

        # Create initial input image (pattern at center)
        center_x = self.space_size // 2
        center_y = self.space_size // 2
        input_image = self._embed_pattern_in_space(pattern_tensor, center_x, center_y)

        # Add extra channels if needed
        if self.extra_channels > 0:
            extra_padding = torch.zeros(
                self.extra_channels,
                self.space_size,
                self.space_size,
                dtype=self.dtype,
                device=self.device,
            )
            input_image = torch.cat([input_image, extra_padding], dim=0)

        # Add domain noise if in training mode
        if self.domain_noise > 0 and self.training:
            input_image = input_image + self.domain_noise * torch.randn_like(input_image)
            input_image = torch.clamp(input_image, 0, 1)

        # Execute trajectory to get sequence of target images
        target_images = self._execute_trajectory(pattern_tensor, trajectory)

        # Extract task encoders from trajectory
        task_encoders = [step[1] for step in trajectory]  # Get task_encoder from each step

        # Convert to tensors
        task_encoders = torch.stack(task_encoders, dim=0)  # [trajectory_length, 4]
        target_images = torch.stack(target_images, dim=0)  # [trajectory_length, 3, H, W]

        return input_image, task_encoders, target_images

    @staticmethod
    def trajectory_collate_fn(batch):
        """Custom collate function to handle variable-length trajectories."""
        # Separate the batch components
        input_images = [item[0] for item in batch]
        task_encoders_list = [item[1] for item in batch]
        target_images_list = [item[2] for item in batch]

        # Stack input images (they're all the same size)
        input_images = torch.stack(input_images, dim=0)

        # Get sequence lengths for each sample
        seq_lengths = [seq.shape[0] for seq in task_encoders_list]
        max_seq_len = max(seq_lengths)

        # Pad task encoders to max length
        batch_size = len(task_encoders_list)
        padded_task_encoders = torch.zeros(
            batch_size,
            max_seq_len,
            4,
            dtype=task_encoders_list[0].dtype,
            device=task_encoders_list[0].device,
        )

        for i, seq in enumerate(task_encoders_list):
            seq_len = seq.shape[0]
            padded_task_encoders[i, :seq_len] = seq

        # Pad target images to max length
        example_target = target_images_list[0]
        channels, height, width = (
            example_target.shape[1],
            example_target.shape[2],
            example_target.shape[3],
        )
        padded_target_images = torch.zeros(
            batch_size,
            max_seq_len,
            channels,
            height,
            width,
            dtype=example_target.dtype,
            device=example_target.device,
        )

        for i, seq in enumerate(target_images_list):
            seq_len = seq.shape[0]
            padded_target_images[i, :seq_len] = seq

        return input_images, padded_task_encoders, padded_target_images, seq_lengths

    def get_sample_info(self, idx):
        """Get additional information about a trajectory sample for debugging/visualization."""
        # Determine which pattern this sample corresponds to
        pattern_idx = idx % self.num_patterns

        # Generate a fresh trajectory to get info (note: this will be different from actual __getitem__)
        trajectory = self._generate_trajectory()

        return {
            "pattern": self.pattern_identifiers[pattern_idx],
            "trajectory_length": len(trajectory),
            "trajectory": [step[0] for step in trajectory],  # Just direction names
            "sample_idx": idx,
            "pattern_idx": pattern_idx,
        }


if __name__ == "__main__":
    # Test morphing dataset
    print("Testing GoalPatternsMorphingDataset:")
    morphing_dataset = GoalPatternsMorphingDataset(
        size=20,
        embedding_dim=3,
        extra_channels=0,
        one_hot_encoder=True,
        external_encoder_dim=4,
        device="cpu",
        dtype=torch.float32,
        target_patterns=["😍", "😂", "🐼", "🐻"],
    )
    print(f"Morphing dataset size: {len(morphing_dataset)} samples")

    # Show combinations matrix
    print("\nCombinations matrix:")
    combinations_matrix = morphing_dataset.get_combinations_matrix()
    for row in combinations_matrix:
        print("  " + " | ".join(row))

    # Test a few samples
    print("\nSample details:")
    for i in range(min(8, len(morphing_dataset))):
        info = morphing_dataset.get_sample_info(i)
        print(f"Sample {i}: {info['transformation']}")

    # Test original dataset
    # dataset = GoalPatternsDataset(
    #     num_patterns=10,
    #     size=20,
    #     seed_type="single_cell_random",
    #     embedding_dim=3,
    #     extra_channels=0,
    #     one_hot_encoder=True,
    #     external_encoder_dim=10,
    #     device="cpu",
    #     dtype=torch.float32,
    # )Patterns
    # print("Original dataset sample:", dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape)

    # Test new movement dataset with limited patterns
    movement_dataset = GoalPatternsTransformDataset(
        pattern_size=20,
        space_size=20,
        embedding_dim=3,
        extra_channels=0,
        device="cpu",
        dtype=torch.float32,
        target_patterns=["😍", "😂"],
        transformation_amount=1,
        transformation_type="rotation",
        boundary_condition="circular",
        batch_size=10,
    )
    print(
        "Movement dataset sample:",
        movement_dataset[0][0].shape,
        movement_dataset[0][1].shape,
        movement_dataset[0][2].shape,
    )
    print(f"Movement dataset size: {len(movement_dataset)} samples")

    # Test a few samples to verify transformations
    for i in range(9):  # 3 patterns × 3 transformations = 9 samples for first angle
        info = movement_dataset.get_sample_info(i)
        print(
            f"Sample {i}: {info['pattern']} - {info['transformation']} - {info['input_angle']}° → {info['target_angle']}°"
        )

    # Test with all patterns
    print("\nTesting with all patterns:")
    full_dataset = GoalPatternsTransformDataset(
        pattern_size=20,
        space_size=20,
        embedding_dim=3,
        extra_channels=0,
        device="cpu",
        dtype=torch.float32,
        target_patterns=["😍", "😂", "🐼", "🐻"],
        transformation_amount=1,
        transformation_type="rotation",
        boundary_condition="circular",
        batch_size=10,
    )
    print(f"Full dataset size: {len(full_dataset)} samples")
