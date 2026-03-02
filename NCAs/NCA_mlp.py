import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torch.nn.functional import conv2d
import numpy as np
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


from NCAs.utils import get_kernels, compute_alive_mask
from NCAs.utils import seed_python_numpy_torch_cuda
from NCAs.visualisation_functions import (
    visualize_all_patterns_grid,
    create_animation_grid,
)
from src.datasets.pattern_dataset import GoalPatternsDataset

torch.set_printoptions(precision=4)
# torch.autograd.set_grad_enabled(False)
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)


def apply_custom_padding(x, custom_padding, thickness, embedding_dim):
    """Apply custom constant padding only to hidden channels (beyond embedding_dim).
    custom_padding: [top, right, bottom, left], thickness: number of pixels"""
    top, right, bottom, left = custom_padding
    t = thickness
    x_padded = F.pad(x, (t, t, t, t), mode="constant", value=0)
    x_padded[:, embedding_dim:, :t, :] = top
    x_padded[:, embedding_dim:, -t:, :] = bottom
    x_padded[:, embedding_dim:, t:-t, :t] = left
    x_padded[:, embedding_dim:, t:-t, -t:] = right
    return x_padded


class NCA_mlp(torch.nn.Module):
    def __init__(
        self,
        num_input_channels,
        num_external_channels,
        num_output_conv_features,  # number of output features of the convolutional layers, only used if convolution_mode == "mixing_features"
        num_conv_layers,
        bias,
        activation_conv,
        activation_fc,
        activation_last,
        hidden_dim_mlp,
        fixed_kernels,  # kernels are fixed, not learned. only used if convolution_mode == "share_kernels_across_channels"
        stochastic_update_ratio,
        convolution_mode,  # "share_kernels_across_channels", "mixing_features" or "one_kernel_per_channel"
        num_kernels,  # only used if convolution_mode == "share_kernels_across_channels"
        custom_kernels,  # kernels provided by user if convolution_mode == "share_kernels_across_channels"
        width_kernel,  # only applies if not fixed_kernels
        additive_update,
        merge_ext,  # if true merges taks vector with post conv state
        dropout,
        boundary_condition,  # "zeros", "circular", "reflect", "replicate"
        alive_mask_goal,  # if true, only apply external input to alive cells()
        alive_threshold,  # threshold for considering a cell alive
        isotropic_only,  # only use isotropic kernels
        extra_kernels,  # use extra kernels
        custom_padding=None,  # [top, right, bottom, left] or None
        custom_padding_thickness=None,  # number of pixels of custom padding border
        embedding_dim=None,  # number of visible channels (RGBA), needed for custom_padding
        alive_channel=3,  # channel index to check for alive cells
    ):
        super(NCA_mlp, self).__init__()
        self.activation_conv = activation_conv
        self.activation_fc = activation_fc
        self.activation_last = activation_last
        self.boundary_condition = boundary_condition
        self.custom_padding = custom_padding
        self.custom_padding_thickness = custom_padding_thickness
        self.embedding_dim = embedding_dim
        if custom_padding is not None and custom_padding_thickness is None:
            raise ValueError("custom_padding_thickness is required when custom_padding is set")
        if custom_padding is not None and convolution_mode != "share_kernels_across_channels":
            raise ValueError("custom_padding is only implemented for 'share_kernels_across_channels' convolution mode")
        # To avoid double activations, raise error if activation_last is not None, identity or clamp
        # print(activation_last)
        if not additive_update:
            if (
                activation_last is not None
                and not isinstance(activation_last, torch.nn.Identity)
                and not isinstance(activation_last, torch.nn.PReLU)
                and not isinstance(activation_last, torch.nn.LeakyReLU)
                and not isinstance(activation_last, torch.nn.ReLU)
                and not (
                    callable(activation_last)
                    and hasattr(activation_last, "__code__")
                    and "clamp" in activation_last.__code__.co_names
                )
                and hidden_dim_mlp
            ):
                raise ValueError("activation_last must be None, torch.nn.Identity(), or a lambda function using torch.clamp.")

        self.additive_update = additive_update
        self.num_input_channels = num_input_channels  # num channels NCAs (alive)
        self.num_external_channels = num_external_channels  # num channels external input (not modified, fed to NCA) 0 if None
        self.stochastic_update_ratio = stochastic_update_ratio
        self.hidden_dim_mlp = hidden_dim_mlp
        self.alive_threshold = alive_threshold
        self.alive_channel = alive_channel
        self.alive_mask_goal = alive_mask_goal

        self.convolution_mode = convolution_mode
        self.convolution_mode in [
            "mixing_features",
            "one_kernel_per_channel",
            "share_kernels_across_channels",
        ]
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
        self.bias = bias
        self.fixed_kernels = fixed_kernels
        self.width_kernel = width_kernel
        self.isotropic_only = isotropic_only
        self.extra_kernels = extra_kernels

        ###############################
        # Assert contructor arguments #
        ###############################
        if convolution_mode == "share_kernels_across_channels":
            assert num_output_conv_features == None, "num_output_conv_features should be None for share_kernels_across_channels"
            if not fixed_kernels:
                assert num_kernels is not None, "Num kernels should not be None for this case."
        elif convolution_mode == "one_kernel_per_channel":
            assert (
                num_kernels == None
            ), "num_kernels should be None for one_kernel_per_channel, it is automatically set to num_input_channels"
            assert (
                num_output_conv_features == None or num_output_conv_features == num_input_channels
            ), "num_output_conv_features should be None or equal to number of input channels for one_kernel_per_channel"
            assert custom_kernels == None, "kernels should be None for one_kernel_per_channel"
            assert fixed_kernels == False, "Kernels can not be fixed in this mode. (coming soon)"
        elif convolution_mode == "mixing_features":
            assert (
                num_kernels == None
            ), "num_kernels should be None for mixing_features, it is controlled bynum_output_conv_features"
            assert custom_kernels == None, "kernels should be None for convolution_mode mixing_features"
            assert fixed_kernels == False, "Kernels can not be fixed in this mode. (coming soon)"
            assert num_output_conv_features is not None, "num_output_conv_features should not be None for mixing_features"

        if fixed_kernels:
            assert width_kernel == None, "Kernels size can not be decided in this case."
            assert num_kernels == None, "Num filters can not be decided in this case."
        else:
            assert width_kernel is not None, "Kernel width shall not be None in this case"

        #################
        # Construct NCA #
        #################

        # Conv Layers
        conv_layers = []

        if convolution_mode == "mixing_features":
            self.num_output_conv_features = num_output_conv_features
            for k in range(self.num_conv_layers):
                conv_layers.extend(
                    [
                        torch.nn.Conv2d(
                            in_channels=(num_input_channels if k == 0 else self.num_output_conv_features),
                            out_channels=num_output_conv_features,
                            kernel_size=width_kernel,
                            padding=1,
                            bias=bias,
                            padding_mode=self.boundary_condition,  # 'zeros', 'reflect', 'replicate' or 'circular'
                        ),
                        activation_conv,
                        (torch.nn.Dropout(p=self.dropout) if self.dropout > 0 else torch.nn.Identity()),
                    ]
                )
                # print("Conv layer ", k, " weight size ", conv_layers[-2].weight.size())
            self.conv_layers = torch.nn.Sequential(*conv_layers)
        elif convolution_mode == "one_kernel_per_channel":  # stays always same number of output features
            self.num_output_conv_features = num_input_channels
            for k in range(self.num_conv_layers):
                conv_layers.extend(
                    [
                        torch.nn.Conv2d(
                            in_channels=num_input_channels,
                            out_channels=self.num_output_conv_features,
                            groups=num_input_channels,
                            kernel_size=width_kernel,
                            padding=1,
                            bias=bias,
                            padding_mode=self.boundary_condition,  # 'zeros', 'reflect', 'replicate' or 'circular'
                        ),
                        activation_conv,
                        (torch.nn.Dropout(p=self.dropout) if self.dropout > 0 else torch.nn.Identity()),
                    ]
                )
                # print("Conv layer ", k, " weight size ", conv_layers[-2].weight.size())
            self.conv_layers = torch.nn.Sequential(*conv_layers)

        elif convolution_mode == "share_kernels_across_channels":
            if not fixed_kernels:
                self.conv_filters = torch.nn.Parameter(
                    torch.randn(num_kernels, width_kernel, width_kernel)
                )  # Initialize with random values
            else:
                self.conv_filters = (
                    get_kernels(isotropic_only=self.isotropic_only, extra_kernels=self.extra_kernels)
                    if not custom_kernels
                    else custom_kernels
                )
            self.num_kernels = self.conv_filters.shape[0] if fixed_kernels else num_kernels
            self.num_output_conv_features = num_input_channels * pow(self.num_kernels, num_conv_layers)

        else:
            raise ValueError(
                "convolution_mode must be 'full', 'share_kernels_across_channels', or 'share_kernels_within_channels'."
            )

        ##############################
        # Merge external input setup #
        ##############################
        self.merge_ext = merge_ext
        if self.merge_ext:
            self.merge_ext_layer = torch.nn.Linear(num_external_channels, self.num_output_conv_features, bias=bias)

        ##########################
        # Fully connected layers #
        ##########################
        # Number of parameters=(kernel height × kernel width × input channels+bias) × output channels

        if self.merge_ext:
            dim_mlp = [self.num_output_conv_features] + hidden_dim_mlp
        else:
            dim_mlp = [self.num_output_conv_features + num_external_channels] + hidden_dim_mlp

        num_fc_extra_layers = len(hidden_dim_mlp)
        self.fc_layers = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(dim_mlp[i], dim_mlp[i + 1], bias=bias),
                    activation_fc,
                    torch.nn.Dropout(p=self.dropout) if self.dropout > 0 else torch.nn.Identity(),
                )
                for i in range(num_fc_extra_layers)
            ]
        )

        # Output Layer
        self.output_layer = torch.nn.Linear(dim_mlp[-1], num_input_channels, bias=bias)

    def forward(self, x, x_ext=None, update_noise=0.0):
        # Store original input state for both masking mechanisms if needed
        if self.alive_mask_goal and x_ext is not None:
            # Check if specified channel per spatial location has values > threshold
            # x shape: [N, num_input_channels, H, W]
            # external_input_mask shape: [N, 1, H, W]
            external_input_mask_pre = (x[:, self.alive_channel, :, :] > 0).unsqueeze(1).float()

        # Add noise to input
        if update_noise > 0:
            x = x + update_noise * torch.randn_like(x)

        # Apply the convolutional layers
        if self.convolution_mode == "mixing_features" or self.convolution_mode == "one_kernel_per_channel":
            x = self.conv_layers(x)
        elif self.convolution_mode == "share_kernels_across_channels":
            num_inputs_channels = self.num_input_channels
            for _ in range(self.num_conv_layers):
                conv_weights = self.conv_filters.unsqueeze(1).repeat(num_inputs_channels, 1, 1, 1).to(x.device)
                # Apply manual padding for share_kernels_across_channels mode
                if self.custom_padding is not None:
                    t = self.custom_padding_thickness
                    x_padded = apply_custom_padding(x, self.custom_padding, t, self.embedding_dim)
                elif self.boundary_condition == "circular":
                    x_padded = F.pad(x, (1, 1, 1, 1), mode="circular")
                elif self.boundary_condition == "reflect":
                    x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
                elif self.boundary_condition == "replicate":
                    x_padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
                else:  # zeros
                    x_padded = F.pad(x, (1, 1, 1, 1), mode="constant", value=0)

                x = self.activation_conv(conv2d(x_padded, conv_weights, padding=0, groups=num_inputs_channels))
                if self.custom_padding is not None and t > 1:
                    crop = t - 1
                    x = x[:, :, crop:-crop, crop:-crop]
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                # increase num inputs channels for next layers
                num_inputs_channels = num_inputs_channels * self.num_kernels

        # print(f"Shape after conv layers: {x.shape}")
        N, C, H, W = x.shape  # N is batch size, C is number of output conv channels, H is height, W is width

        # Merge or concatenate x_ext
        if x_ext is not None:
            if self.merge_ext:
                # Transform x_ext to (N, num_external_channels) then map to conv_features
                x_ext_transformed = self.merge_ext_layer(x_ext)

                # Unsqueeze to broadcast over H and W : [N, conv_features, 1, 1]
                x_ext_transformed = x_ext_transformed.unsqueeze(-1).unsqueeze(-1)

                # Apply external input mask if needed
                if self.alive_mask_goal:
                    x_ext_transformed = x_ext_transformed * external_input_mask_pre

                # Elementwise add, automatically broadcast over spatial dims
                x = x + x_ext_transformed
            else:
                # Zero out external input vector goal in cells that aren't alive yet
                if self.alive_mask_goal:
                    x_ext = x_ext * external_input_mask_pre

                # Concatenate x and x_ext along the channel dimension (dim=1)
                x = torch.cat((x, x_ext), dim=1)

        # Transform into right shape for fully connected layers
        if self.merge_ext:
            x = x.permute(0, 2, 3, 1).reshape(-1, self.num_output_conv_features)
        else:
            x = x.permute(0, 2, 3, 1).reshape(-1, self.num_output_conv_features + self.num_external_channels)

        x = self.fc_layers(x)
        # print(f"Shape after fc layers: {x.shape}")

        # Apply the final output layer and activation function
        if self.activation_last is None:
            x = self.output_layer(x)
        else:
            x = self.activation_last(self.output_layer(x))
        # print(f"Shape after last layer: {x.shape}")

        assert x.shape == (N * H * W, self.num_input_channels)
        # Transform Back into correct shape ie [N*H*W, elf.num_input_channels] -> [N, H, W, self.num_input_channels]-> [N,, self.num_input_channels, H, W]
        x = x.reshape(N, H, W, self.num_input_channels).permute(0, 3, 1, 2)

        if self.additive_update and self.stochastic_update_ratio < 1.0:
            random_tensor = torch.rand(N, 1, H, W, device=x.device)  # between 0 and 1 for each cell!
            # Threshold the random values based on alpha
            random_mask_binary = (random_tensor <= self.stochastic_update_ratio).float()
            # Scale accross channels (since same mask for all channels for one cell)
            random_mask_binary = random_mask_binary.repeat(1, self.num_input_channels, 1, 1).to(x.device)
            # print("Random_mask_binary shape", random_mask_binary.shape)
            # Apply stochastic update
            x = x * random_mask_binary
        elif not self.additive_update and self.stochastic_update_ratio < 1.0:
            # TODO
            raise NotImplementedError("Stochastic update is only implemented for additive updates")

        return x


def evaluate_nca(
    model,
    initial_state,
    encoder_vector,
    nca_steps,
    additive,
    state_norm,
    alive_mask,
    update_noise=0.0,
):
    state = initial_state.clone().detach().requires_grad_(False)

    # Broadcast one-hot encoder to spatial dimensions for visualization
    # Handle case where encoder_vector is None (no task encoder)
    if encoder_vector is not None:
        _, _, H, W = state.shape
        sample_one_hot_spatial = encoder_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
    else:
        sample_one_hot_spatial = None

    states = []
    states.append(state)
    for step in range(nca_steps):
        if alive_mask:
            pre_life_mask = compute_alive_mask(state, model.alive_threshold)

        if additive:
            state = state + model(state, x_ext=sample_one_hot_spatial, update_noise=update_noise)

        else:
            state = model(state, x_ext=sample_one_hot_spatial, update_noise=update_noise)

        if alive_mask:
            post_life_mask = compute_alive_mask(state, model.alive_threshold)
            life_mask = (pre_life_mask & post_life_mask).float()
            state = state * life_mask

        if state_norm:
            state = torch.clamp(state, 0, 1)

        states.append(state)

    states = torch.stack(states, axis=0)
    return states[-1], states


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using device:", device)

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    # Load emoji dataset
    size = 20
    extra_channels = 10
    nca_steps = size // 2
    # nca_steps = 3
    num_patterns = 8  # Number of different patterns to train on
    embedding_dim = 4
    additive = True

    # Configure external channels based on encoder type
    use_one_hot_encoder = True  # True for one-hot encoder, False for random encoding U(-1, 1)
    external_encoder_dim = num_patterns
    alive_mask_goal = True
    alive_mask = True
    alive_threshold = 0.0

    # Create dataset and dataloader
    dataset = GoalPatternsDataset(
        num_patterns=num_patterns,
        size=size,
        seed_type="single_cell_ones",  # "single_cell_ones", "single_cell_random", "all_cells_random", "all_cells_ones", "single_cell_RGB_ones", "single_cell_RGB_random"
        embedding_dim=embedding_dim,
        extra_channels=extra_channels,
        one_hot_encoder=use_one_hot_encoder,
        external_encoder_dim=external_encoder_dim,
        device=device,
        dtype=dtype,
    )

    batch_size = num_patterns  # Process all patterns in one batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Dataset created with {len(dataset)} patterns")
    print(f"Selected patterns: {dataset.pattern_identifiers}\n")
    # print selected emajist next to their encoding vector
    for emoji, encoding in zip(dataset.pattern_identifiers, dataset.task_encoders):
        print(f"{emoji}: {encoding.cpu().detach().numpy()}")

    print(f"\nUsing one-hot encoder with dimension {external_encoder_dim}\n")

    model = NCA_mlp(
        num_input_channels=embedding_dim + extra_channels,
        num_external_channels=external_encoder_dim,
        num_output_conv_features=None,
        num_conv_layers=1,
        hidden_dim_mlp=[32],
        bias=False,
        # torch.nn.LeakyReLU(),torch.nn.Tanh(),torch.nn.Identity(),torch.nn.Sigmoid(), lambda x: torch.clamp(x, 0, 1), torch.nn.ReLU6()
        activation_conv=torch.nn.Tanh(),
        # activation_conv=torch.nn.LeakyReLU(),
        activation_fc=torch.nn.Tanh(),
        # Only these are valid: None, torch.nn.Identity(),  lambda x: torch.clamp(x, 0, 1),  lambda x: torch.clamp(x, -1, 1)
        # activation_last=None,
        activation_last=torch.nn.Identity(),
        stochastic_update_ratio=1,
        convolution_mode="share_kernels_across_channels",
        fixed_kernels=False,
        num_kernels=8,
        custom_kernels=None,
        width_kernel=3,
        additive_update=additive,
        merge_ext=False,
        dropout=0,
        alive_mask_goal=alive_mask_goal,
        alive_threshold=alive_threshold,
        alive_channel=3,  # Use 4th channel (index 3, default)
        boundary_condition="circular",
        custom_padding=None,
    ).to(device)

    # Training loop
    num_epochs = 1000

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    tic = time.time()
    for epoch in tqdm(range(num_epochs), desc="Training NCA"):
        for batch_idx, (initial_seeds, one_hot_encoders, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            state = initial_seeds.clone().detach().requires_grad_(True)  # Reset the state for each batch

            # Broadcast one-hot encoders to spatial dimensions [batch_size, num_patterns, H, W]
            batch_size_actual, num_channels, H, W = state.shape
            one_hot_spatial = one_hot_encoders.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

            # Run the NCA for a fixed number of steps
            for _ in range(nca_steps):
                if additive:
                    state = state + model(state, x_ext=one_hot_spatial)
                else:
                    state = model(state, x_ext=one_hot_spatial)

            loss = criterion(state[:, :embedding_dim], targets[:, :embedding_dim])
            loss.backward(retain_graph=True)
            optimizer.step()

            if epoch % 100 == 0 and batch_idx == 0:
                tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")

    print("Training took:", time.time() - tic)

    # Visualize all patterns in a grid after training
    # Batch all initial seeds and one-hot encoders
    all_initial_seeds = []
    all_one_hots = []
    for idx in range(len(dataset)):
        initial_seed, one_hot, target = dataset[idx]
        all_initial_seeds.append(initial_seed)
        all_one_hots.append(one_hot)
    all_initial_seeds = torch.stack(all_initial_seeds, dim=0).to(device)  # [batch, C, H, W]
    all_one_hots = torch.stack(all_one_hots, dim=0).to(device)  # [batch, num_patterns]

    # Run evaluate_nca in batch
    results = evaluate_nca(
        model=model,
        initial_state=all_initial_seeds,
        encoder_vector=all_one_hots,
        nca_steps=nca_steps,
        additive=additive,
        state_norm=True,
        alive_mask=True,
        update_noise=0.0,
    )

    # Visualize final step
    final_step = results[-1]
    visualize_all_patterns_grid(dataset, embedding_dim, final_step)

    # Create grid animation of NCA evolution for all patterns
    id_ = np.random.randint(0, 10e6)
    create_animation_grid(dataset, embedding_dim, results, output_path="media", id_=id_)
