import numpy as np
import cv2
from collections import namedtuple
from typing import Tuple, List

Roi = namedtuple('Roi', ['xmin', 'ymin', 'xmax', 'ymax'])
RoiKernel = namedtuple('RoiKernel', ['roi', 'kernel'])


class EmphasiseROI:
    def __init__(self, kernel_width: int, kernel_height: int, rois: Tuple[Roi, ...], merge_strategy='max') -> None:
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.rois = rois
        self.merge_strategy = merge_strategy
        self._validate_initial_arguments()

    def _validate_initial_arguments(self) -> None:
        if self.merge_strategy not in ('min', 'max', 'avg'):
            raise ValueError('combine strategy must be one of min, max, avg')
        for roi in self.rois:
            roi_width = roi.xmax - roi.xmin
            roi_height = roi.ymax - roi.ymin
            if roi_width <= 3 or roi_height <= 3:
                raise ValueError("roi_width and roi_height must be greater than 5")
            if self.kernel_width <= 5 or self.kernel_width <= roi_width + 1:
                raise ValueError("kernel_width must be greater than 8 and roi_width + 1")
            if self.kernel_height <= 5 or self.kernel_height <= roi_height + 1:
                raise ValueError("kernel_height must be greater than 8 and roi_height + 1")

    def _validate_call_arguments(self, alpha: float, beta: float, merge_strategy: str) -> None:
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if not (0 <= beta <= 1):
            raise ValueError("beta must be between 0 and 1")
        if merge_strategy not in ('min', 'max', 'avg'):
            raise ValueError('combine strategy must be one of min, max, avg')

    def _generate_roi_kernel(self, roi: Roi, alpha: float, beta: float) -> RoiKernel:
        roi_width = roi.xmax - roi.xmin
        roi_height = roi.ymax - roi.ymin
        kernel = np.full((roi_height, roi_width), 1, dtype=float)
        return RoiKernel(roi=roi, kernel=kernel)

    def _place_roi_kernels(self, main_kernel: np.ndarray, roi_kernels: List[RoiKernel],
                           merge_strategy: str) -> np.ndarray:
        def place_roi_kernel(roi_kernel: RoiKernel):
            roi, kernel = roi_kernel
            main_kernel_layer = main_kernel.copy()
            main_kernel_layer[
            roi.ymin: roi.ymax,
            roi.xmin: roi.xmax,
            ] = kernel
            return main_kernel_layer

        layers = []
        for roi_kernel in roi_kernels:
            layers.append(place_roi_kernel(roi_kernel))

        if merge_strategy == 'min':
            main_kernel = np.min([*layers], axis=0)
        elif merge_strategy == 'max':
            main_kernel = np.max([*layers], axis=0)
        elif merge_strategy == 'avg':
            main_kernel = np.mean([*layers], axis=0)
        else:
            raise ValueError('merge_strategy must be one of min, max, avg')

        return main_kernel

    def __call__(self, alpha: float = 0.3, beta: float = 1.0, merge_strategy: str = None) -> np.ndarray:
        merge_strategy = merge_strategy if merge_strategy else self.merge_strategy
        self._validate_call_arguments(alpha=alpha, beta=beta, merge_strategy=merge_strategy)

        kernel = np.full((self.kernel_height, self.kernel_width), alpha, dtype=float)
        roi_kernels = [self._generate_roi_kernel(roi=roi, alpha=alpha, beta=beta) for roi in self.rois]
        kernel = self._place_roi_kernels(main_kernel=kernel, roi_kernels=roi_kernels, merge_strategy=merge_strategy)

        return kernel


class EmphasiseEllipticalROI(EmphasiseROI):
    def _generate_roi_kernel(self, roi: Roi, alpha: float, beta: float) -> RoiKernel:
        roi_width = roi.xmax - roi.xmin
        roi_height = roi.ymax - roi.ymin
        kernel = np.full((roi_height, roi_width), 0, dtype=np.uint8)

        # Find the center coordinates of the kernel
        center_x = roi_width // 2
        center_y = roi_height // 2

        # Calculate the major and minor axes lengths for the ellipse
        major_axis = roi_width // 2
        minor_axis = roi_height // 2

        # assign elliptic area of roi_kernel as beta
        cv2.ellipse(kernel, (center_x, center_y), (major_axis, minor_axis), 0, 0, 360, (beta * 255),
                    thickness=-1)

        # Calculate the distance of each pixel from the center of the ellipse
        distances = np.zeros_like(kernel, dtype=np.float32)
        for y in range(roi_height):
            for x in range(roi_width):
                distances[y, x] = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Scale the distances to the range of 0-(1-alpha)
        scaled_distances = (distances / np.max(distances)) * ((1 - alpha) * 255)

        # Add the scaled distances to the image to smoothly decrease pixel values
        kernel = np.clip(kernel + 255 - scaled_distances, 0, 255).astype(np.uint8)

        cv2.ellipse(kernel, (center_x, center_y), (major_axis, minor_axis), 0, 0, 360, (beta * 255),
                    thickness=-1)

        # Normalize kernel values to range 0-1
        kernel = kernel.astype(np.float32) / 255.0

        return RoiKernel(roi=roi, kernel=kernel)


class AdjustedTransparencyKernel:
    def __init__(
            self,
            kernel_width: int,
            kernel_height: int,
            roi_xmin: int,
            roi_ymin: int,
            roi_xmax: int,
            roi_ymax: int,
            alpha: float = 0.3,
            beta: float = 0.9,
    ):
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.roi_xmin = roi_xmin
        self.roi_ymin = roi_ymin
        self.roi_xmax = roi_xmax
        self.roi_ymax = roi_ymax
        self.alpha = alpha
        self.beta = beta

        self.roi_width = self.roi_xmax - self.roi_xmin
        self.roi_height = self.roi_ymax - self.roi_ymin

        self._validate_arguments()

        self.kernel = np.full((self.kernel_height, self.kernel_width), self.alpha, dtype=float)
        self.roi_kernel = self._generate_roi_kernel()

    def __call__(self):
        self._place_roi_kernel_in_kernel()
        return self.kernel

    def _validate_arguments(self):
        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if not (0 <= self.beta <= 1):
            raise ValueError("beta must be between 0 and 1")
        if self.roi_width <= 3 or self.roi_height <= 3:
            raise ValueError("roi_width and roi_height must be greater than 5")
        if self.kernel_width <= 5 or self.kernel_width <= self.roi_width + 1:
            raise ValueError("kernel_width must be greater than 8 and roi_width + 1")
        if self.kernel_height <= 5 or self.kernel_height <= self.roi_height + 1:
            raise ValueError("kernel_height must be greater than 8 and roi_height + 1")

    def __generate_roi_kernel(self):
        center_x = self.roi_width // 2
        center_y = self.roi_height // 2
        radius = min(center_x, center_y)
        x_indices, y_indices = np.indices((self.roi_height, self.roi_width))

        distance_from_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        Elliptical_mask = distance_from_center <= radius

        roi_kernel = np.where(Elliptical_mask, self.beta, self.alpha)

        return roi_kernel

    def _generate_roi_kernel(self):
        roi_kernel = np.zeros((self.roi_height, self.roi_width), dtype=np.uint8)

        # Find the center coordinates of the kernel
        center_x = self.roi_width // 2
        center_y = self.roi_height // 2

        # Calculate the major and minor axes lengths for the ellipse
        major_axis = self.roi_width // 2
        minor_axis = self.roi_height // 2

        # assign elliptic area of roi_kernel as beta
        cv2.ellipse(roi_kernel, (center_x, center_y), (major_axis, minor_axis), 0, 0, 360, (self.beta * 255),
                    thickness=-1)

        # Calculate the distance of each pixel from the center of the ellipse
        distances = np.zeros_like(roi_kernel, dtype=np.float32)
        for y in range(self.roi_height):
            for x in range(self.roi_width):
                distances[y, x] = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Scale the distances to the range of 0-(1-alpha)
        scaled_distances = (distances / np.max(distances)) * ((1 - self.alpha) * 255)

        # Add the scaled distances to the image to smoothly decrease pixel values
        roi_kernel = np.clip(roi_kernel + 255 - scaled_distances, 0, 255).astype(np.uint8)

        cv2.ellipse(roi_kernel, (center_x, center_y), (major_axis, minor_axis), 0, 0, 360, (self.beta * 255),
                    thickness=-1)

        # Normalize kernel values to range 0-1
        roi_kernel = roi_kernel.astype(np.float32) / 255.0

        return roi_kernel

    def _place_roi_kernel_in_kernel(self):
        self.kernel[
        self.roi_ymin: self.roi_ymax,
        self.roi_xmin: self.roi_xmax,
        ] = self.roi_kernel


import matplotlib.pyplot as plt

if __name__ == '__main__':
    rois = (Roi(xmin=20, ymin=20, xmax=240, ymax=160), Roi(xmin=80, ymin=20, xmax=300, ymax=160))
    kernel_generator = EmphasiseROI(kernel_width=320, kernel_height=180, rois=rois)
    Elliptical_kernel_generator = EmphasiseEllipticalROI(kernel_width=320, kernel_height=180, rois=rois)

    # Generate kernels using kernel_generator
    alpha_values = [0.3, 0.5]
    beta_value = 1.0
    merge_strategies = ['min', 'max', 'avg']

    kernel_images = []
    for alpha in alpha_values:
        for merge_strategy in merge_strategies:
            kernel = kernel_generator(alpha=alpha, beta=beta_value, merge_strategy=merge_strategy)
            kernel = (kernel * 255).astype(np.uint8)  # Scale kernel to 0-255 range
            kernel_rgb = cv2.cvtColor(kernel, cv2.COLOR_GRAY2RGB)  # Convert to colored image
            kernel_images.append((kernel_rgb, f"alpha={alpha}, merge_strategy={merge_strategy}"))

    # Generate kernels using Elliptical_kernel_generator
    alpha_values = [0.3, 0.5]
    beta_values = [0.75, 0.9]
    merge_strategies = ['min', 'max', 'avg']

    Elliptical_kernel_images = []
    for alpha in alpha_values:
        for beta in beta_values:
            for merge_strategy in merge_strategies:
                kernel = Elliptical_kernel_generator(alpha=alpha, beta=beta, merge_strategy=merge_strategy)
                kernel = (kernel * 255).astype(np.uint8)  # Scale kernel to 0-255 range
                kernel_rgb = cv2.cvtColor(kernel, cv2.COLOR_GRAY2RGB)  # Convert to colored image
                Elliptical_kernel_images.append((kernel_rgb, f"alpha={alpha}, beta={beta}\nmerge_strategy={merge_strategy}"))

    # Create a full-screen figure with 3 rows and 6 columns
    fig, axes = plt.subplots(3, 6, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    # Plot the kernels generated by kernel_generator
    for i, (kernel, info) in enumerate(kernel_images):
        row = 0
        col = i % 6
        axes[row, col].imshow(kernel)
        axes[row, col].axis('off')
        axes[row, col].text(0.5, 1.05, info, color='red', fontsize=10, ha='center', transform=axes[row, col].transAxes)

    # Plot the kernels generated by Elliptical_kernel_generator
    for i, (kernel, info) in enumerate(Elliptical_kernel_images):
        row = (i // 6) + 1
        col = i % 6
        axes[row, col].imshow(kernel)
        axes[row, col].axis('off')
        axes[row, col].text(0.5, 1.05, info, color='red', fontsize=10, ha='center', transform=axes[row, col].transAxes)

    # Add the text "Original Rectangle Kernel" above the first row
    fig.text(0.5, 0.95, "Original - Rectangular ROI Kernels", color='black', fontsize=15, ha='center', weight='bold')

    # Add the text "Elliptical Smoothed ROI Kernel" above the second row
    fig.text(0.5, 0.64, "Proposed - Elliptical Smoothed ROI Kernels", color='black', fontsize=15, ha='center', weight='bold')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)

    # Show the figure
    plt.show()
