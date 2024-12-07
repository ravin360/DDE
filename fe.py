import cv2
import numpy as np
import matplotlib.pyplot as plt





def dynamic_top_edge_roi(edges, initial_height=10, step=10, min_white_pixels=50):
    """Dynamically adjust the top edge ROI based on edge density."""
    height, _ = edges.shape
    for y in range(initial_height, height, step):
        top_edge_roi = edges[:y, :]
        white_pixel_count = np.sum(top_edge_roi == 255)
        if white_pixel_count >= min_white_pixels:
            return top_edge_roi, y, white_pixel_count
    return edges[:initial_height, :], initial_height, np.sum(edges[:initial_height, :] == 255)


def highlight_white_pixels(image, edge_roi, roi_height):
    """Create a visualization with highlighted white pixels."""
    # Create a copy of the original image for highlighting
    highlighted_image = image.copy()

    # Create a mask for white pixels in the ROI
    mask = (edge_roi == 255)

    # Create a colored overlay (using red for visibility)
    overlay = np.zeros_like(image)
    overlay[:roi_height, :] = [0, 0, 255]  # Red color for highlights

    # Apply the mask to the overlay
    for i in range(3):
        overlay[:roi_height, :, i] = overlay[:roi_height, :, i] * mask

    # Blend the overlay with the original image
    highlighted_image = cv2.addWeighted(highlighted_image, 1, overlay, 0.5, 0)

    return highlighted_image

# Detect edges using Canny with adjusted thresholds


# Get the dynamically adjusted top edge ROI
top_edge_roi_canny_adjusted, roi_height, white_pixel_count = dynamic_top_edge_roi(canny_edges_adjusted)

# Create the highlighted visualization
highlighted_image = highlight_white_pixels(image, top_edge_roi_canny_adjusted, roi_height)

# Display the results
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Edge detection result
plt.subplot(1, 3, 2)
plt.imshow(canny_edges_adjusted, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

# Highlighted result
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
plt.title(f'Highlighted White Pixels\nCount: {white_pixel_count}')
plt.axis('off')

plt.tight_layout()
plt.show()