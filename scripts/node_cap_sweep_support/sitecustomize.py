"""Allow dimension-only audits of trusted, locally rendered Graphviz PNGs."""
from PIL import Image

# image_audit only reads Image.size. Some cap-40/cap-60 Graphviz canvases exceed
# Pillow's generic untrusted-input threshold even though no pixels are decoded.
Image.MAX_IMAGE_PIXELS = None
