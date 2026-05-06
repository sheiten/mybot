#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Paint by Numbers Bot v3.0 Professional
Telegram bot for generating paint-by-number coloring pages from images.

Features:
• LAB color space clustering for perceptually accurate colors
• Spatial-aware KMeans for coherent regions
• Smart small-region merging with neighbor analysis
• Distance-transform based number placement
• SVG/PNG dual export support
• Production-ready error handling & security
"""

import asyncio
import logging
import os
import io
import math
import ssl
import urllib.request
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy import stats as sp_stats
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

# Security: External config via env vars only
PORTAINER_WEBHOOK_URL = os.environ.get('PORTAINER_WEBHOOK_URL', '')
PORTAINER_TOKEN = os.environ.get('PORTAINER_TOKEN', '')
ADMIN_ID = int(os.environ.get('ADMIN_USER_ID', '931848809'))

DEFAULT_CONFIG = {
    'n_colors': 24,
    'min_region_size': 150,
    'max_image_size': 1500,
    'line_thickness': 1,
    'line_color': 'gray',
    'font_size': 11,
    'preprocess_strength': 'medium',
    'color_space': 'lab',  # 'rgb' or 'lab'
    'spatial_weight': 0.1,
    'use_minibatch': True,
    'export_svg': False,
}

LINE_COLORS = {
    'gray': (180, 180, 180),
    'dark': (100, 100, 100),
    'light': (210, 210, 210),
    'black': (0, 0, 0),
}

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pbn_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class PBNConfig:
    """Configuration for Paint by Numbers generation"""
    n_colors: int = 24
    min_region_size: int = 150
    max_image_size: int = 1500
    line_thickness: int = 1
    line_color: str = 'gray'
    font_size: int = 11
    preprocess_strength: str = 'medium'
    color_space: str = 'lab'
    spatial_weight: float = 0.1
    use_minibatch: bool = True
    export_svg: bool = False
    
    @property
    def line_rgb(self) -> Tuple[int, int, int]:
        return LINE_COLORS.get(self.line_color, LINE_COLORS['gray'])
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PBNConfig':
        """Create config from dict, filtering unknown keys"""
        valid_keys = cls.__annotations__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_keys})
    
    @classmethod
    def from_user_data(cls, user_data: dict) -> 'PBNConfig':
        """Create config from Telegram user_data"""
        return cls.from_dict(user_data)


# ============================================
# SECURITY & UTILS
# ============================================

def trigger_self_update() -> bool:
    """Secure webhook trigger for Portainer update"""
    if not PORTAINER_WEBHOOK_URL or not PORTAINER_TOKEN:
        logger.error("Portainer credentials not configured")
        return False
    
    try:
        # Create verified SSL context
        ctx = ssl.create_default_context()
        
        req = urllib.request.Request(
            PORTAINER_WEBHOOK_URL,
            method='POST',
            headers={
                'Authorization': f'Bearer {PORTAINER_TOKEN}',
                'Content-Type': 'application/json',
                'User-Agent': 'PBN-Bot/3.0'
            },
            data=json.dumps({'action': 'redeploy'}).encode('utf-8')
        )
        
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            return response.status in (200, 204)
            
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error {e.code}: {e.reason}")
        return False
    except Exception as e:
        logger.error(f"Update webhook error: {type(e).__name__}: {e}")
        return False


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Cross-platform font loader with robust fallbacks"""
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Current directory / bundled
        "Arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(str(font_path), size)
        except (IOError, OSError, ValueError, AttributeError):
            continue
    
    # 👇 Ultimate fallback: use PIL's built-in bitmap font safely
    logger.warning(f"No TrueType font found for size {size}, using fallback")
    try:
        # Try to load a small bitmap font that always exists
        return ImageFont.load_default()
    except Exception:
        # If all else fails, return None and handle in caller
        return None

# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image: Image.Image, target_size: int, 
                    strength: str = 'medium') -> np.ndarray:
    """
    Preprocess image: resize, denoise, prepare for clustering.
    Returns numpy array in RGB format.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Smart resize preserving aspect ratio
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    # Adaptive denoising based on strength
    denoise_params = {
        'light': {'d': 5, 'sigmaColor': 50, 'sigmaSpace': 50},
        'medium': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
        'strong': {'d': 15, 'sigmaColor': 100, 'sigmaSpace': 100},
    }
    params = denoise_params.get(strength, denoise_params['medium'])
    
    # Bilateral filter preserves edges while smoothing
    img_array = cv2.bilateralFilter(img_array, **params)
    
    return img_array


# ============================================
# COLOR CLUSTERING (LAB + SPATIAL)
# ============================================

def cluster_colors(img_array: np.ndarray, config: PBNConfig) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """
    Cluster colors using KMeans in LAB or RGB space with optional spatial weighting.
    Returns: (quantized_image, label_map, palette_colors_rgb)
    """
    h, w = img_array.shape[:2]
    total_pixels = h * w
    
    # Choose color space
    if config.color_space == 'lab':
        img_processed = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        convert_back = lambda centers: cv2.cvtColor(
            centers.astype(np.uint8).reshape(-1, 1, 3), 
            cv2.COLOR_LAB2RGB
        ).reshape(-1, 3)
    else:
        img_processed = img_array.astype(np.float32)
        convert_back = lambda centers: centers
    
    # Prepare features: [color_channels, optional_spatial]
    if config.spatial_weight > 0:
        # Normalized coordinates
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        x_norm = (x_coords / w).astype(np.float32) * config.spatial_weight
        y_norm = (y_coords / h).astype(np.float32) * config.spatial_weight
        
        features = np.stack([
            img_processed[:, :, 0].ravel(),
            img_processed[:, :, 1].ravel(),
            img_processed[:, :, 2].ravel(),
            x_norm.ravel(),
            y_norm.ravel()
        ], axis=1)
        color_dims = 3
    else:
        features = img_processed.reshape(-1, 3)
        color_dims = 3
    
    # Choose clustering algorithm based on image size
    use_minibatch = config.use_minibatch and total_pixels > 300_000
    
    if use_minibatch:
        # Sample for MiniBatchKMeans fitting
        sample_size = min(50_000, total_pixels)
        sample_indices = np.random.choice(total_pixels, sample_size, replace=False)
        sample_features = features[sample_indices]
        
        kmeans = MiniBatchKMeans(
            n_clusters=config.n_colors,
            batch_size=1000,
            random_state=42,
            n_init=3,
            max_iter=100
        )
        kmeans.fit(sample_features)
        labels = kmeans.predict(features).reshape(h, w)
    else:
        # Full KMeans for smaller images
        kmeans = KMeans(
            n_clusters=config.n_colors,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(features).reshape(h, w)
    
    # Extract and convert centers to RGB
    centers = kmeans.cluster_centers_[:, :color_dims]  # Only color dimensions
    centers_rgb = convert_back(centers).astype(np.uint8)
    
    # Sort palette by brightness for intuitive numbering
    brightness = 0.299 * centers_rgb[:, 0] + 0.587 * centers_rgb[:, 1] + 0.114 * centers_rgb[:, 2]
    sorted_indices = np.argsort(brightness)
    
    # Remap labels to sorted order
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    labels = np.vectorize(label_map.get)(labels)
    centers_rgb = centers_rgb[sorted_indices]
    
    # Reconstruct quantized image
    quantized = centers_rgb[labels]
    
    return quantized, labels, [tuple(c) for c in centers_rgb]


# ============================================
# REGION MERGING & CLEANUP
# ============================================

def merge_small_regions(quantized: np.ndarray, labels: np.ndarray, 
                       palette: List[Tuple[int, int, int]], 
                       min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge regions smaller than min_area with their most similar neighbor.
    Returns: (cleaned_quantized, cleaned_labels)
    """
    h, w = quantized.shape[:2]
    new_labels = labels.copy()
    new_quantized = quantized.copy()
    
    # Process each color channel separately for connected components
    for color_idx, color in enumerate(palette):
        color_mask = (labels == color_idx).astype(np.uint8)
        
        # Morphological cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Find connected components
        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=8, ltype=cv2.CV_32S
        )
        
        for comp_id in range(1, num_labels):  # Skip background (0)
            area = stats[comp_id, cv2.CC_STAT_AREA]
            
            if area < min_area:
                # Create mask for this small component
                comp_mask = (comp_labels == comp_id)
                
                # Find boundary pixels (dilate - original)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated = cv2.dilate(comp_mask.astype(np.uint8), kernel)
                boundary = dilated - comp_mask.astype(np.uint8)
                
                if np.sum(boundary) > 0:
                    # Get neighbor colors
                    neighbor_colors = new_labels[boundary > 0]
                    
                    if len(neighbor_colors) > 0:
                        # Find most frequent neighbor color
                        neighbor_counts = np.bincount(neighbor_colors.flatten())
                        dominant_neighbor = np.argmax(neighbor_counts)
                        
                        # Merge: assign to dominant neighbor
                        new_labels[comp_mask] = dominant_neighbor
                        new_quantized[comp_mask] = palette[dominant_neighbor]
    
    return new_quantized, new_labels


# ============================================
# SMART NUMBER PLACEMENT
# ============================================

def find_optimal_number_position(contour: np.ndarray, mask: np.ndarray, 
                                font_size: int, placed_positions: List[Tuple[int, int]],
                                min_distance_factor: float = 2.5) -> Optional[Tuple[int, int]]:
    """
    Find optimal position for number inside region using distance transform.
    Avoids boundaries and other numbers.
    FIXED: Convert numpy ints to native Python types for OpenCV compatibility.
    """
    if len(contour) < 3:
        return None
    
    # Try centroid first — CONVERT TO NATIVE INT
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    
    # 👇 FIX: explicit int() conversion for OpenCV compatibility
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cx, cy = int(cx), int(cy)  # Ensure native Python int, not np.int64
    
    # Check distance to boundary — CONVERT POINT TO TUPLE OF FLOATS
    min_margin = font_size * 2
    
    try:
        dist_to_boundary = cv2.pointPolygonTest(contour, (float(cx), float(cy)), True)
    except (cv2.error, TypeError) as e:
        logger.warning(f"pointPolygonTest failed at ({cx},{cy}): {e}, using fallback")
        dist_to_boundary = -1  # Fallback to distance transform
    
    if dist_to_boundary < min_margin:
        # Use distance transform to find best interior point
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Create search mask: only points far from boundary
        search_mask = (dist_transform >= min_margin).astype(np.uint8)
        
        if np.sum(search_mask) == 0:
            # Fallback: use point with max distance even if close to boundary
            max_loc = np.argmax(dist_transform)
            cy, cx = np.unravel_index(max_loc, dist_transform.shape)
            cx, cy = int(cx), int(cy)
        else:
            # Mask the distance transform
            masked_dist = dist_transform * search_mask
            max_loc = np.argmax(masked_dist)
            cy, cx = np.unravel_index(max_loc, masked_dist.shape)
            cx, cy = int(cx), int(cy)
    
    # Check for collisions with other placed numbers
    min_dist = font_size * min_distance_factor
    for px, py in placed_positions:
        if math.hypot(cx - px, cy - py) < min_dist:
            # Try to nudge away from collision
            angle = math.atan2(cy - py, cx - px) + math.pi
            cx += int(min_dist * 0.5 * math.cos(angle))
            cy += int(min_dist * 0.5 * math.sin(angle))
            
            # Verify new position is still inside contour
            try:
                if cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) < 0:
                    return None  # Can't place without collision
            except (cv2.error, TypeError):
                return None  # Safety fallback
            break
    
    return cx, cy


# ============================================
# CONTOUR & SVG GENERATION
# ============================================

def simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.01) -> np.ndarray:
    """Simplify contour using Ramer-Douglas-Peucker algorithm"""
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    return cv2.approxPolyDP(contour, epsilon, True)


def contour_to_svg_path(contour: np.ndarray) -> str:
    """Convert OpenCV contour to SVG path data string"""
    if len(contour) < 2:
        return ""
    
    simplified = simplify_contour(contour)
    
    # Build SVG path commands
    path_d = f"M {simplified[0][0][0]},{simplified[0][0][1]}"
    for point in simplified[1:]:
        x, y = point[0]
        path_d += f" L {x},{y}"
    path_d += " Z"  # Close path
    
    return path_d


def generate_svg_output(quantized: np.ndarray, palette: List[Tuple[int, int, int]], 
                       config: PBNConfig) -> str:
    """Generate SVG format coloring page"""
    h, w = quantized.shape[:2]
    line_rgb = config.line_rgb
    font_size = config.font_size
    
    # SVG header
    svg_parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<style>.region{{fill:none;stroke:rgb{line_rgb};stroke-width:{config.line_thickness};stroke-linejoin:round}}</style>',
    ]
    
    placed_positions = []
    
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.min_region_size:
                continue
            
            # Add contour path
            path_d = contour_to_svg_path(contour)
            if path_d:
                svg_parts.append(f'<path class="region" d="{path_d}"/>')
            
            # Place number
            pos = find_optimal_number_position(contour, color_mask, font_size, placed_positions)
            if pos:
                cx, cy = pos
                num_str = str(color_idx + 1)
                
                # White background circle for readability
                svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{font_size//2 + 2}" fill="white"/>')
                svg_parts.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
                               f'font-size="{font_size}" font-family="Arial" fill="black">{num_str}</text>')
                placed_positions.append((cx, cy))
    
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


# ============================================
# RASTER OUTPUT GENERATION
# ============================================

def create_coloring_page_raster(quantized: np.ndarray, palette: List[Tuple[int, int, int]], 
                               config: PBNConfig) -> io.BytesIO:
    """Generate PNG coloring page with contours and numbers — FIXED VERSION"""
    h, w = quantized.shape[:2]
    line_rgb = config.line_rgb
    font_size = config.font_size
    
    # White canvas
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    font = get_font(font_size)
    
    # 👇 FIX: Safe text bounds calculation for any font type
    def get_text_size(text: str, font) -> Tuple[int, int]:
        """Get text dimensions compatible with both FreeType and bitmap fonts"""
        if hasattr(font, 'getbbox'):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        elif hasattr(font, 'getsize'):
            return font.getsize(text)
        else:
            # Fallback estimate for bitmap font
            return font_size * len(text) // 2, font_size
    
    placed_positions = []
    
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.min_region_size:
                continue
            
            # Draw contour
            cv2.drawContours(canvas, [contour], -1, line_rgb, config.line_thickness)
            
            # Place number
            pos = find_optimal_number_position(contour, color_mask, font_size, placed_positions)
            if pos:
                cx, cy = pos
                num_str = str(color_idx + 1)
                
                # 👇 FIX: Safe text size calculation
                text_w, text_h = get_text_size(num_str, font)
                
                # White background rectangle for readability
                padding = 3
                x1 = max(0, cx - text_w // 2 - padding)
                y1 = max(0, cy - text_h // 2 - padding)
                x2 = min(w, cx + text_w // 2 + padding)
                y2 = min(h, cy + text_h // 2 + padding)
                
                # Draw background using OpenCV (faster and more reliable)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
                
                # Draw text using PIL
                pil_img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(pil_img)
                
                # Center text properly
                text_x = cx - text_w // 2
                text_y = cy - text_h // 2 + 2  # Small vertical adjustment for baseline
                draw.text((text_x, text_y), num_str, fill='black', font=font)
                
                # Update canvas
                canvas = np.array(pil_img)
                placed_positions.append((cx, cy))
    
    # Save to buffer
    output = io.BytesIO()
    Image.fromarray(canvas).save(output, format='PNG', dpi=(300, 300))
    output.seek(0)
    return output


def create_palette_image(palette: List[Tuple[int, int, int]], config: PBNConfig) -> Image.Image:
    """Generate palette reference image"""
    n_colors = len(palette)
    palette_width = 320
    palette_height = 80 + n_colors * 40
    
    palette_img = Image.new('RGB', (palette_width, palette_height), 'white')
    draw = ImageDraw.Draw(palette_img)
    
    font = get_font(13)
    title_font = get_font(16)
    
    # Header
    draw.text((15, 15), "🎨 ПАЛИТРА ЦВЕТОВ", fill='black', font=title_font)
    draw.text((15, 38), f"Цветов: {n_colors} | Размер: {config.max_image_size}px", 
             fill='gray', font=font)
    
    # Color swatches
    for idx, color in enumerate(palette, start=1):
        y_pos = 65 + (idx - 1) * 40
        
        # Color box with border
        draw.rectangle([(15, y_pos), (48, y_pos + 30)], fill=color, outline=(200, 200, 200), width=2)
        
        # Number
        draw.text((58, y_pos + 6), f"{idx}.", fill='black', font=font)
        
        # HEX code
        hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'.upper()
        draw.text((95, y_pos + 6), hex_color, fill=(100, 100, 100), font=font)
        
        # RGB values
        rgb_text = f"RGB({color[0]}, {color[1]}, {color[2]})"
        draw.text((95, y_pos + 22), rgb_text, fill=(150, 150, 150), font=get_font(10))
    
    return palette_img


# ============================================
# MAIN PROCESSING PIPELINE
# ============================================

def process_image_for_coloring(photo_bytes: bytes, config: PBNConfig) -> Tuple[io.BytesIO, io.BytesIO, Optional[str]]:
    """
    Full processing pipeline.
    Returns: (coloring_page_buffer, palette_buffer, svg_string_or_none)
    """
    # Load and preprocess
    image = Image.open(io.BytesIO(photo_bytes))
    img_array = preprocess_image(image, config.max_image_size, config.preprocess_strength)
    
    # Color clustering
    quantized, labels, palette = cluster_colors(img_array, config)
    
    # Merge small regions
    quantized, labels = merge_small_regions(quantized, labels, palette, config.min_region_size)
    
    # Generate outputs
    coloring_buffer = create_coloring_page_raster(quantized, palette, config)
    
    palette_buffer = io.BytesIO()
    palette_img = create_palette_image(palette, config)
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    # Optional SVG
    svg_output = None
    if config.export_svg:
        try:
            svg_output = generate_svg_output(quantized, palette, config)
        except Exception as e:
            logger.warning(f"SVG generation failed: {e}, falling back to PNG only")
    
    return coloring_buffer, palette_buffer, svg_output


# ============================================
# TELEGRAM BOT HANDLERS
# ============================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message with command overview"""
    await update.message.reply_text(
        '🎨 <b>Paint by Numbers Bot v3.0 Professional</b>\n\n'
        'Отправьте фото — получите профессиональную раскраску!\n\n'
        '<b>🎛️ Быстрые настройки:</b>\n'
        '• <code>/colors 24</code> — количество цветов (3-48)\n'
        '• <code>/detail 150</code> — мин. размер области (50-500)\n'
        '• <code>/smooth medium</code> — сглаживание (light/medium/strong)\n\n'
        '<b>⚙️ Продвинутые:</b>\n'
        '• <code>/space 0.1</code> — вес координат (0-0.5)\n'
        '• <code>/colorspace lab</code> — LAB или RGB\n'
        '• <code>/svg on</code> — включить SVG экспорт\n'
        '• <code>/settings</code> — показать все настройки\n'
        '• <code>/help</code> — полная справка',
        parse_mode='HTML'
    )


async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display current user configuration"""
    cfg = PBNConfig.from_user_data(context.user_data)
    
    settings = (
        '⚙️ <b>Текущие настройки:</b>\n\n'
        f'🎨 Цветов: {cfg.n_colors}\n'
        f'📏 Мин. область: {cfg.min_region_size}px\n'
        f'📝 Линии: {cfg.line_thickness}px ({cfg.line_color})\n'
        f'🔤 Шрифт: {cfg.font_size}pt\n'
        f'🖼️ Макс. размер: {cfg.max_image_size}px\n'
        f'🌊 Сглаживание: {cfg.preprocess_strength}\n'
        f'🎨 Пространство: {cfg.color_space.upper()}\n'
        f'📍 Вес координат: {cfg.spatial_weight}\n'
        f'📄 SVG экспорт: {"✅ Вкл" if cfg.export_svg else "❌ Выкл"}\n'
        f'⚡ MiniBatch: {"✅ Вкл" if cfg.use_minibatch else "❌ Выкл"}'
    )
    await update.message.reply_text(settings, parse_mode='HTML')


# Generic setter factory
def make_setter(param_name: str, param_type: type, min_val: Union[int, float], 
                max_val: Union[int, float], success_msg: str, 
                valid_values: Optional[List[str]] = None):
    """Factory for creating parameter setter handlers"""
    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not context.args:
            await update.message.reply_text(
                f'❌ Используйте: <code>/{param_name} {min_val}</code>', 
                parse_mode='HTML'
            )
            return
        
        arg = context.args[0].lower()
        
        # Handle string enums
        if valid_values:
            if arg not in valid_values:
                await update.message.reply_text(
                    f'❌ Допустимые значения: {", ".join(valid_values)}', 
                    parse_mode='HTML'
                )
                return
            value = arg
        else:
            # Handle numeric
            try:
                value = param_type(arg)
                if not (min_val <= value <= max_val):
                    raise ValueError
            except ValueError:
                await update.message.reply_text(
                    f'❌ Число от {min_val} до {max_val}', 
                    parse_mode='HTML'
                )
                return
        
        context.user_data[param_name] = value
        await update.message.reply_text(f'✅ {success_msg.format(value)}', parse_mode='HTML')
    
    return handler


# Create all setter handlers
set_colors = make_setter('n_colors', int, 3, 48, 'Установлено {} цветов')
set_detail = make_setter('min_region_size', int, 50, 500, 'Мин. область: {}px')
set_line = make_setter('line_thickness', int, 1, 3, 'Толщина линий: {}')
set_linecolor = make_setter('line_color', str, '', '', 'Цвет линий: {}', 
                          valid_values=list(LINE_COLORS.keys()))
set_font = make_setter('font_size', int, 9, 18, 'Размер шрифта: {}pt')
set_size = make_setter('max_image_size', int, 800, 4000, 'Макс. размер: {}px')
set_smooth = make_setter('preprocess_strength', str, '', '', 'Сглаживание: {}', 
                        valid_values=['light', 'medium', 'strong'])
set_space = make_setter('spatial_weight', float, 0, 0.5, 'Вес координат: {}')
set_colorspace = make_setter('color_space', str, '', '', 'Цветовое пространство: {}', 
                            valid_values=['rgb', 'lab'])

async def set_svg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle SVG export"""
    if not context.args or context.args[0].lower() not in ['on', 'off', 'true', 'false', '1', '0']:
        await update.message.reply_text('❌ Используйте: <code>/svg on</code> или <code>/svg off</code>', parse_mode='HTML')
        return
    value = context.args[0].lower() in ['on', 'true', '1']
    context.user_data['export_svg'] = value
    await update.message.reply_text(f'✅ SVG экспорт: {"включён" if value else "выключен"}', parse_mode='HTML')

async def set_minibatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle MiniBatchKMeans optimization"""
    if not context.args or context.args[0].lower() not in ['on', 'off', 'true', 'false', '1', '0']:
        await update.message.reply_text('❌ Используйте: <code>/minibatch on</code> или <code>/minibatch off</code>', parse_mode='HTML')
        return
    value = context.args[0].lower() in ['on', 'true', '1']
    context.user_data['use_minibatch'] = value
    await update.message.reply_text(f'✅ MiniBatch оптимизация: {"включена" if value else "выключена"}', parse_mode='HTML')


async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user ID for admin verification"""
    user_id = update.effective_user.id
    is_admin = "✅ Вы админ!" if user_id == ADMIN_ID else "❌ Не админ"
    await update.message.reply_text(
        f'🆔 Ваш ID: <code>{user_id}</code>\n{is_admin}', 
        parse_mode='HTML'
    )


async def update_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin-only bot update trigger"""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text('❌ Только для администратора')
        return
    
    status_msg = await update.message.reply_text('🔄 Запрос обновления...')
    
    if trigger_self_update():
        await status_msg.edit_text('✅ Запрос принят. Бот перезагрузится через ~30 сек.')
    else:
        await status_msg.edit_text('❌ Ошибка обновления. Проверьте логи.')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed help message"""
    await update.message.reply_text(
        '📖 <b>Полная справка v3.0</b>\n\n'
        '<b>🎨 Основные параметры:</b>\n'
        '<code>/colors N</code> — 3-48 цветов (по умолч. 24)\n'
        '<code>/detail N</code> — мин. область 50-500px (по умолч. 150)\n'
        '<code>/smooth X</code> — сглаживание: light/medium/strong\n\n'
        '<b>📐 Визуальные настройки:</b>\n'
        '<code>/line N</code> — толщина контура 1-3px\n'
        '<code>/linecolor X</code> — gray/dark/light/black\n'
        '<code>/font N</code> — размер номера 9-18pt\n'
        '<code>/size N</code> — макс. размер 800-4000px\n\n'
        '<b>⚙️ Продвинутые:</b>\n'
        '<code>/colorspace lab</code> — LAB (качество) или RGB (скорость)\n'
        '<code>/space 0.1</code> — вес координат 0-0.5 (выше = компактнее области)\n'
        '<code>/svg on</code> — экспорт в векторный SVG\n'
        '<code>/minibatch off</code> — отключить оптимизацию для малых изображений\n\n'
        '<b>🔧 Системные:</b>\n'
        '<code>/settings</code> — показать текущие настройки\n'
        '<code>/myid</code> — узнать ID для админ-доступа',
        parse_mode='HTML'
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main image processing handler"""
    message = update.message
    photo = message.photo[-1]
    
    # Get config with defaults
    cfg = PBNConfig.from_user_data(context.user_data)
    
    # Status message with settings summary
    status_text = (
        f'🎨 Обработка...\n'
        f'🎨 {cfg.n_colors} цветов | 📏 мин. {cfg.min_region_size}px\n'
        f'🎨 {cfg.color_space.upper()} | ⚡ {"MB" if cfg.use_minibatch else "Full"}'
    )
    status_msg = await message.reply_text(status_text)
    
    try:
        # Download image
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        
        # Process in thread pool (CPU-bound)
        coloring_buf, palette_buf, svg_output = await asyncio.to_thread(
            process_image_for_coloring, bytes(photo_bytes), cfg
        )
        
        # Send coloring page
        filename = 'coloring_page.svg' if (svg_output and cfg.export_svg) else 'coloring_page.png'
        content_type = 'image/svg+xml' if filename.endswith('.svg') else 'image/png'
        
        if svg_output and cfg.export_svg:
            # Send SVG as document
            svg_buf = io.BytesIO(svg_output.encode('utf-8'))
            await message.reply_document(
                document=svg_buf,
                filename=filename,
                caption=f'🖼️ Раскраска (SVG)\n{status_text}',
                disable_content_type_detection=True
            )
        else:
            # Send PNG
            await message.reply_document(
                document=coloring_buf,
                filename='coloring_page.png',
                caption=f'🖼️ Раскраска (PNG)\n{status_text}'
            )
        
        # Send palette
        await message.reply_photo(palette_buf, caption='🎨 Палитра цветов')
        
        # Tip for small regions
        if cfg.min_region_size < 100:
            await message.reply_text(
                '💡 <i>Совет:</i> Маленькие области (<100px) могут быть сложны для раскрашивания. '
                'Используйте <code>/detail 150</code> для более крупных зон.',
                parse_mode='HTML'
            )
            
    except MemoryError:
        logger.error("Out of memory during processing")
        await message.reply_text(
            '❌ <b>Недостаточно памяти</b>\nПопробуйте уменьшить <code>/size 800</code> или <code>/detail 200</code>',
            parse_mode='HTML'
        )
    except cv2.error as e:
        logger.error(f"OpenCV error: {e}")
        await message.reply_text('❌ Ошибка обработки изображения. Попробуйте другое фото.')
    except asyncio.TimeoutError:
        logger.error("Processing timeout")
        await message.reply_text('⏱️ Таймаут. Попробуйте уменьшить сложность (/detail ↑, /size ↓)')
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        await message.reply_text('❌ Внутренняя ошибка. Попробуйте позже или измените настройки.')
    finally:
        await status_msg.delete()


# ============================================
# APPLICATION SETUP
# ============================================

def create_application() -> Application:
    """Factory for bot application with all handlers"""
    app = Application.builder().token(TOKEN).build()
    
    # Core commands
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('settings', show_settings))
    
    # Parameter setters
    app.add_handler(CommandHandler('colors', set_colors))
    app.add_handler(CommandHandler('detail', set_detail))
    app.add_handler(CommandHandler('line', set_line))
    app.add_handler(CommandHandler('linecolor', set_linecolor))
    app.add_handler(CommandHandler('font', set_font))
    app.add_handler(CommandHandler('size', set_size))
    app.add_handler(CommandHandler('smooth', set_smooth))
    
    # Advanced parameters
    app.add_handler(CommandHandler('space', set_space))
    app.add_handler(CommandHandler('colorspace', set_colorspace))
    app.add_handler(CommandHandler('svg', set_svg))
    app.add_handler(CommandHandler('minibatch', set_minibatch))
    
    # System commands
    app.add_handler(CommandHandler('myid', myid))
    app.add_handler(CommandHandler('update', update_bot))
    
    # Image handler (must be last to not intercept commands)
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
    
    return app


def main() -> None:
    """Entry point"""
    logger.info("🎨 PBN Bot v3.0 Professional starting...")
    logger.info(f"Config: color_space={DEFAULT_CONFIG['color_space']}, "
               f"admin_id={ADMIN_ID}, svg_export={DEFAULT_CONFIG['export_svg']}")
    
    application = create_application()
    
    try:
        logger.info("🚀 Polling started...")
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
