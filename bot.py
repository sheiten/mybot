#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Paint by Numbers Bot v4.0 (Hybrid Algorithm)
Telegram bot for generating paint-by-number coloring pages.
Uses SLIC + Mean Shift + Custom RAG Merging.
"""

import asyncio
import logging
import os
import io
import math
import ssl
import urllib.request
import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, estimate_bandwidth
from skimage.segmentation import slic
import cv2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ============================================
# GLOBAL EXCEPTION HANDLER (Для отладки в Docker)
# ============================================
def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("💥 FATAL ERROR STARTING BOT:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

sys.excepthook = global_exception_handler

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================

TOKEN = os.environ.get('BOT_TOKEN', '')
if not TOKEN:
    raise ValueError('BOT_TOKEN environment variable is not set!')

PORTAINER_WEBHOOK_URL = os.environ.get('PORTAINER_WEBHOOK_URL', '')
PORTAINER_TOKEN = os.environ.get('PORTAINER_TOKEN', '')
ADMIN_ID = int(os.environ.get('ADMIN_USER_ID', '931848809'))

DEFAULT_CONFIG = {
    'n_colors': 16,           # Оптимально для нового алгоритма
    'min_region_size': 180,   # Чуть больше для чистоты
    'max_image_size': 1200,   # Баланс скорости/качества
    'line_thickness': 1,
    'line_color': 'gray',
    'font_size': 12,
    'preprocess_strength': 'medium',
    'color_space': 'lab',
    'spatial_weight': 0.12,
    'use_minibatch': True,
    'export_svg': False,
}

LINE_COLORS = {
    'gray': (180, 180, 180),
    'dark': (100, 100, 100),
    'light': (210, 210, 210),
    'black': (0, 0, 0),
}

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
    n_colors: int = 16
    min_region_size: int = 180
    max_image_size: int = 1200
    line_thickness: int = 1
    line_color: str = 'gray'
    font_size: int = 12
    preprocess_strength: str = 'medium'
    color_space: str = 'lab'
    spatial_weight: float = 0.12
    use_minibatch: bool = True
    export_svg: bool = False
    
    @property
    def line_rgb(self) -> Tuple[int, int, int]:
        return LINE_COLORS.get(self.line_color, LINE_COLORS['gray'])
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PBNConfig':
        valid_keys = cls.__annotations__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_keys})
    
    @classmethod
    def from_user_data(cls, user_data: dict) -> 'PBNConfig':
        return cls.from_dict(user_data)


# ============================================
# UTILS & SECURITY
# ============================================

def trigger_self_update() -> bool:
    if not PORTAINER_WEBHOOK_URL or not PORTAINER_TOKEN:
        return False
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            PORTAINER_WEBHOOK_URL, method='POST',
            headers={'Authorization': f'Bearer {PORTAINER_TOKEN}', 'Content-Type': 'application/json'},
            data=json.dumps({'action': 'redeploy'}).encode('utf-8')
        )
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            return response.status in (200, 204)
    except Exception as e:
        logger.error(f"Update webhook error: {e}")
        return False

def get_font(size: int) -> ImageFont.FreeTypeFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "Arial.ttf", "arial.ttf", "DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(str(font_path), size)
        except:
            continue
    try:
        return ImageFont.load_default()
    except:
        return None


# ============================================
# 🎨 ADVANCED GENERATOR ENGINE (Встроено)
# ============================================

def rgb2lab_batch(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

def lab2rgb_batch(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def get_pole_of_inaccessibility(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Нахождение точки, максимально удалённой от границы региона"""
    if not np.any(mask):
        return None
    
    # Distance transform
    # Убедимся, что маска uint8
    mask_uint8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_5)
    
    # Находим максимум
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    
    if max_val <= 0:
        return None
        
    x, y = max_loc  # OpenCV возвращает (x, y)
    h, w = mask.shape
    
    # Строгая проверка границ
    if 0 <= x < w and 0 <= y < h:
        if mask[y, x]:
            return (int(x), int(y))
            
    return None

def segment_image_advanced(img_array: np.ndarray, config: PBNConfig) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int,int,int]]]:
    h, w = img_array.shape[:2]
    img_smooth = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    img_lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_lab[:, :, 0] /= 100.0
    img_lab[:, :, 1:] /= 255.0
    
    target_segments = min(config.n_colors * 8, 500)
    segments = slic(img_lab, n_segments=target_segments, compactness=20.0, sigma=2.0, start_label=1, enforce_connectivity=True)
    
    segment_colors_lab = []
    unique_segs = np.unique(segments)
    for seg_id in unique_segs:
        if seg_id == 0: continue
        mask = (segments == seg_id)
        if np.sum(mask) < 10: continue
        segment_colors_lab.append(np.mean(img_lab[mask], axis=0))
    
    if len(segment_colors_lab) == 0:
        pixels = img_lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=config.n_colors, random_state=42, n_init=5)
        labels_flat = kmeans.fit_predict(pixels)
        centers_lab = kmeans.cluster_centers_
        quantized_lab = centers_lab[labels_flat].reshape(h, w, 3)
    else:
        colors_array = np.array(segment_colors_lab)
        if len(colors_array) > 100:
            sample_idx = np.random.choice(len(colors_array), 100, replace=False)
            bandwidth = estimate_bandwidth(colors_array[sample_idx], quantile=0.3, n_samples=50)
        else:
            bandwidth = estimate_bandwidth(colors_array, quantile=0.3)
        
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=2)
        ms.fit(colors_array)
        cluster_centers_lab = ms.cluster_centers_
        segment_to_cluster = ms.predict(colors_array)
        
        quantized_lab = np.zeros_like(img_lab)
        valid_seg_ids = [s for s in unique_segs if s != 0]
        for i, seg_id in enumerate(valid_seg_ids):
            if i >= len(segment_to_cluster): break
            mask = (segments == seg_id)
            cluster_idx = segment_to_cluster[i]
            quantized_lab[mask] = cluster_centers_lab[cluster_idx]

    quantized_lab[:, :, 0] *= 100.0
    quantized_lab[:, :, 1:] *= 255.0
    quantized_rgb = lab2rgb_batch(quantized_lab).astype(np.uint8)
    
    for c in range(3):
        quantized_rgb[:, :, c] = cv2.medianBlur(quantized_rgb[:, :, c], 3)
    
    unique_colors = np.unique(quantized_rgb.reshape(-1, 3), axis=0)
    if len(unique_colors) > config.n_colors:
        kmeans = KMeans(n_clusters=config.n_colors, random_state=42, n_init=3)
        kmeans.fit(unique_colors.astype(np.float32))
        centers = kmeans.cluster_centers_.astype(np.uint8)
        flat = quantized_rgb.reshape(-1, 3).astype(np.float32)
        labels = kmeans.predict(flat)
        quantized_rgb = centers[labels].reshape(h, w, 3)
        palette = [tuple(c) for c in centers]
    else:
        palette = [tuple(c) for c in unique_colors]
    
    brightness = [0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in palette]
    sorted_idx = np.argsort(brightness)
    palette = [palette[i] for i in sorted_idx]
    
    label_map = np.zeros((h, w), dtype=np.int32)
    for new_idx, color in enumerate(palette):
        mask = np.all(quantized_rgb == color, axis=2)
        label_map[mask] = new_idx
        
    return quantized_rgb, label_map, palette

def remove_thin_regions_scan(quantized: np.ndarray, min_length: int = 7, iterations: int = 3) -> np.ndarray:
    result = quantized.copy()
    h, w = result.shape[:2]
    for _ in range(iterations):
        for transpose in [False, True]:
            if transpose:
                result = result.transpose(1, 0, 2)
                h, w = result.shape[:2]  # ✅ ВАЖНО: обновляем h,w после транспонирования
            for row in range(h):
                line = result[row]
                transitions = np.any(line[:-1] != line[1:], axis=1)
                boundaries = np.where(transitions)[0] + 1
                boundaries = np.concatenate([[0], boundaries, [w]])
                for i in range(1, len(boundaries) - 1):
                    start, end = boundaries[i], boundaries[i+1]
                    if end - start < min_length:
                        left_len = start - boundaries[i-1]
                        if i + 2 < len(boundaries):
                            right_len = boundaries[i+2] - end
                        else:
                            right_len = 0
                        
                        # ✅ ИСПРАВЛЕНИЕ: безопасный доступ к индексам
                        left_color = line[max(0, start - 1)]
                        right_idx = min(end, w - 1)
                        right_color = line[right_idx]
                        
                        fill_color = left_color if left_len >= right_len else right_color
                        result[row, start:end] = fill_color
            if transpose:
                result = result.transpose(1, 0, 2)
                h, w = result.shape[:2]  # ✅ ВАЖНО: обновляем h,w после обратного транспонирования
    return result
def merge_regions_rag(quantized: np.ndarray, labels: np.ndarray, 
                      palette: List[Tuple[int,int,int]], 
                      min_area: int, target_regions: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int,int,int]]]:
    """
    Быстрое слияние регионов с полной проверкой границ.
    """
    if target_regions is None:
        target_regions = len(palette) * 4

    h, w = quantized.shape[:2]
    current_labels = labels.copy()
    current_quantized = quantized.copy()
    current_palette = list(palette)
    
    if len(current_palette) <= 1:
        return current_quantized, current_labels, current_palette

    # Предварительный расчет LAB цветов палитры
    try:
        palette_array = np.uint8([[c for c in current_palette]])
        palette_lab = cv2.cvtColor(palette_array, cv2.COLOR_RGB2LAB).astype(np.float32).squeeze()
        if palette_lab.ndim == 1:
            palette_lab = palette_lab.reshape(1, -1)
    except Exception as e:
        logger.error(f"Palette conversion error: {e}")
        return current_quantized, current_labels, current_palette
    
    max_iterations = 10
    for iteration in range(max_iterations):
        logger.info(f"RAG iteration {iteration + 1}/{max_iterations}")  # ДОБАВИТЬ ЛОГ
        
        unique_colors_in_use = np.unique(current_labels)
        
        # Создаем карту уникальных ID регионов
        temp_label_map = np.zeros((h, w), dtype=np.int32)
        region_info = {}
        next_region_id = 1
        
        for color_idx in unique_colors_in_use:
            if color_idx < 0 or color_idx >= len(current_palette):
                continue
                
            mask = (current_labels == color_idx).astype(np.uint8)
            num, comps, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)
            
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] < 5: 
                    continue
                reg_id = next_region_id
                region_mask = (comps == i)
                
                if region_mask.shape != (h, w):
                    continue
                    
                temp_label_map[region_mask] = reg_id
                region_info[reg_id] = {
                    'color_idx': int(color_idx),
                    'area': int(stats[i, cv2.CC_STAT_AREA]),
                    'mask': region_mask
                }
                next_region_id += 1

        total_regions = len(region_info)
        logger.info(f"Current regions: {total_regions}, target: {target_regions}")  # ДОБАВИТЬ ЛОГ
        
        if total_regions <= target_regions:
            break
            
        # ИСПРАВЛЕНИЕ: создаем список ID для обработки
        sorted_regions = sorted(region_info.items(), key=lambda x: x[1]['area'])
        merged_ids = set()  # Отслеживаем уже смерженные ID
        merged_count = 0
        
        # Обрабатываем только маленькие регионы
        for reg_id, info in sorted_regions[:50]:
            if reg_id in merged_ids:
                continue
                
            # Проверка площади
            all_areas = [r['area'] for r in region_info.values()]
            if not all_areas:
                break
            median_area = np.median(all_areas)
            
            if info['area'] > median_area * 2:
                continue
                
            mask = info['mask']
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel)
            boundary = dilated - mask.astype(np.uint8)
            
            if np.sum(boundary) == 0:
                continue
            
            try:
                neighbor_ids = np.unique(temp_label_map[boundary > 0])
                neighbor_ids = [nid for nid in neighbor_ids 
                               if nid != reg_id and nid != 0 and nid in region_info]
            except Exception:
                continue
                
            if not neighbor_ids:
                continue
            
            current_color_idx = info['color_idx']
            if current_color_idx >= len(palette_lab):
                continue
            
            current_lab = palette_lab[current_color_idx]
            
            best_neighbor_id = None
            min_dist = float('inf')
            
            # Ищем лучшего соседа (только среди не смерженных)
            for nb_id in neighbor_ids:
                if nb_id in merged_ids:
                    continue
                    
                nb_info = region_info[nb_id]
                nb_color_idx = nb_info['color_idx']
                
                if nb_color_idx >= len(palette_lab):
                    continue
                    
                nb_lab = palette_lab[nb_color_idx]
                dist = np.linalg.norm(current_lab - nb_lab)
                
                # Эвристика
                if nb_info['area'] < 1:
                    continue
                score = dist / (nb_info['area'])
                
                if score < min_dist:
                    min_dist = score
                    best_neighbor_id = nb_id
            
            if best_neighbor_id is not None and best_neighbor_id in region_info:
                target_color_idx = region_info[best_neighbor_id]['color_idx']
                
                if target_color_idx < len(current_palette):
                    # Мержим регионы
                    current_labels[mask] = target_color_idx
                    current_quantized[mask] = current_palette[target_color_idx]
                    merged_ids.add(reg_id)  # Отмечаем как смерженный
                    merged_count += 1
                    logger.debug(f"Merged region {reg_id} into {best_neighbor_id}")  # ДОБАВИТЬ ЛОГ
        
        logger.info(f"Merged {merged_count} regions in iteration {iteration + 1}")  # ДОБАВИТЬ ЛОГ
        
        if merged_count == 0:
            logger.info("No more regions to merge, stopping")  # ДОБАВИТЬ ЛОГ
            break
            
    # Финальная очистка
    final_unique_colors = np.unique(current_quantized.reshape(-1, 3), axis=0)
    final_palette = [tuple(c) for c in final_unique_colors]
    
    final_label_map = np.zeros((h, w), dtype=np.int32)
    for new_idx, color in enumerate(final_palette):
        mask = np.all(current_quantized == color, axis=2)
        final_label_map[mask] = new_idx
        
    return current_quantized, final_label_map, final_palette

def create_coloring_page_raster(
   quantized: np.ndarray,
   palette: List[Tuple[int, int, int]],
   config: PBNConfig
) -> io.BytesIO:

   h, w = quantized.shape[:2]
   canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
   line_rgb = config.line_rgb
   font = get_font(config.font_size)

   # --------------------------------------------------
   # label map
   # --------------------------------------------------
   label_map = np.zeros((h, w), dtype=np.int32)

   for idx, color in enumerate(palette):
       mask = np.all(quantized == color, axis=2)
       label_map[mask] = idx + 1

   # --------------------------------------------------
   # ГРАНИЦЫ (единые, без двойных жирных линий)
   # --------------------------------------------------
   gx = np.abs(np.diff(label_map, axis=1, prepend=label_map[:, :1]))
   gy = np.abs(np.diff(label_map, axis=0, prepend=label_map[:1, :]))

   edges = ((gx > 0) | (gy > 0)).astype(np.uint8) * 255

   if config.line_thickness > 1:
       kernel = np.ones((3, 3), np.uint8)
       edges = cv2.dilate(edges, kernel, iterations=config.line_thickness - 1)

   # --------------------------------------------------
   # НОМЕРА
   # --------------------------------------------------
   placed_mask = np.zeros((h, w), dtype=np.uint8)

   for color_idx, color in enumerate(palette):

       color_mask = np.all(quantized == color, axis=2).astype(np.uint8)

       num, labels_img, stats, _ = cv2.connectedComponentsWithStats(
           color_mask,
           connectivity=8
       )

       for comp_id in range(1, num):

           area = stats[comp_id, cv2.CC_STAT_AREA]

           if area < config.min_region_size:
               continue

           comp_mask = (labels_img == comp_id).astype(np.uint8)

           pole = get_pole_of_inaccessibility(comp_mask)

           if pole is None:
               continue

           cx, cy = pole

           # Исправление выхода за границы массива
           cx = min(max(int(cx), 0), w - 1)
           cy = min(max(int(cy), 0), h - 1)

           # ------------------------------------------
           # Авторазмер шрифта
           # ------------------------------------------
           local_font_size = int(
               max(
                   9,
                   min(
                       config.font_size + 8,
                       math.sqrt(area) * 0.22
                   )
               )
           )

           local_font = get_font(local_font_size)
           num_str = str(color_idx + 1)

           pil_img = Image.fromarray(canvas)
           draw = ImageDraw.Draw(pil_img)

           try:
               bbox = draw.textbbox((0, 0), num_str, font=local_font)
               text_w = bbox[2] - bbox[0]
               text_h = bbox[3] - bbox[1]
           except:
               text_w, text_h = draw.textsize(num_str, font=local_font)

           pad = 3

           x1 = max(0, cx - text_w // 2 - pad)
           y1 = max(0, cy - text_h // 2 - pad)
           x2 = min(w, cx + text_w // 2 + pad)
           y2 = min(h, cy + text_h // 2 + pad)

           # Приведение к int и повторная проверка границ
           x1 = int(max(0, x1))
           y1 = int(max(0, y1))
           x2 = int(min(w - 1, x2))
           y2 = int(min(h - 1, y2))

           # Проверка на пустой срез
           if x2 <= x1 or y2 <= y1:
               continue

           # уже занято?
           if np.any(placed_mask[y1:y2, x1:x2]):
               continue

           # текст не должен лечь на границу
           if np.mean(edges[y1:y2, x1:x2]) > 20:
               continue

           draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
           draw.text(
               (cx - text_w // 2, cy - text_h // 2),
               num_str,
               fill="black",
               font=local_font
           )

           canvas = np.array(pil_img)

           placed_mask[y1:y2, x1:x2] = 1

   # --------------------------------------------------
   # Контуры поверх всего
   # --------------------------------------------------
   canvas[edges > 0] = line_rgb

   output = io.BytesIO()
   Image.fromarray(canvas).save(output, format="PNG", dpi=(300, 300))
   output.seek(0)

   return output

def generate_svg_output(quantized: np.ndarray, palette: List[Tuple[int,int,int]], config: PBNConfig) -> str:
    h, w = quantized.shape[:2]
    line_rgb = config.line_rgb
    font_size = config.font_size
    svg_parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<style>.region{{fill:none;stroke:rgb{line_rgb};stroke-width:{config.line_thickness};stroke-linejoin:round}}</style>',
    ]
    placed_positions = []
    label_map = np.zeros((h, w), dtype=np.int32)
    for color_idx, color in enumerate(palette):
        mask = np.all(quantized == color, axis=2)
        label_map[mask] = color_idx + 1
        
    grad_x = np.abs(np.diff(label_map, axis=1, prepend=label_map[:, :1]))
    grad_y = np.abs(np.diff(label_map, axis=0, prepend=label_map[:1, :]))
    boundary_mask = ((grad_x > 0) | (grad_y > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < config.min_region_size * 0.5: continue
        simplified = cv2.approxPolyDP(contour, 0.5, True)
        if len(simplified) < 3: continue
        path_d = f"M {simplified[0][0][0]},{simplified[0][0][1]}"
        for point in simplified[1:]:
            x, y = point[0]
            path_d += f" L {x},{y}"
        path_d += " Z"
        svg_parts.append(f'<path class="region" d="{path_d}"/>')
        
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
        num, labels_img, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=4, ltype=cv2.CV_32S)
        for comp_id in range(1, num):
            if stats[comp_id, cv2.CC_STAT_AREA] < config.min_region_size: continue
            comp_mask = (labels_img == comp_id).astype(np.uint8)
            pole = get_pole_of_inaccessibility(comp_mask)
            if pole is None: continue
            cx, cy = pole
            if any(math.hypot(cx - px, cy - py) < font_size * 2.5 for px, py in placed_positions): continue
            num_str = str(color_idx + 1)
            svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{font_size//2 + 3}" fill="white" stroke="none"/>')
            svg_parts.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" font-size="{font_size}" font-family="Arial" fill="black">{num_str}</text>')
            placed_positions.append((cx, cy))
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)

def create_palette_image(palette: List[Tuple[int, int, int]], config: PBNConfig) -> Image.Image:
    n_colors = len(palette)
    palette_width = 320
    palette_height = 80 + n_colors * 40
    palette_img = Image.new('RGB', (palette_width, palette_height), 'white')
    draw = ImageDraw.Draw(palette_img)
    font = get_font(13)
    title_font = get_font(16)
    draw.text((15, 15), "🎨 ПАЛИТРА ЦВЕТОВ", fill='black', font=title_font)
    draw.text((15, 38), f"Цветов: {n_colors}", fill='gray', font=font)
    for idx, color in enumerate(palette, start=1):
        y_pos = 65 + (idx - 1) * 40
        draw.rectangle([(15, y_pos), (48, y_pos + 30)], fill=color, outline=(200, 200, 200), width=2)
        draw.text((58, y_pos + 6), f"{idx}.", fill='black', font=font)
        hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'.upper()
        draw.text((95, y_pos + 6), hex_color, fill=(100, 100, 100), font=font)
    return palette_img

def process_image_for_coloring(
 photo_bytes: bytes,
    config: PBNConfig
) -> Tuple[io.BytesIO, io.BytesIO, Optional[str]]:
    """
    Профессиональный пайплайн: SLIC + K-Means палитра + постобработка
    """
    image = Image.open(io.BytesIO(photo_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 1. Ресайз
    width, height = image.size
    if max(width, height) > config.max_image_size:
        ratio = config.max_image_size / max(width, height)
        image = image.resize(
            (int(width * ratio), int(height * ratio)),
            Image.Resampling.LANCZOS
        )
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # 2. Легкая фильтрация
    img_smooth = cv2.bilateralFilter(img_array, d=7, sigmaColor=50, sigmaSpace=50)
    
    # 3. SLIC сегментация — создаем суперпиксели
    img_lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_lab[:, :, 0] /= 100.0
    img_lab[:, :, 1:] /= 255.0
    
    n_segments = min(config.n_colors * 30, 1500)
    segments = slic(
        img_lab,
        n_segments=n_segments,
        compactness=15.0,
        sigma=1.5,
        start_label=1,
        enforce_connectivity=True
    )
    
    logger.info(f"SLIC created {len(np.unique(segments))} superpixels")
    
    # 4. Собираем средние цвета суперпикселей
    unique_segs = np.unique(segments)
    superpixel_colors = []
    superpixel_masks = []
    
    for seg_id in unique_segs:
        if seg_id == 0:
            continue
        mask = (segments == seg_id)
        if np.sum(mask) < 20:  # игнорируем крошечные
            continue
        avg_color = np.mean(img_array[mask], axis=0)
        superpixel_colors.append(avg_color)
        superpixel_masks.append((seg_id, mask))
    
    logger.info(f"Valid superpixels: {len(superpixel_colors)}")
    
    # 5. K-Means для итоговой палитры
    superpixel_colors = np.array(superpixel_colors)
    kmeans = MiniBatchKMeans(
        n_clusters=config.n_colors,
        random_state=42,
        batch_size=256,
        n_init=5
    )
    cluster_labels = kmeans.fit_predict(superpixel_colors)
    palette_rgb = kmeans.cluster_centers_.astype(np.uint8)
    
    # 6. Сортировка палитры по яркости
    brightness = np.dot(palette_rgb, [0.299, 0.587, 0.114])
    sorted_idx = np.argsort(brightness)
    palette = [tuple(palette_rgb[i]) for i in sorted_idx]
    
    # 7. Переназначаем цвета суперпикселей
    quantized = np.zeros_like(img_array)
    label_map = np.zeros((h, w), dtype=np.int32)
    
    for i, (seg_id, mask) in enumerate(superpixel_masks):
        cluster = cluster_labels[i]
        new_idx = list(sorted_idx).index(cluster)
        quantized[mask] = palette[new_idx]
        label_map[mask] = new_idx + 1
    
    # 8. Сглаживание границ
    for c in range(3):
        quantized[:, :, c] = cv2.medianBlur(quantized[:, :, c], 3)
    
    logger.info(f"Final palette: {len(palette)} colors")
    
    # 9. Рендерим раскраску и палитру
    coloring_buffer = create_coloring_page_raster(quantized, palette, config)
    palette_buffer = io.BytesIO()
    palette_img = create_palette_image(palette, config)
    palette_img.save(palette_buffer, format="PNG", dpi=(300, 300))
    palette_buffer.seek(0)
    
    return coloring_buffer, palette_buffer, None
    
# ============================================
# TELEGRAM BOT HANDLERS
# ============================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '🎨 <b>Paint by Numbers Bot v4.0</b>\n\n'
        'Отправьте фото — получите раскраску!\n\n'
        '<b>Настройки:</b>\n'
        '<code>/colors 16</code> (3-48)\n'
        '<code>/detail 180</code> (50-500)\n'
        '<code>/settings</code>',
        parse_mode='HTML'
    )

async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = PBNConfig.from_user_data(context.user_data)
    settings = (
        f'⚙️ <b>Настройки:</b>\n'
        f'🎨 Цветов: {cfg.n_colors}\n'
        f'📏 Мин. область: {cfg.min_region_size}px\n'
        f'🖼️ Размер: {cfg.max_image_size}px\n'
        f'📄 SVG: {"✅" if cfg.export_svg else "❌"}'
    )
    await update.message.reply_text(settings, parse_mode='HTML')

def make_setter(param_name: str, param_type: type, min_val, max_val, success_msg: str, valid_values: Optional[List[str]] = None):
    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not context.args:
            await update.message.reply_text(f'❌ Используйте: <code>/{param_name} значение</code>', parse_mode='HTML')
            return
        arg = context.args[0].lower()
        if valid_values:
            if arg not in valid_values:
                await update.message.reply_text(f'❌ Допустимо: {", ".join(valid_values)}', parse_mode='HTML')
                return
            value = arg
        else:
            try:
                value = param_type(arg)
                if not (min_val <= value <= max_val): raise ValueError
            except ValueError:
                await update.message.reply_text(f'❌ Число от {min_val} до {max_val}', parse_mode='HTML')
                return
        context.user_data[param_name] = value
        await update.message.reply_text(f'✅ {success_msg.format(value)}', parse_mode='HTML')
    return handler

set_colors = make_setter('n_colors', int, 3, 48, 'Установлено {} цветов')
set_detail = make_setter('min_region_size', int, 50, 500, 'Мин. область: {}px')
set_size = make_setter('max_image_size', int, 800, 4000, 'Макс. размер: {}px')
set_svg = make_setter('export_svg', lambda x: x.lower() in ['on','true','1'], 0, 1, 'SVG: {}', valid_values=['on','off','true','false','1','0'])

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    photo = message.photo[-1]
    cfg = PBNConfig.from_user_data(context.user_data)
    
    status_msg = await message.reply_text('🎨 Обработка...')
    try:
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        coloring_buf, palette_buf, svg_output = await asyncio.to_thread(process_image_for_coloring, bytes(photo_bytes), cfg)
        
        await message.reply_document(document=coloring_buf, filename='coloring_page.png', caption='🖼️ Раскраска')
        await message.reply_photo(palette_buf, caption='🎨 Палитра')
        if svg_output and cfg.export_svg:
            svg_buf = io.BytesIO(svg_output.encode('utf-8'))
            await message.reply_document(document=svg_buf, filename='coloring_page.svg', caption='📐 SVG версия')
    except Exception as e:
        logger.exception(f"Error: {e}")
        await message.reply_text(f'❌ Ошибка: {str(e)}')
    finally:
        await status_msg.delete()

async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'ID: <code>{update.effective_user.id}</code>', parse_mode='HTML')

async def update_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text('❌ Только админ')
        return
    status_msg = await update.message.reply_text('🔄 Обновление...')
    if trigger_self_update():
        await status_msg.edit_text('✅ Запрос отправлен')
    else:
        await status_msg.edit_text('❌ Ошибка')

def main() -> None:
    logger.info("🚀 Starting Bot v4.0...")
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('settings', show_settings))
    application.add_handler(CommandHandler('colors', set_colors))
    application.add_handler(CommandHandler('detail', set_detail))
    application.add_handler(CommandHandler('size', set_size))
    application.add_handler(CommandHandler('svg', set_svg))
    application.add_handler(CommandHandler('myid', myid))
    application.add_handler(CommandHandler('update', update_bot))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
