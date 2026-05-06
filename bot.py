#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Paint by Numbers — Advanced Generator Module
Гибридный алгоритм: SLIC → Mean Shift → RAG Merging → Smart Placement

Заменяет функции в вашем боте:
• cluster_colors()
• merge_small_regions() 
• create_coloring_page_raster()
• generate_svg_output()

Добавляет:
• remove_thin_regions_scan()
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import slic, mark_boundaries
from skimage.future import graph
from scipy import ndimage as ndi
from typing import List, Tuple, Optional, Dict
import io
import math
import logging

logger = logging.getLogger(__name__)


# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def rgb2lab_batch(rgb: np.ndarray) -> np.ndarray:
    """Быстрая конвертация RGB→LAB для массивов"""
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def lab2rgb_batch(lab: np.ndarray) -> np.ndarray:
    """Обратная конвертация"""
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def get_pole_of_inaccessibility(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Нахождение точки, максимально удалённой от границы региона"""
    if not np.any(mask):
        return None
    
    # Distance transform
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    
    # Проверка: точка должна быть внутри маски
    if mask[max_loc[1], max_loc[0]]:
        return max_loc  # (x, y)
    return None


def safe_get_font(size: int):
    """Безопасная загрузка шрифта (из вашего бота)"""
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
# 🔄 ЗАМЕНА: cluster_colors() → segment_image_advanced()
# ============================================

def segment_image_advanced(img_array: np.ndarray, config) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int,int,int]]]:
    """
    ГИБРИДНЫЙ АЛГОРИТМ сегментации:
    1. SLIC superpixels → пространственная когерентность
    2. Mean Shift в LAB → адаптивное квантование цветов
    3. Пост-обработка → очистка артефактов
    
    Returns: (quantized_image, segment_labels, palette_rgb)
    """
    h, w = img_array.shape[:2]
    
    # 🔹 Шаг 1: Предобработка
    # Bilateral filter для сохранения границ при сглаживании
    img_smooth = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Конвертация в LAB для лучшего восприятия цвета
    img_lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Нормализация: L∈[0,100], a,b∈[0,255]
    img_lab[:, :, 0] /= 100.0
    img_lab[:, :, 1:] /= 255.0
    
    # 🔹 Шаг 2: SLIC Superpixels (пространственная сегментация)
    # n_segments ≈ желаемое количество регионов × коэффициент запаса
    target_segments = min(config.n_colors * 8, 500)  # не больше 500 для скорости
    compactness = 20.0  # баланс цвет/пространство
    
    segments = slic(
        img_lab, 
        n_segments=target_segments, 
        compactness=compactness,
        sigma=2.0,  # дополнительное сглаживание перед сегментацией
        start_label=1,  # 0 зарезервирован для фона
        enforce_connectivity=True,
        max_num_iter=10
    )
    
    # 🔹 Шаг 3: Адаптивное квантование цветов через Mean Shift
    # Собираем средний цвет каждого суперпикселя в LAB
    segment_colors_lab = []
    for seg_id in np.unique(segments):
        mask = (segments == seg_id)
        if np.sum(mask) < 10:  # пропускаем микро-сегменты
            continue
        mean_color = np.mean(img_lab[mask], axis=0)
        segment_colors_lab.append(mean_color)
    
    if len(segment_colors_lab) == 0:
        # Fallback: простой K-Means если Mean Shift не сработал
        from sklearn.cluster import KMeans
        pixels = img_lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=config.n_colors, random_state=42, n_init=5)
        labels_flat = kmeans.fit_predict(pixels)
        centers_lab = kmeans.cluster_centers_
        quantized_lab = centers_lab[labels_flat].reshape(h, w, 3)
    else:
        # Mean Shift: автоматически определяет количество кластеров
        from sklearn.cluster import MeanShift, estimate_bandwidth
        
        colors_array = np.array(segment_colors_lab)
        
        # Оценка bandwidth: квантиль попарных расстояний
        if len(colors_array) > 100:
            sample_idx = np.random.choice(len(colors_array), 100, replace=False)
            bandwidth = estimate_bandwidth(colors_array[sample_idx], quantile=0.3, n_samples=50)
        else:
            bandwidth = estimate_bandwidth(colors_array, quantile=0.3)
        
        # Mean Shift кластеризация
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=2)
        ms.fit(colors_array)
        cluster_centers_lab = ms.cluster_centers_
        
        # Маппинг суперпикселей → кластеры
        segment_to_cluster = ms.predict(colors_array)
        
        # Построение квантизованного изображения
        quantized_lab = np.zeros_like(img_lab)
        for seg_id in np.unique(segments):
            if seg_id == 0:
                continue
            mask = (segments == seg_id)
            # Находим индекс суперпикселя в colors_array
            idx = np.where(np.array([s for s in np.unique(segments) if s != 0]) == seg_id)[0]
            if len(idx) > 0 and idx[0] < len(segment_to_cluster):
                cluster_idx = segment_to_cluster[idx[0]]
                quantized_lab[mask] = cluster_centers_lab[cluster_idx]
    
    # 🔹 Шаг 4: Обратная конвертация + пост-обработка
    quantized_lab[:, :, 0] *= 100.0
    quantized_lab[:, :, 1:] *= 255.0
    quantized_rgb = lab2rgb_batch(quantized_lab).astype(np.uint8)
    
    # Медианный фильтр для устранения шума на границах
    for c in range(3):
        quantized_rgb[:, :, c] = cv2.medianBlur(quantized_rgb[:, :, c], 3)
    
    # 🔹 Шаг 5: Извлечение палитры и маппинг
    unique_colors = np.unique(quantized_rgb.reshape(-1, 3), axis=0)
    
    # Ограничиваем палитру до config.n_colors через дополнительное слияние
    if len(unique_colors) > config.n_colors:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=config.n_colors, random_state=42, n_init=3)
        kmeans.fit(unique_colors.astype(np.float32))
        centers = kmeans.cluster_centers_.astype(np.uint8)
        
        # Перемаппинг пикселей к ближайшему центру
        flat = quantized_rgb.reshape(-1, 3).astype(np.float32)
        labels = kmeans.predict(flat)
        quantized_rgb = centers[labels].reshape(h, w, 3)
        palette = [tuple(c) for c in centers]
    else:
        palette = [tuple(c) for c in unique_colors]
    
    # Сортировка палитры по яркости для интуитивной нумерации
    brightness = [0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in palette]
    sorted_idx = np.argsort(brightness)
    palette = [palette[i] for i in sorted_idx]
    
    # Создание label map: pixel → color_index
    label_map = np.zeros((h, w), dtype=np.int32)
    for new_idx, color in enumerate(palette):
        mask = np.all(quantized_rgb == color, axis=2)
        label_map[mask] = new_idx
    
    logger.info(f"🎨 Сегментация: {len(palette)} цветов, ~{len(np.unique(label_map))} регионов")
    
    return quantized_rgb, label_map, palette


# ============================================
# ✂️ НОВАЯ: remove_thin_regions_scan()
# ============================================

def remove_thin_regions_scan(quantized: np.ndarray, min_length: int = 7, iterations: int = 3) -> np.ndarray:
    """
    Удаление тонких регионов сканированием по строкам/столбцам (Axecrafted approach)
    Эффективно убирает «нити», ветки, артефакты без сложных дескрипторов формы.
    """
    result = quantized.copy()
    h, w = result.shape[:2]
    
    for _ in range(iterations):
        for transpose in [False, True]:  # horizontal → vertical
            if transpose:
                result = result.transpose(1, 0, 2)
                h, w = w, h
            
            for row in range(h):
                line = result[row]
                # Находим границы цветовых переходов
                transitions = np.any(line[:-1] != line[1:], axis=1)
                boundaries = np.where(transitions)[0] + 1
                boundaries = np.concatenate([[0], boundaries, [w]])
                
                # Обрабатываем внутренние сегменты
                for i in range(1, len(boundaries) - 1):
                    start, end = boundaries[i], boundaries[i+1]
                    length = end - start
                    if length < min_length:
                        # Выбираем цвет БОЛЬШЕГО соседа
                        left_len = start - boundaries[i-1]
                        right_len = boundaries[i+2] - end if i+2 < len(boundaries) else 0
                        fill_color = line[start-1] if left_len >= right_len else line[end]
                        result[row, start:end] = fill_color
            
            if transpose:
                result = result.transpose(1, 0, 2)  # обратно
    
    return result


# ============================================
# 🔗 ЗАМЕНА: merge_small_regions() → merge_regions_rag()
# ============================================

def merge_regions_rag(quantized: np.ndarray, labels: np.ndarray, 
                      palette: List[Tuple[int,int,int]], 
                      min_area: int, target_regions: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Умное слияние регионов через Region Adjacency Graph (RAG)
    Эвристика: площадь × цветовое расстояние / количество соседей
    """
    if target_regions is None:
        target_regions = len(palette) * 3  # эвристика: ~3 региона на цвет
    
    h, w = quantized.shape[:2]
    new_labels = labels.copy()
    new_quantized = quantized.copy()
    
    # 🔹 Шаг 1: Удаление микро-регионов через connected components
    for color_idx in range(len(palette)):
        color_mask = (labels == color_idx).astype(np.uint8)
        num, comp_labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=4, ltype=cv2.CV_32S)
        
        for comp_id in range(1, num):
            if stats[comp_id, cv2.CC_STAT_AREA] < min_area // 3:
                comp_mask = (comp_labels == comp_id)
                # Найти соседей через dilate
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated = cv2.dilate(comp_mask.astype(np.uint8), kernel)
                boundary = dilated - comp_mask.astype(np.uint8)
                
                if np.sum(boundary) > 0:
                    neighbor_labels = new_labels[boundary > 0]
                    if len(neighbor_labels) > 0:
                        dominant = np.bincount(neighbor_labels.flatten()).argmax()
                        new_labels[comp_mask] = dominant
                        new_quantized[comp_mask] = palette[dominant]
    
    # 🔹 Шаг 2: Построение RAG для умного слияния
    for iteration in range(5):  # максимум 5 итераций
        # Подсчёт текущих регионов
        region_data = []
        for color_idx in range(len(palette)):
            mask = (new_labels == color_idx).astype(np.uint8)
            num, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=4, ltype=cv2.CV_32S)
            
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= min_area // 2:
                    region_data.append({
                        'color_idx': color_idx,
                        'comp_id': i,
                        'area': stats[i, cv2.CC_STAT_AREA],
                        'centroid': centroids[i],
                        'mask': (comp_labels == i)
                    })
        
        total_regions = len(region_data)
        if total_regions <= target_regions:
            break
        
        # Построение графа смежности регионов
        # Для каждого региона находим соседей
        region_neighbors = {}
        for reg in region_data:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(reg['mask'].astype(np.uint8), kernel)
            boundary = dilated - reg['mask'].astype(np.uint8)
            
            if np.sum(boundary) > 0:
                neighbor_colors = new_labels[boundary > 0]
                unique_nb, counts = np.unique(neighbor_colors, return_counts=True)
                region_neighbors[reg['comp_id']] = list(zip(unique_nb, counts))
            else:
                region_neighbors[reg['comp_id']] = []
        
        # Поиск лучшей пары для слияния
        best_score = float('inf')
        best_pair = None
        
        # LAB цвета для расчёта расстояния
        palette_lab = cv2.cvtColor(
            np.uint8([[c for c in palette]]), cv2.COLOR_RGB2LAB
        ).astype(np.float32).squeeze()
        
        for reg in region_data:
            if reg['area'] > np.median([r['area'] for r in region_data]) * 3:
                continue  # пропускаем крупные регионы
            
            current_lab = palette_lab[reg['color_idx']]
            
            for nb_color, count in region_neighbors.get(reg['comp_id'], []):
                if nb_color == reg['color_idx']:
                    continue
                nb_lab = palette_lab[nb_color]
                color_dist = np.linalg.norm(current_lab - nb_lab)
                
                # Эвристика: маленькая площадь × расстояние / частота соседа
                score = reg['area'] * color_dist / (count + 1)
                
                if score < best_score:
                    best_score = score
                    best_pair = (reg, nb_color)
        
        if best_pair is None:
            break  # нечего сливать
        
        # Выполняем слияние
        reg_to_merge, target_color = best_pair
        new_labels[reg_to_merge['mask']] = target_color
        new_quantized[reg_to_merge['mask']] = palette[target_color]
    
    # Обновление палитры
    final_palette = np.unique(new_quantized.reshape(-1, 3), axis=0)
    final_palette = [tuple(c) for c in final_palette]
    
    # Ремаппинг label_map под новую палитру
    color_to_idx = {color: idx for idx, color in enumerate(final_palette)}
    new_label_map = np.zeros_like(new_labels)
    for color_idx, color in enumerate(palette):
        if color in color_to_idx:
            mask = (new_labels == color_idx)
            new_label_map[mask] = color_to_idx[color]
    
    logger.info(f"🔗 Слияние: {len(final_palette)} цветов, ~{len(np.unique(new_label_map))} регионов")
    
    return new_quantized, new_label_map, final_palette


# ============================================
# ✏️ ЗАМЕНА: create_coloring_page_raster()
# ============================================

def create_coloring_page_raster(quantized: np.ndarray, palette: List[Tuple[int,int,int]], 
                               config) -> io.BytesIO:
    """
    Генерация PNG раскраски:
    • Единая карта границ (без дублей)
    • Pole of inaccessibility для номеров
    • Проверка наложений
    """
    h, w = quantized.shape[:2]
    line_rgb = config.line_rgb
    font_size = config.font_size
    
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    font = safe_get_font(font_size)
    
    def safe_text_size(text: str, font_obj) -> Tuple[int, int]:
        try:
            if hasattr(font_obj, 'getbbox'):
                bbox = font_obj.getbbox(text)
                return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
            elif hasattr(font_obj, 'getsize'):
                return font_obj.getsize(text)
        except:
            pass
        return max(1, font_size * len(text) // 2), max(1, font_size)
    
    # 🔹 STEP 1: Label map
    label_map = np.zeros((h, w), dtype=np.int32)
    for color_idx, color in enumerate(palette):
        mask = np.all(quantized == color, axis=2)
        label_map[mask] = color_idx + 1  # 0 = background
    
    # 🔹 STEP 2: Единая карта границ
    grad_x = np.abs(np.diff(label_map, axis=1, prepend=label_map[:, :1]))
    grad_y = np.abs(np.diff(label_map, axis=0, prepend=label_map[:1, :]))
    boundary_mask = ((grad_x > 0) | (grad_y > 0)).astype(np.uint8)
    
    # Утолщение
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=config.line_thickness)
    
    # 🔹 STEP 3: Контуры
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_canvas, contours, -1, 255, thickness=1)
    canvas[contour_canvas > 0] = line_rgb
    
    # 🔹 STEP 4: Размещение номеров
    placed_positions = []
    regions_with_numbers = 0
    
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
        
        # Очистка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Connected components
        num, labels_img, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=4, ltype=cv2.CV_32S)
        
        for comp_id in range(1, num):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < config.min_region_size:
                continue
            
            comp_mask = (labels_img == comp_id).astype(np.uint8)
            
            # Pole of inaccessibility
            pole = get_pole_of_inaccessibility(comp_mask)
            if pole is None:
                continue
            cx, cy = pole
            
            # Проверка наложения
            collision = any(
                math.hypot(cx - px, cy - py) < font_size * 2.5 
                for px, py in placed_positions
            )
            if collision:
                continue
            
            num_str = str(color_idx + 1)
            text_w, text_h = safe_text_size(num_str, font)
            
            # Белый фон
            padding = 3
            x1 = max(0, cx - text_w//2 - padding)
            y1 = max(0, cy - text_h//2 - padding)
            x2 = min(w, cx + text_w//2 + padding)
            y2 = min(h, cy + text_h//2 + padding)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
            
            # Номер через PIL
            pil_img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_img)
            draw.text((cx - text_w//2, cy - text_h//2 + 1), num_str, 
                     fill='black', font=font)
            canvas = np.array(pil_img)
            
            placed_positions.append((cx, cy))
            regions_with_numbers += 1
    
    logger.info(f"✅ Номеров размещено: {regions_with_numbers} из {len(palette)} цветов")
    
    # Сохранение
    output = io.BytesIO()
    Image.fromarray(canvas).save(output, format='PNG', dpi=(300, 300))
    output.seek(0)
    return output


# ============================================
# 📐 ЗАМЕНА: generate_svg_output()
# ============================================

def generate_svg_output(quantized: np.ndarray, palette: List[Tuple[int,int,int]], 
                       config) -> str:
    """Генерация SVG с оптимизированными путями"""
    h, w = quantized.shape[:2]
    line_rgb = config.line_rgb
    font_size = config.font_size
    
    svg_parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<style>.region{{fill:none;stroke:rgb{line_rgb};stroke-width:{config.line_thickness};stroke-linejoin:round;stroke-linecap:round}}</style>',
    ]
    
    placed_positions = []
    
    # Единая карта границ для SVG
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
        area = cv2.contourArea(contour)
        if area < config.min_region_size * 0.5:
            continue
        
        # Упрощение контура для SVG
        epsilon = 0.5  # пиксели
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(simplified) < 3:
            continue
        
        # Построение path
        path_d = f"M {simplified[0][0][0]},{simplified[0][0][1]}"
        for point in simplified[1:]:
            x, y = point[0]
            path_d += f" L {x},{y}"
        path_d += " Z"
        
        svg_parts.append(f'<path class="region" d="{path_d}"/>')
    
    # Номера
    for color_idx, color in enumerate(palette):
        color_mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
        
        num, labels_img, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=4, ltype=cv2.CV_32S)
        
        for comp_id in range(1, num):
            if stats[comp_id, cv2.CC_STAT_AREA] < config.min_region_size:
                continue
            
            comp_mask = (labels_img == comp_id).astype(np.uint8)
            pole = get_pole_of_inaccessibility(comp_mask)
            if pole is None:
                continue
            cx, cy = pole
            
            # Проверка наложения
            if any(math.hypot(cx - px, cy - py) < font_size * 2.5 for px, py in placed_positions):
                continue
            
            num_str = str(color_idx + 1)
            
            # Фон и текст
            svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{font_size//2 + 3}" fill="white" stroke="none"/>')
            svg_parts.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
                           f'font-size="{font_size}" font-family="Arial, sans-serif" fill="black" '
                           f'stroke="white" stroke-width="0.5">{num_str}</text>')
            placed_positions.append((cx, cy))
    
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


# ============================================
# 🔧 ОБНОВЛЁННЫЙ: process_image_for_coloring()
# ============================================

def process_image_for_coloring(photo_bytes: bytes, config) -> Tuple[io.BytesIO, io.BytesIO, Optional[str]]:
    """
    Полный пайплайн с новым алгоритмом.
    Drop-in замена для вашей функции.
    """
    from PIL import Image
    
    # Загрузка и препроцессинг (ваш код)
    image = Image.open(io.BytesIO(photo_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    width, height = image.size
    if max(width, height) > config.max_image_size:
        ratio = config.max_image_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    # 🔹 Новый пайплайн
    logger.info("🎨 Запуск гибридной сегментации...")
    
    # 1. Сегментация (SLIC + Mean Shift)
    quantized, labels, palette = segment_image_advanced(img_array, config)
    
    # 2. Удаление тонких регионов
    logger.info("✂️  Удаление тонких областей...")
    quantized = remove_thin_regions_scan(quantized, min_length=7, iterations=3)
    
    # 3. Умное слияние регионов
    logger.info("🔗 Слияние регионов через RAG...")
    quantized, labels, palette = merge_regions_rag(
        quantized, labels, palette, 
        min_area=config.min_region_size,
        target_regions=config.n_colors * 4
    )
    
    # 4. Генерация выходов
    logger.info("✏️  Генерация раскраски...")
    coloring_buffer = create_coloring_page_raster(quantized, palette, config)
    
    palette_buffer = io.BytesIO()
    from your_bot_file import create_palette_image  # импортируем вашу функцию
    palette_img = create_palette_image(palette, config)
    palette_img.save(palette_buffer, format='PNG', dpi=(300, 300))
    palette_buffer.seek(0)
    
    # SVG (опционально)
    svg_output = None
    if config.export_svg:
        try:
            svg_output = generate_svg_output(quantized, palette, config)
        except Exception as e:
            logger.warning(f"SVG generation failed: {e}")
    
    logger.info(f"✅ Готово! Палитра: {len(palette)} цветов")
    
    return coloring_buffer, palette_buffer, svg_output
