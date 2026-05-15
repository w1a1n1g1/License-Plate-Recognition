import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSlider, QSpinBox,
                             QGroupBox, QGridLayout, QTabWidget, QTextEdit,
                             QProgressBar, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    if os.name == 'nt':
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe'
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
except ImportError:
    TESSERACT_AVAILABLE = False


class LicensePlateDetector:
    def __init__(self):
        self.params = {
            'gaussian_kernel': (5, 5),
            'canny_low': 50,
            'canny_high': 150,
            'morph_kernel_close': (11, 5),
            'morph_kernel_open': (3, 3),
            'aspect_ratio_min': 3.0,
            'aspect_ratio_max': 6.0,
            'area_min': 3000,
            'area_max': 80000,
            'plate_width': 140,
            'plate_height': 35,
            'char_width': 20,
            'char_height': 30
        }
        self.debug_images = {}

    def set_params(self, params):
        self.params.update(params)

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        self.img_original = img
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess_image(self):
        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        self.debug_images['gray'] = gray

        # 添加CLAHE直方图均衡化以提高对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        self.debug_images['clahe'] = gray

        blurred = cv2.GaussianBlur(gray, self.params['gaussian_kernel'], 0)
        self.debug_images['blurred'] = blurred

        # 使用自适应阈值代替Otsu，以更好地处理不同光照
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        self.debug_images['binary'] = binary

        self.gray, self.blurred, self.binary = gray, blurred, binary
        return gray, blurred, binary

    def edge_detection(self):
        edges = cv2.Canny(self.blurred, self.params['canny_low'], self.params['canny_high'])
        self.debug_images['edges'] = edges
        self.edges = edges
        return edges

    def morphological_processing(self):
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, self.params['morph_kernel_close'])
        closed = cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, self.params['morph_kernel_open'])
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

        self.debug_images['morphological'] = opened
        self.morph_result = opened
        return opened

    def find_license_plate(self):
        contours, _ = cv2.findContours(self.morph_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / h

            if not (self.params['aspect_ratio_min'] < aspect_ratio < self.params['aspect_ratio_max']):
                continue
            if not (self.params['area_min'] < area < self.params['area_max']):
                continue

            rect_area = w * h
            rect_score = area / rect_area

            if rect_score < 0.5:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.6:
                continue

            extent = area / rect_area
            if extent < 0.4:
                continue

            if h < 20 or w < 60:
                continue

            img_height = self.img_original.shape[0]
            if y < img_height * 0.1 or y > img_height * 0.9:
                continue

            candidates.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'aspect_ratio': aspect_ratio,
                'area': area,
                'score': rect_score,
                'solidity': solidity,
                'extent': extent,
                'contour': cnt
            })

        candidates.sort(key=lambda x: (x['aspect_ratio'] - 4.5)**2 + (1 - x['score']), reverse=False)
        self.candidates = candidates[:3]
        return self.candidates

    def find_license_plate_enhanced(self):
        """增强版车牌检测，添加纹理特征"""
        contours, _ = cv2.findContours(self.morph_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / h

            if not (self.params['aspect_ratio_min'] < aspect_ratio < self.params['aspect_ratio_max']):
                continue
            if not (self.params['area_min'] < area < self.params['area_max']):
                continue

            rect_area = w * h
            rect_score = area / rect_area
            if rect_score < 0.5:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.6:
                continue

            extent = area / rect_area
            if extent < 0.4:
                continue

            if h < 20 or w < 60:
                continue

            img_height = self.img_original.shape[0]
            if y < img_height * 0.1 or y > img_height * 0.9:
                continue

            # 添加纹理特征：计算车牌区域的梯度方差
            plate_roi = self.gray[y:y+h, x:x+w]
            if plate_roi.size > 0:
                sobelx = cv2.Sobel(plate_roi, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(plate_roi, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                texture_score = np.var(gradient_magnitude)
            else:
                texture_score = 0

            # 车牌通常有较高的纹理复杂度
            if texture_score < 100:
                continue

            candidates.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'aspect_ratio': aspect_ratio,
                'area': area,
                'score': rect_score,
                'solidity': solidity,
                'extent': extent,
                'texture_score': texture_score,
                'contour': cnt
            })

        # 改进排序：结合多个特征
        candidates.sort(key=lambda x: (
            x['texture_score'],  # 纹理分数优先
            (x['aspect_ratio'] - 4.5)**2,  # 接近标准宽高比
            1 - x['score'],  # 矩形度
            1 - x['solidity']  # 实心度
        ), reverse=True)
        self.candidates = candidates[:5]  # 保留更多候选
        return self.candidates

    def extract_plate_region(self):
        if not self.candidates:
            return None

        best = self.candidates[0]
        x, y, w, h = best['x'], best['y'], best['w'], best['h']

        padding = 3
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(self.img_original.shape[1] - x, w + padding * 2)
        h = min(self.img_original.shape[0] - y, h + padding * 2)

        plate_region = cv2.cvtColor(self.img_original[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        plate_region = cv2.resize(plate_region, (self.params['plate_width'], self.params['plate_height']))

        self.debug_images['plate_region'] = plate_region
        self.plate_region = plate_region
        return plate_region

    def segment_characters(self):
        if not hasattr(self, 'plate_region') or self.plate_region is None:
            return [], None, []

        plate = self.plate_region

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        plate = cv2.morphologyEx(plate, cv2.MORPH_OPEN, kernel, iterations=1)

        # 使用自适应阈值进行二值化
        binary = cv2.adaptiveThreshold(
            plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.erode(binary, kernel, iterations=1)

        # 使用连通组件分析进行字符分割
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        char_candidates = []
        for i in range(1, num_labels):  # 跳过背景(标签0)
            x, y, w, h, area = stats[i]
            if w > 5 and h > 10 and area > 50:  # 过滤太小的组件
                char_candidates.append((x, y, w, h, area))

        # 按x坐标排序字符
        char_candidates.sort(key=lambda x: x[0])

        # 合并真正重叠的小组件，但避免将正常字符连成一个
        merged_chars = []
        if char_candidates:
            current_group = char_candidates[0]
            for candidate in char_candidates[1:]:
                x, y, w, h, area = candidate
                prev_x, prev_y, prev_w, prev_h, prev_area = current_group

                # 仅在完全重叠或高度重叠时合并
                overlap = max(0, min(prev_x + prev_w, x + w) - max(prev_x, x))
                if overlap > min(prev_w, w) * 0.5 or x < prev_x + prev_w - 3:
                    min_x = min(prev_x, x)
                    min_y = min(prev_y, y)
                    max_x = max(prev_x + prev_w, x + w)
                    max_y = max(prev_y + prev_h, y + h)
                    current_group = (min_x, min_y, max_x - min_x, max_y - min_y, prev_area + area)
                else:
                    merged_chars.append(current_group)
                    current_group = candidate

            if current_group:
                merged_chars.append(current_group)

        # 如果连通组件数不足，则退回固定宽度分割
        if len(merged_chars) < 5:
            merged_chars = []
            for start, end in self.fixed_width_segment(binary):
                merged_chars.append((start, 0, end - start, binary.shape[0], 0))

        # 提取字符图像
        character_images = []
        char_texts = []
        for idx, item in enumerate(merged_chars[:7]):
            x, y, w, h, area = item
            char_img = binary[y:y+h, x:x+w]
            char_img = self.normalize_char(char_img)
            character_images.append(char_img)
            char_char, char_conf = self.recognize_char(char_img, idx)
            char_texts.append((char_char, float(char_conf)))

        self.debug_images['plate_binary'] = binary
        return character_images, binary, char_texts

    def fixed_width_segment(self, binary):
        chars = []
        total_width = binary.shape[1]
        char_width = total_width // 7
        remainder = total_width % 7

        start = 0
        for i in range(7):
            end = start + char_width + (1 if i < remainder else 0)
            if end <= total_width:
                chars.append((start, end))
                start = end
        return chars

    def normalize_char(self, char_img):
        # 保证输入为灰度二值图
        if len(char_img.shape) != 2:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

        # 先去噪与平滑小碎片
        char_img = cv2.medianBlur(char_img, 3)

        # 二值化（如果尚未二值）
        _, th = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 确保与车牌二值方向一致（车牌字符为白/黑取决于前处理），我们期望字符为白(255)
        white_ratio = np.count_nonzero(th) / th.size
        if white_ratio < 0.5:
            th = 255 - th

        # 找最大的轮廓作为字符主体
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = int(np.argmax(areas))
            x, y, w, h = cv2.boundingRect(contours[max_idx])
            th = th[y:y+h, x:x+w]

        # 填充为正方形并缩放
        rows, cols = th.shape
        max_dim = max(rows, cols, 1)
        pad_top = (max_dim - rows) // 2
        pad_bottom = max_dim - rows - pad_top
        pad_left = (max_dim - cols) // 2
        pad_right = max_dim - cols - pad_left
        th = cv2.copyMakeBorder(th, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        # 缩放到固定尺寸并做轻微形态学闭运算
        th = cv2.resize(th, (self.params['char_width'], self.params['char_height']), interpolation=cv2.INTER_AREA)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        return th

    def recognize_char(self, char_img, position=0):
        rows, cols = char_img.shape

        # 扩边以避免裁切影响
        char_proc = cv2.copyMakeBorder(char_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        rows, cols = char_proc.shape

        black_pixels = np.count_nonzero(char_proc)
        total_pixels = rows * cols
        density = black_pixels / total_pixels if total_pixels > 0 else 0

        # 过滤空白或过饱和
        if density < 0.03:
            return ('#', 0.0)
        if density > 0.95:
            return ('#', 0.0)

        # 先尝试快速规则识别
        if position == 0:
            rule_res = self.recognize_chinese(char_proc, density)
        elif position == 1:
            rule_res = self.recognize_letter(char_proc, density)
        elif position == 2:
            rule_res = self.recognize_separator(char_proc)
        else:
            rule_res = self.recognize_alphanumeric(char_proc, density)

        # 规则置信度基于像素密度
        density_clamped = max(0.0, min(1.0, (density - 0.03) / 0.92))
        rule_conf = 0.2 + 0.7 * density_clamped

        # 如果可用，使用单字符Tesseract作为回退/验证
        ocr_res = None
        ocr_conf = 0.0
        if TESSERACT_AVAILABLE:
            try:
                # 放大以提高OCR质量
                scale = max(1, 64 // max(rows, cols))
                ocr_img = cv2.resize(char_proc, (cols*scale, rows*scale), interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                ocr_img = clahe.apply(ocr_img)
                _, ocr_img = cv2.threshold(ocr_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                if position == 0:
                    # 中文位置，尝试中文识别
                    config = '--oem 3 --psm 10'
                    txt = pytesseract.image_to_string(ocr_img, config=config, lang='chi_sim')
                    ocr_res = txt.strip()
                else:
                    # 字母/数字位置，限制字符集
                    whitelist = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'
                    config = f"--oem 3 --psm 10 -c tessedit_char_whitelist={whitelist}"
                    txt = pytesseract.image_to_string(ocr_img, config=config, lang='eng')
                    txt = txt.strip().upper()
                    ocr_res = txt[0] if txt else None

                # 粗略估计OCR置信度：字符长度和像素占比作为代理
                if ocr_res:
                    ocr_conf = min(0.98, 0.4 + 0.6 * density_clamped)
            except Exception:
                ocr_res = None
                ocr_conf = 0.0

        # 优先返回可信的OCR结果
        if ocr_res and len(ocr_res) >= 1:
            # 取第一个字符作为识别结果
            return (ocr_res[0], float(ocr_conf))

        # 否则返回规则结果与置信度
        return (rule_res, float(rule_conf))

    def recognize_chinese(self, char_img, density):
        rows, cols = char_img.shape
        
        contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return '?'
        
        x, y, w, h = cv2.boundingRect(contours[0])
        aspect_ratio = w / h
        
        left_part = char_img[:, :cols//3]
        mid_part = char_img[:, cols//3:2*cols//3]
        right_part = char_img[:, 2*cols//3:]
        
        left_density = np.count_nonzero(left_part) / left_part.size if left_part.size > 0 else 0
        mid_density = np.count_nonzero(mid_part) / mid_part.size if mid_part.size > 0 else 0
        right_density = np.count_nonzero(right_part) / right_part.size if right_part.size > 0 else 0
        
        top_part = char_img[:rows//3, :]
        bottom_part = char_img[2*rows//3:, :]
        top_density = np.count_nonzero(top_part) / top_part.size if top_part.size > 0 else 0
        bottom_density = np.count_nonzero(bottom_part) / bottom_part.size if bottom_part.size > 0 else 0
        
        h_proj = np.sum(char_img, axis=0)
        v_proj = np.sum(char_img, axis=1)
        h_empty = np.sum(h_proj == 0)
        v_empty = np.sum(v_proj == 0)
        h_ratio = h_empty / cols
        v_ratio = v_empty / rows
        
        if density > 0.4 and density < 0.6:
            if mid_density > left_density and mid_density > right_density:
                if h_ratio > 0.25:
                    return '沪'
                else:
                    return '京'
            elif left_density > 0.3 and right_density > 0.3:
                if bottom_density > top_density:
                    return '粤'
                else:
                    return '浙'
        
        if density > 0.35 and density < 0.55:
            if left_density > right_density * 1.5:
                return '浙'
            elif right_density > left_density * 1.5:
                return '苏'
        
        if aspect_ratio > 0.6 and aspect_ratio < 1.1:
            if density > 0.42:
                return '沪'
            else:
                return '京'
        
        if h_ratio > 0.22:
            return '沪'
        
        if v_ratio > 0.3:
            return '京'
        
        return '沪'

    def recognize_letter(self, char_img, density):
        rows, cols = char_img.shape
        
        contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        
        char_img_inv = 255 - char_img
        contours2, _ = cv2.findContours(char_img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_hole = any(cv2.contourArea(c) > 8 for c in contours2)
        
        mid_region = char_img[3:rows-3, 3:cols-3]
        mid_density = np.count_nonzero(mid_region) / mid_region.size if mid_region.size > 0 else 0
        
        top_half = char_img[:rows//2, :]
        bottom_half = char_img[rows//2:, :]
        top_density = np.count_nonzero(top_half) / top_half.size
        bottom_density = np.count_nonzero(bottom_half) / bottom_half.size
        
        left_region = char_img[:, :cols//3]
        right_region = char_img[:, 2*cols//3:]
        left_density = np.count_nonzero(left_region) / left_region.size
        right_density = np.count_nonzero(right_region) / right_region.size
        
        if has_hole:
            if mid_density > 0.25:
                return 'O'
            elif bottom_density > top_density * 1.3:
                return 'Q'
            else:
                return 'O'
        
        if left_density > 0.35 and right_density < 0.15:
            return 'L'
        
        if num_contours >= 2:
            if bottom_density > top_density * 1.2:
                return 'R'
            else:
                return 'K'
        
        if mid_density < 0.08:
            if left_density > right_density:
                return 'L'
            else:
                return 'P'
        
        if density > 0.5:
            if bottom_density > top_density:
                if bottom_density > mid_density:
                    return 'Z'
                else:
                    return 'R'
            else:
                return 'X'
        else:
            if bottom_density > top_density * 1.1:
                return 'R'
            elif top_density > bottom_density * 1.1:
                return 'P'
        
        return 'K'

    def recognize_separator(self, char_img):
        rows, cols = char_img.shape
        density = np.count_nonzero(char_img) / (rows * cols)
        
        if density < 0.2:
            return '·'
        return '·'

    def recognize_alphanumeric(self, char_img, density):
        rows, cols = char_img.shape
        
        contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        
        char_img_inv = 255 - char_img
        contours2, _ = cv2.findContours(char_img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_hole = any(cv2.contourArea(c) > 8 for c in contours2)
        
        top_half = char_img[:rows//2, :]
        bottom_half = char_img[rows//2:, :]
        top_density = np.count_nonzero(top_half) / top_half.size
        bottom_density = np.count_nonzero(bottom_half) / bottom_half.size
        
        left_region = char_img[:, :cols//3]
        mid_region = char_img[:, cols//3:2*cols//3]
        right_region = char_img[:, 2*cols//3:]
        left_density = np.count_nonzero(left_region) / left_region.size
        mid_density = np.count_nonzero(mid_region) / mid_region.size
        right_density = np.count_nonzero(right_region) / right_region.size
        
        if has_hole:
            if bottom_density > top_density * 1.2:
                return '6'
            elif top_density > bottom_density * 1.2:
                return '8'
            elif mid_density > 0.25:
                return '0'
            else:
                return '0'
        
        if left_density > 0.35 and right_density < 0.12:
            return '1'
        
        if num_contours >= 2:
            return '4'
        
        if mid_density < 0.06:
            if left_density > right_density:
                return '1'
            else:
                return '7'
        
        if density > 0.45:
            if bottom_density > top_density:
                if bottom_density > mid_density:
                    return '2'
                else:
                    return '3'
            else:
                return '5'
        else:
            if bottom_density > top_density * 1.1:
                return '6'
            elif top_density > bottom_density * 1.1:
                if mid_density < 0.08:
                    return '7'
                else:
                    return '9'
            elif mid_density > 0.3:
                return '8'
        
        return '8'

    def recognize_plate_tesseract(self, plate_region):
        if not TESSERACT_AVAILABLE or plate_region is None:
            return None
        
        try:
            # 预处理车牌图像以提高OCR准确性
            # 调整大小
            height, width = plate_region.shape
            if width < 200:
                scale_factor = 200 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                plate_region = cv2.resize(plate_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 增强对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            plate_region = clahe.apply(plate_region)
            
            # 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            plate_region = cv2.filter2D(plate_region, -1, kernel)
            
            # 使用改进的Tesseract配置
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHJKLMNPQRSTUVWXYZ0123456789沪京粤浙苏川鲁皖豫云辽黑湘皖鲁新苏浙赣桂甘晋蒙陕吉闽贵粤青藏川宁琼 -c tessedit_pageseg_mode=7'
            text = pytesseract.image_to_string(plate_region, config=custom_config, lang='chi_sim+eng')
            text = text.strip().replace(' ', '').replace('\n', '').replace('·', '')
            
            # 后处理：验证车牌格式
            if self.validate_plate_format(text):
                return text
            else:
                # 如果格式不正确，尝试其他PSM模式
                for psm in [8, 6]:
                    config_alt = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHJKLMNPQRSTUVWXYZ0123456789沪京粤浙苏川鲁皖豫云辽黑湘皖鲁新苏浙赣桂甘晋蒙陕吉闽贵粤青藏川宁琼'
                    text_alt = pytesseract.image_to_string(plate_region, config=config_alt, lang='chi_sim+eng')
                    text_alt = text_alt.strip().replace(' ', '').replace('\n', '').replace('·', '')
                    if self.validate_plate_format(text_alt):
                        return text_alt
                
            return text if len(text) >= 6 else None
        except Exception as e:
            return None
    
    def validate_plate_format(self, text):
        """验证车牌号格式"""
        if not text or len(text) < 7:
            return False
        
        # 中国车牌格式：省份简称 + 字母/数字 + 5位字母数字
        provinces = '沪京粤浙苏川鲁皖豫云辽黑湘皖鲁新苏浙赣桂甘晋蒙陕吉闽贵粤青藏川宁琼'
        
        if text[0] not in provinces:
            return False
        
        # 检查其余字符是否为字母或数字
        for char in text[1:]:
            if not (char.isalnum() or char in 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'):
                return False
        
        return True

    def recognize_plate(self, char_texts, plate_region=None):
        tesseract_result = self.recognize_plate_tesseract(plate_region)

        # 处理 char_texts 现在为 (char, conf) 列表
        rule_text = ''
        rule_confidence = 0.0
        if char_texts:
            chars = [c for c, _ in char_texts]
            confs = [conf for _, conf in char_texts]
            rule_text = ''.join(chars)
            # 平均置信度
            rule_confidence = sum(confs) / len(confs) if confs else 0.0

        # 如果Tesseract结果有效且格式通过验证，优先使用
        if tesseract_result and self.validate_plate_format(tesseract_result):
            # 若规则识别存在，则比较一致性
            if rule_text and len(tesseract_result) == len(rule_text):
                match_score = sum(1 for a, b in zip(tesseract_result, rule_text) if a == b) / len(tesseract_result)
                if match_score > 0.6:
                    return f"{tesseract_result} (高置信度)"
                else:
                    # 若规则置信度更高则采用规则结果
                    if rule_confidence > 0.75:
                        return f"{rule_text} (规则识别)"
                    else:
                        return f"{tesseract_result} (OCR识别)"
            else:
                return f"{tesseract_result} (OCR识别)"

        # 若无OCR结果，则使用规则识别并在字符串上标注置信度
        if not rule_text:
            return "未检测到字符"

        if rule_confidence < 0.4:
            return f"{rule_text} (置信度低)"
        elif rule_confidence < 0.75:
            return f"{rule_text} (中等置信度)"
        else:
            return f"{rule_text} (高置信度)"

    def draw_results(self):
        result = self.img_rgb.copy()
        for idx, candidate in enumerate(self.candidates):
            x, y, w, h = candidate['x'], candidate['y'], candidate['w'], candidate['h']
            color = (0, 255, 0) if idx == 0 else (255, 0, 0)
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result, f"候选{idx+1}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return result

    def run(self, image_path):
        self.load_image(image_path)
        
        # 多尺度检测
        best_result = None
        best_plate = None
        best_chars = []
        best_plate_binary = None
        best_plate_number = "未检测到车牌"
        best_score = 0
        
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # 不同缩放比例
        
        for scale in scales:
            try:
                # 创建缩放版本的图像
                if scale != 1.0:
                    scaled_img = cv2.resize(self.img_original, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    self.img_original_scaled = scaled_img
                    self.img_rgb_scaled = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
                    
                    # 临时替换图像进行处理
                    original_img = self.img_original
                    original_rgb = self.img_rgb
                    self.img_original = scaled_img
                    self.img_rgb = self.img_rgb_scaled
                
                self.preprocess_image()
                self.edge_detection()
                self.morphological_processing()
                candidates = self.find_license_plate_enhanced()
                
                if candidates:
                    plate = self.extract_plate_region()
                    chars = []
                    plate_binary = None
                    char_texts = []
                    
                    if plate is not None:
                        chars, plate_binary, char_texts = self.segment_characters()
                    
                    plate_number = self.recognize_plate(char_texts, plate)
                    
                    # 计算当前尺度的评分
                    score = len(candidates) * 0.3 + (len(chars) if chars else 0) * 0.7
                    
                    # 如果这个尺度更好，保存结果
                    if score > best_score:
                        best_score = score
                        best_result = self.draw_results()
                        best_plate = plate
                        best_chars = chars
                        best_plate_binary = plate_binary
                        best_plate_number = plate_number
                
                # 恢复原始图像
                if scale != 1.0:
                    self.img_original = original_img
                    self.img_rgb = original_rgb
                    
            except Exception as e:
                continue  # 跳过失败的尺度
        
        # 如果没有找到，使用原始尺度
        if best_score == 0:
            self.preprocess_image()
            self.edge_detection()
            self.morphological_processing()
            self.find_license_plate_enhanced()
            best_plate = self.extract_plate_region()
            if best_plate is not None:
                best_chars, best_plate_binary, char_texts = self.segment_characters()
                best_plate_number = self.recognize_plate(char_texts, best_plate)
            best_result = self.draw_results()
        
        return best_result, best_plate, best_chars, best_plate_binary, best_plate_number


class DetectionThread(QThread):
    finished = pyqtSignal(object, object, object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, detector, image_path):
        super().__init__()
        self.detector = detector
        self.image_path = image_path

    def run(self):
        try:
            result, plate, chars, plate_binary, plate_number = self.detector.run(self.image_path)
            self.finished.emit(result, plate, chars, plate_binary, plate_number, self.detector)
        except Exception as e:
            self.error.emit(str(e))


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #ccc;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if self.pixmap():
            self.setPixmap(self.pixmap())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车牌检测系统")
        self.setGeometry(100, 100, 1200, 800)

        self.detector = LicensePlateDetector()
        self.current_image_path = None
        self.thread = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(280)

        self.create_control_panel(left_layout)
        self.create_param_panel(left_layout)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.tab_widget = QTabWidget()
        self.create_result_tab()
        self.create_debug_tab()
        right_layout.addWidget(self.tab_widget)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def create_control_panel(self, layout):
        group_box = QGroupBox("控制面板")
        box_layout = QVBoxLayout(group_box)

        self.btn_open = QPushButton("打开图像")
        self.btn_open.clicked.connect(self.open_image)
        box_layout.addWidget(self.btn_open)

        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_detect.setEnabled(False)
        box_layout.addWidget(self.btn_detect)

        self.btn_batch = QPushButton("批量处理")
        self.btn_batch.clicked.connect(self.batch_process)
        box_layout.addWidget(self.btn_batch)

        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)
        box_layout.addWidget(self.btn_save)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        box_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        box_layout.addWidget(self.status_label)

        layout.addWidget(group_box)

    def create_param_panel(self, layout):
        group_box = QGroupBox("参数设置")
        box_layout = QVBoxLayout(group_box)

        params_grid = QGridLayout()

        params_grid.addWidget(QLabel("Canny低阈值:"), 0, 0)
        self.canny_low_slider = QSlider(Qt.Horizontal)
        self.canny_low_slider.setRange(10, 200)
        self.canny_low_slider.setValue(50)
        self.canny_low_slider.valueChanged.connect(self.update_params)
        params_grid.addWidget(self.canny_low_slider, 0, 1)
        self.canny_low_label = QLabel("50")
        params_grid.addWidget(self.canny_low_label, 0, 2)

        params_grid.addWidget(QLabel("Canny高阈值:"), 1, 0)
        self.canny_high_slider = QSlider(Qt.Horizontal)
        self.canny_high_slider.setRange(50, 300)
        self.canny_high_slider.setValue(150)
        self.canny_high_slider.valueChanged.connect(self.update_params)
        params_grid.addWidget(self.canny_high_slider, 1, 1)
        self.canny_high_label = QLabel("150")
        params_grid.addWidget(self.canny_high_label, 1, 2)

        params_grid.addWidget(QLabel("面积最小值:"), 2, 0)
        self.area_min_spin = QSpinBox()
        self.area_min_spin.setRange(1000, 20000)
        self.area_min_spin.setValue(3000)
        self.area_min_spin.valueChanged.connect(self.update_params)
        params_grid.addWidget(self.area_min_spin, 2, 1)

        params_grid.addWidget(QLabel("面积最大值:"), 3, 0)
        self.area_max_spin = QSpinBox()
        self.area_max_spin.setRange(10000, 200000)
        self.area_max_spin.setValue(80000)
        self.area_max_spin.valueChanged.connect(self.update_params)
        params_grid.addWidget(self.area_max_spin, 3, 1)

        params_grid.addWidget(QLabel("宽高比最小:"), 4, 0)
        self.ar_min_spin = QSpinBox()
        self.ar_min_spin.setRange(10, 50)
        self.ar_min_spin.setValue(30)
        self.ar_min_spin.valueChanged.connect(self.update_params)
        params_grid.addWidget(self.ar_min_spin, 4, 1)

        params_grid.addWidget(QLabel("宽高比最大:"), 5, 0)
        self.ar_max_spin = QSpinBox()
        self.ar_max_spin.setRange(40, 100)
        self.ar_max_spin.setValue(60)
        self.ar_max_spin.valueChanged.connect(self.update_params)
        params_grid.addWidget(self.ar_max_spin, 5, 1)

        box_layout.addLayout(params_grid)
        layout.addWidget(group_box)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.log_text)

    def create_result_tab(self):
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)

        self.original_label = ImageLabel()
        self.original_label.setFixedSize(320, 240)
        self.original_label.setText("原始图像")
        top_layout.addWidget(self.original_label)

        self.result_label = ImageLabel()
        self.result_label.setFixedSize(320, 240)
        self.result_label.setText("检测结果")
        top_layout.addWidget(self.result_label)

        self.plate_label = ImageLabel()
        self.plate_label.setFixedSize(200, 60)
        self.plate_label.setText("提取的车牌")
        top_layout.addWidget(self.plate_label)

        result_layout.addWidget(top_row)

        self.chars_layout = QHBoxLayout()
        self.char_labels = []
        for i in range(7):
            label = ImageLabel()
            label.setFixedSize(40, 50)
            label.setText(f"字符{i+1}")
            self.char_labels.append(label)
            self.chars_layout.addWidget(label)
        result_layout.addLayout(self.chars_layout)

        plate_number_layout = QHBoxLayout()
        plate_number_layout.addWidget(QLabel("车牌号:"))
        self.plate_number_label = QLabel("")
        self.plate_number_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3;")
        self.plate_number_label.setAlignment(Qt.AlignLeft)
        plate_number_layout.addWidget(self.plate_number_label)
        plate_number_layout.addStretch()
        result_layout.addLayout(plate_number_layout)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        result_layout.addWidget(self.info_text)

        self.tab_widget.addTab(result_tab, "检测结果")

    def create_debug_tab(self):
        debug_tab = QScrollArea()
        debug_tab.setWidgetResizable(True)

        debug_widget = QWidget()
        debug_layout = QGridLayout(debug_widget)

        self.debug_labels = {}
        debug_items = [
            ('gray', '灰度图像'),
            ('clahe', '直方图均衡化'),
            ('blurred', '高斯滤波'),
            ('binary', '二值化'),
            ('edges', '边缘检测'),
            ('morphological', '形态学处理'),
            ('plate_binary', '车牌二值化')
        ]

        for i, (key, title) in enumerate(debug_items):
            label = ImageLabel()
            label.setFixedSize(250, 180)
            label.setText(title)
            self.debug_labels[key] = label
            row = i // 3
            col = i % 3
            debug_layout.addWidget(label, row, col)

        debug_tab.setWidget(debug_widget)
        self.tab_widget.addTab(debug_tab, "调试图像")

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )

        if file_path:
            self.current_image_path = file_path
            self.load_image_to_label(file_path, self.original_label)
            self.btn_detect.setEnabled(True)
            self.status_label.setText(f"状态: 已加载 {os.path.basename(file_path)}")
            self.clear_results()

    def load_image_to_label(self, file_path, label):
        img = cv2.imread(file_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(q_image))

    def clear_results(self):
        self.result_label.clear()
        self.result_label.setText("检测结果")
        self.plate_label.clear()
        self.plate_label.setText("提取的车牌")
        for label in self.char_labels:
            label.clear()
            label.setText(f"字符{self.char_labels.index(label)+1}")
        self.plate_number_label.setText("")
        self.info_text.clear()
        for debug_label in self.debug_labels.values():
            debug_label.clear()

    def update_params(self):
        params = {
            'canny_low': self.canny_low_slider.value(),
            'canny_high': self.canny_high_slider.value(),
            'area_min': self.area_min_spin.value(),
            'area_max': self.area_max_spin.value(),
            'aspect_ratio_min': self.ar_min_spin.value() / 10.0,
            'aspect_ratio_max': self.ar_max_spin.value() / 10.0
        }
        self.detector.set_params(params)

        self.canny_low_label.setText(str(params['canny_low']))
        self.canny_high_label.setText(str(params['canny_high']))

    def start_detection(self):
        if not self.current_image_path:
            return

        self.btn_detect.setEnabled(False)
        self.status_label.setText("状态: 检测中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.thread = DetectionThread(self.detector, self.current_image_path)
        self.thread.finished.connect(self.on_detection_finished)
        self.thread.error.connect(self.on_detection_error)
        self.thread.start()

    def on_detection_finished(self, result, plate, chars, plate_binary, plate_number, detector):
        self.progress_bar.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.status_label.setText("状态: 检测完成")

        h, w, ch = result.shape
        bytes_per_line = ch * w
        q_image = QImage(result.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.result_label.setPixmap(QPixmap.fromImage(q_image))

        self.plate_number_label.setText(plate_number)

        if plate is not None:
            plate_rgb = cv2.cvtColor(plate, cv2.COLOR_GRAY2RGB)
            h, w, ch = plate_rgb.shape
            bytes_per_line = ch * w
            q_plate = QImage(plate_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.plate_label.setPixmap(QPixmap.fromImage(q_plate))

        for i, char_img in enumerate(chars):
            if i < len(self.char_labels):
                char_rgb = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)
                h, w, ch = char_rgb.shape
                bytes_per_line = ch * w
                q_char = QImage(char_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.char_labels[i].setPixmap(QPixmap.fromImage(q_char))

        info_text = f"检测到候选区域: {len(detector.candidates)} 个\n\n"
        for i, c in enumerate(detector.candidates):
            info_text += f"候选{i+1}:\n"
            info_text += f"  位置: ({c['x']}, {c['y']})\n"
            info_text += f"  尺寸: {c['w']} x {c['h']}\n"
            info_text += f"  宽高比: {c['aspect_ratio']:.2f}\n"
            info_text += f"  面积: {c['area']}\n"
            info_text += f"  矩形度: {c['score']:.2f}\n\n"
        info_text += f"提取到字符: {len(chars)} 个"
        self.info_text.setText(info_text)

        self.log_text.append(f"检测完成: {os.path.basename(self.current_image_path)}")

        for key, label in self.debug_labels.items():
            if key in detector.debug_images:
                img = detector.debug_images[key]
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                label.setPixmap(QPixmap.fromImage(q_img))

        self.thread = None

    def on_detection_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.status_label.setText("状态: 检测失败")
        QMessageBox.critical(self, "错误", f"检测失败: {error_msg}")
        self.log_text.append(f"错误: {error_msg}")
        self.thread = None

    def save_results(self):
        if not hasattr(self.detector, 'candidates'):
            QMessageBox.warning(self, "警告", "没有检测结果可保存")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not output_dir:
            return

        try:
            result = self.detector.draw_results()
            cv2.imwrite(os.path.join(output_dir, 'result.jpg'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

            if hasattr(self.detector, 'plate_region') and self.detector.plate_region is not None:
                cv2.imwrite(os.path.join(output_dir, 'plate.jpg'), self.detector.plate_region)

            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            for name, img in self.detector.debug_images.items():
                cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), img)

            QMessageBox.information(self, "成功", "结果已保存")
            self.log_text.append(f"结果已保存到: {output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def batch_process(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择多个图像文件", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )

        if not files:
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return

        self.status_label.setText(f"状态: 批量处理中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(files))

        count = 0
        success_count = 0
        for file_path in files:
            count += 1
            self.progress_bar.setValue(count)

            try:
                result, plate, chars, plate_binary, plate_number = self.detector.run(file_path)
                filename = os.path.basename(file_path)
                name, ext = os.path.splitext(filename)

                cv2.imwrite(os.path.join(output_dir, f"{name}_result{ext}"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

                if plate is not None:
                    cv2.imwrite(os.path.join(output_dir, f"{name}_plate.jpg"), plate)

                success_count += 1
                self.log_text.append(f"处理成功: {filename} - {plate_number}")
            except Exception as e:
                self.log_text.append(f"处理失败 {filename}: {e}")

        self.progress_bar.setVisible(False)
        self.status_label.setText(f"状态: 批量处理完成 ({success_count}/{len(files)})")
        QMessageBox.information(self, "完成", f"批量处理完成!\n成功: {success_count}/{len(files)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())