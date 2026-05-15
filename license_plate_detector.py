import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
from typing import List, Tuple, Optional, Dict, Any


class LicensePlateDetector:
    """
    车牌检测类，包含诊断和调试功能
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # 控制台处理器
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.debug_images: Dict[str, np.ndarray] = {}
        
        # 可配置参数
        self.params: Dict[str, Any] = {
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
    
    def load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        self.logger.info(f"加载图像: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"无法读取图像: {image_path}")
            raise ValueError(f"无法读取图像: {image_path}")
        
        self.img_original = img
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.logger.info(f"图像尺寸: {img.shape}")
        return img
    
    def preprocess_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """图像预处理流程"""
        self.logger.info("开始预处理...")
        
        # 1. 灰度化
        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        self.debug_images['gray'] = gray
        
        # 2. 高斯滤波去噪
        blurred = cv2.GaussianBlur(gray, self.params['gaussian_kernel'], 0)
        self.debug_images['blurred'] = blurred
        
        # 3. Otsu二值化
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.debug_images['binary'] = binary
        
        self.logger.info(f"Otsu阈值: {_}")
        self.gray, self.blurred, self.binary = gray, blurred, binary
        return gray, blurred, binary
    
    def edge_detection(self) -> np.ndarray:
        """边缘检测"""
        self.logger.info("边缘检测...")
        edges = cv2.Canny(self.blurred, self.params['canny_low'], self.params['canny_high'])
        self.debug_images['edges'] = edges
        self.logger.info(f"边缘像素数: {np.count_nonzero(edges)}")
        self.edges = edges
        return edges
    
    def morphological_processing(self) -> np.ndarray:
        """形态学运算"""
        self.logger.info("形态学处理...")
        
        # 闭运算连接边缘
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, self.params['morph_kernel_close'])
        closed = cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # 开运算去除噪点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, self.params['morph_kernel_open'])
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        self.debug_images['morphological'] = opened
        self.logger.info(f"形态学处理后像素数: {np.count_nonzero(opened)}")
        self.morph_result = opened
        return opened
    
    def find_license_plate(self) -> List[Dict[str, Any]]:
        """查找车牌候选区域"""
        self.logger.info("查找车牌区域...")
        
        contours, _ = cv2.findContours(self.morph_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.info(f"检测到轮廓数: {len(contours)}")
        
        candidates = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / h
            
            # 宽高比筛选
            if not (self.params['aspect_ratio_min'] < aspect_ratio < self.params['aspect_ratio_max']):
                continue
            
            # 面积筛选
            if not (self.params['area_min'] < area < self.params['area_max']):
                continue
            
            # 计算矩形度
            rect_area = w * h
            rect_score = area / rect_area
            
            candidates.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'aspect_ratio': aspect_ratio,
                'area': area,
                'score': rect_score,
                'contour': cnt
            })
            self.logger.debug(f"候选区域{i}: x={x}, y={y}, w={w}, h={h}, 宽高比={aspect_ratio:.2f}, 面积={area}")
        
        # 按面积排序
        candidates.sort(key=lambda x: x['area'], reverse=True)
        self.candidates = candidates[:3]
        self.logger.info(f"筛选后候选数: {len(self.candidates)}")
        return self.candidates
    
    def extract_plate_region(self) -> Optional[np.ndarray]:
        """提取车牌区域"""
        if not self.candidates:
            self.logger.warning("没有找到候选区域")
            return None
        
        best = self.candidates[0]
        x, y, w, h = best['x'], best['y'], best['w'], best['h']
        
        # 扩大边界
        padding = 3
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(self.img_original.shape[1] - x, w + padding * 2)
        h = min(self.img_original.shape[0] - y, h + padding * 2)
        
        # 提取并转为灰度
        plate_region = cv2.cvtColor(self.img_original[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # 归一化尺寸
        plate_region = cv2.resize(plate_region, (self.params['plate_width'], self.params['plate_height']))
        
        self.debug_images['plate_region'] = plate_region
        self.plate_region = plate_region
        return plate_region
    
    def segment_characters(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """字符分割"""
        if not hasattr(self, 'plate_region') or self.plate_region is None:
            self.logger.warning("无法分割字符：没有车牌区域")
            return [], None
        
        self.logger.info("字符分割...")
        
        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            self.plate_region, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            9, 2
        )
        
        # 垂直投影
        vertical_proj = np.sum(binary, axis=0)
        proj_min = np.min(vertical_proj)
        proj_max = np.max(vertical_proj)
        proj_diff = proj_max - proj_min
        
        self.logger.debug(f"投影范围: {proj_min}-{proj_max}, 差值: {proj_diff}")
        
        # 如果投影差异很小，尝试Otsu二值化
        if proj_diff < 1000:
            self.logger.debug("投影差异小，切换到Otsu二值化")
            _, binary = cv2.threshold(self.plate_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            vertical_proj = np.sum(binary, axis=0)
            proj_min = np.min(vertical_proj)
            proj_max = np.max(vertical_proj)
            proj_diff = proj_max - proj_min
        
        # 使用最小值加上差值的10%作为阈值
        threshold = proj_min + proj_diff * 0.1
        
        # 找到字符边界
        chars = []
        start = None
        
        for i, val in enumerate(vertical_proj):
            if val > threshold and start is None:
                start = i
            elif val <= threshold and start is not None:
                width = i - start
                if 6 < width < 30:
                    chars.append((start, i))
                start = None
        
        if start is not None and (len(vertical_proj) - start) > 6:
            chars.append((start, len(vertical_proj)))
        
        # 如果检测到的字符太少，使用固定宽度分割
        if len(chars) < 3:
            self.logger.debug("字符数不足，使用固定宽度分割")
            char_width = len(vertical_proj) // 7
            chars = []
            for i in range(7):
                start = i * char_width
                end = start + char_width
                if end <= len(vertical_proj):
                    chars.append((start, end))
        
        # 提取字符
        character_images = []
        for start, end in chars[:7]:
            char_img = binary[:, start:end]
            char_img = cv2.resize(char_img, (self.params['char_width'], self.params['char_height']))
            character_images.append(char_img)
        
        self.debug_images['plate_binary'] = binary
        self.logger.info(f"分割出 {len(character_images)} 个字符")
        return character_images, binary
    
    def draw_results(self) -> np.ndarray:
        """绘制检测结果"""
        result = self.img_rgb.copy()
        
        for idx, candidate in enumerate(self.candidates):
            x, y, w, h = candidate['x'], candidate['y'], candidate['w'], candidate['h']
            color = (0, 255, 0) if idx == 0 else (255, 0, 0)
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result, f"候选{idx+1}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def save_debug_images(self, output_dir: str = 'debug_output') -> None:
        """保存调试图像"""
        os.makedirs(output_dir, exist_ok=True)
        for name, img in self.debug_images.items():
            cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img)
        self.logger.info(f"调试图像已保存到 {output_dir}")
    
    def run(self, image_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        """完整检测流程"""
        try:
            self.load_image(image_path)
            self.preprocess_image()
            self.edge_detection()
            self.morphological_processing()
            self.find_license_plate()
            plate = self.extract_plate_region()
            chars = []
            plate_binary = None
            
            if plate is not None:
                chars, plate_binary = self.segment_characters()
            
            result = self.draw_results()
            return result, plate, chars, plate_binary
        
        except Exception as e:
            self.logger.error(f"检测失败: {e}")
            raise


def visualize_results(detector):
    """可视化处理流程"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 主处理流程
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(detector.img_rgb)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(detector.gray, cmap='gray')
    axes[0, 1].set_title('灰度图像')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(detector.binary, cmap='gray')
    axes[0, 2].set_title('二值化图像')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(detector.edges, cmap='gray')
    axes[1, 0].set_title('边缘检测(Canny)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(detector.morph_result, cmap='gray')
    axes[1, 1].set_title('形态学处理')
    axes[1, 1].axis('off')
    
    result = detector.draw_results()
    axes[1, 2].imshow(result)
    axes[1, 2].set_title('检测结果')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 显示车牌区域
    if hasattr(detector, 'plate_region') and detector.plate_region is not None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].imshow(detector.plate_region, cmap='gray')
        ax[0].set_title('提取的车牌区域')
        ax[0].axis('off')
        
        ax[1].imshow(detector.debug_images.get('plate_binary', detector.plate_region), cmap='gray')
        ax[1].set_title('二值化车牌')
        ax[1].axis('off')
        plt.show()


def main():
    """主函数"""
    import argparse
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='车牌检测程序')
    parser.add_argument('--image', default=os.path.join(script_dir, 'license.png'), help='输入图像路径')
    parser.add_argument('--verbose', action='store_true', help='启用详细日志')
    parser.add_argument('--save-debug', action='store_true', help='保存调试图像')
    
    args = parser.parse_args()
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在 - {args.image}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"脚本目录: {script_dir}")
        return
    
    # 创建检测器
    detector = LicensePlateDetector(verbose=args.verbose)
    
    try:
        result, plate, chars, plate_binary = detector.run(args.image)
        
        # 可视化
        visualize_results(detector)
        
        # 保存调试图像
        if args.save_debug:
            detector.save_debug_images()
        
        # 输出结果
        print(f"\n=== 检测结果 ===")
        print(f"检测到候选区域: {len(detector.candidates)} 个")
        for i, c in enumerate(detector.candidates):
            print(f"  候选{i+1}: x={c['x']}, y={c['y']}, w={c['w']}, h={c['h']}")
            print(f"            宽高比: {c['aspect_ratio']:.2f}, 面积: {c['area']}, 矩形度: {c['score']:.2f}")
        print(f"提取到字符: {len(chars)} 个")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()