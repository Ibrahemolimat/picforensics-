
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageEnhance
import os
import numpy as np
import cv2
from PIL import Image, ImageChops
import tempfile
import logging
from collections import deque
from datetime import datetime
from PIL.ExifTags import TAGS
import pyexiv2
import io
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from scipy import fftpack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_ela(image_path, q=90):
    try:
        orig = Image.open(image_path).convert('RGB')
        buffer = io.BytesIO()
        orig.save(buffer, "JPEG", quality=q)
        buffer.seek(0)
        recompressed = Image.open(buffer)
        diff = ImageChops.difference(orig, recompressed)
        extrema = diff.getextrema()
        max_diff = max([e[1] for e in extrema]) or 1
        scale = 255.0 / max_diff
        ela_img = ImageEnhance.Brightness(diff).enhance(scale)
        return ela_img
    except Exception as e:
        logger.error(f"ELA computation failed: {e}")
        return None


def extract_noise(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((100, 100), dtype=np.float32)
        img_f = np.float32(img) / 255.0
        d = cv2.fastNlMeansDenoisingColored(img.astype(np.uint8), None, 10, 10, 7, 21)
        d_f = np.float32(d) / 255.0
        residual = img_f - d_f
        g = cv2.cvtColor((residual * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return (g - g.mean()) / (g.std() + 1e-8)
    except Exception as e:
        logger.error(f"Noise extraction failed: {e}")
        return np.zeros((100, 100), dtype=np.float32)


def fft_features(image_path):
    try:
        img = Image.open(image_path).convert("L")
        arr = np.asarray(img).astype(np.float32)
        F = fftpack.fftshift(fftpack.fft2(arr))
        mag = np.log1p(np.abs(F))
        return mag.std()
    except Exception as e:
        logger.error(f"FFT features failed: {e}")
        return 0.0


class Embedding:
    def __init__(self, name, device, size=224):
        try:
            self.model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg").to(device)
            self.model.eval()
            self.device = device
            self.tf = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        except Exception as e:
            logger.error(f"Embedding model {name} failed: {e}")
            self.model = None

    def extract(self, img):
        if self.model is None:
            return np.zeros(1000, dtype=np.float32)
        try:
            t = self.tf(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(t).cpu().numpy().flatten()
            return feat
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(1000, dtype=np.float32)


class SmallNoiseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )

    def forward(self, x): return self.net(x)


class CombinedAIDetectorMinimal:
    def __init__(self, device="cpu"):
        self.device = device
        self.effnet = Embedding("efficientnet_b0", device)
        self.vit = Embedding("vit_base_patch16_224", device)
        self.noise_model = SmallNoiseCNN().to(device)
        self.noise_model.eval()
        self.scaler = StandardScaler()
        self.clf = LogisticRegression()
        self._initialize_fallback_model()

    def _initialize_fallback_model(self):
        self.fallback_weights = {
            'noise_std': 0.3,
            'fft_features': 0.3,
            'file_characteristics': 0.4
        }

    def extract_features(self, path):
        try:
            img = Image.open(path).convert("RGB")
            f1 = self.effnet.extract(img)
            f2 = self.vit.extract(img)
            ela_img = compute_ela(path)
            f3 = self.effnet.extract(ela_img) if ela_img else np.zeros(1000)
            noise = extract_noise(path)
            noise_r = cv2.resize(noise, (128, 128))
            t = torch.from_numpy(noise_r).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f4 = self.noise_model(t).cpu().numpy().flatten()
            fft_std = fft_features(path)
            f5 = np.array([fft_std], dtype=np.float32)
            return np.concatenate([f1, f2, f3, f4, f5])
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._extract_fallback_features(path)

    def _extract_fallback_features(self, path):
        try:
            noise = extract_noise(path)
            fft_std = fft_features(path)
            img = Image.open(path)
            file_size = os.path.getsize(path) / (1024 * 1024)
            mp_ratio = (img.width * img.height) / (file_size * 1000000) if file_size > 0 else 0
            return np.array([noise.std(), fft_std, mp_ratio], dtype=np.float32)
        except:
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def analyze(self, path):
        try:
            feats = self.extract_features(path).reshape(1, -1)
            if hasattr(self.scaler, 'mean_') and hasattr(self.clf, 'classes_'):
                feats_s = self.scaler.transform(feats)
                p = float(self.clf.predict_proba(feats_s)[0, 1])
            else:
                p = self._fallback_analysis(path)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            p = self._fallback_analysis(path)

        conf = (
            "VERY HIGH" if p > 0.90 else
            "HIGH" if p > 0.75 else
            "MEDIUM" if p > 0.50 else
            "LOW" if p > 0.25 else
            "VERY LOW"
        )

        return {"ai_probability": round(p, 4), "confidence": conf}

    def _fallback_analysis(self, path):
        try:
            noise = extract_noise(path)
            fft_std = fft_features(path)
            img = Image.open(path)
            file_size = os.path.getsize(path) / (1024 * 1024)
            score = 0.0
            score += (1.0 - min(noise.std() / 10.0, 1.0)) * self.fallback_weights['noise_std']
            score += min(fft_std / 50.0, 1.0) * self.fallback_weights['fft_features']
            if file_size > 0 and (img.width * img.height) / (file_size * 1000000) > 10:
                score += 0.3 * self.fallback_weights['file_characteristics']
            return min(score, 1.0)
        except:
            return 0.5


class AdvancedImageTimelineAnalyzer:
    def __init__(self):
        self.date_format = "%Y:%m:%d %H:%M:%S"

    class ImageAnalysisResult:
        def __init__(self, file_name):
            self.file_name = file_name
            self.date_time_results = {}
            self.file_system_dates = {}
            self.exif_dates = {}
            self.best_guess_date = ""
            self.camera_info = {}
            self.gps_info = {}

        def get_file_name(self):
            return self.file_name

        def get_date_time_results(self):
            return self.date_time_results

        def get_file_system_dates(self):
            return self.file_system_dates

        def get_exif_dates(self):
            return self.exif_dates

        def get_best_guess_date(self):
            return self.best_guess_date

        def set_best_guess_date(self, best_guess_date):
            self.best_guess_date = best_guess_date

    def analyze_image(self, image_path):
        result = self.ImageAnalysisResult(os.path.basename(image_path))
        try:
            self.analyze_file_system_dates(image_path, result)
            self.analyze_exif_metadata(image_path, result)
            self.analyze_camera_info(image_path, result)
            self.analyze_gps_info(image_path, result)
            self.determine_best_guess_date(result)
            self.compile_all_results(result)
        except Exception as e:
            result.date_time_results["ERROR"] = f"Failed to analyze image: {str(e)}"
        return result

    def analyze_file_system_dates(self, image_path, result):
        try:
            stat_info = os.stat(image_path)
            creation_time = datetime.fromtimestamp(stat_info.st_ctime)
            modified_time = datetime.fromtimestamp(stat_info.st_mtime)
            access_time = datetime.fromtimestamp(stat_info.st_atime)
            result.file_system_dates["FileCreationTime"] = creation_time.strftime(self.date_format)
            result.file_system_dates["FileModifiedTime"] = modified_time.strftime(self.date_format)
            result.file_system_dates["FileAccessTime"] = access_time.strftime(self.date_format)
        except Exception as e:
            result.file_system_dates["ERROR"] = "File system analysis failed"

    def analyze_exif_metadata(self, image_path, result):
        try:
            self.extract_exif_with_pil(image_path, result)
            self.extract_exif_with_pyexiv2(image_path, result)
        except Exception as e:
            result.exif_dates["ERROR"] = "EXIF metadata analysis failed"

    def analyze_camera_info(self, image_path, result):
        try:
            with pyexiv2.Image(image_path) as img:
                exif_data = img.read_exif()
                camera_tags = {
                    'Exif.Image.Make': 'Camera Make',
                    'Exif.Image.Model': 'Camera Model',
                    'Exif.Photo.FNumber': 'Aperture',
                    'Exif.Photo.ExposureTime': 'Exposure Time',
                    'Exif.Photo.ISOSpeedRatings': 'ISO',
                    'Exif.Photo.FocalLength': 'Focal Length'
                }
                for exif_tag, friendly_name in camera_tags.items():
                    if exif_tag in exif_data:
                        result.camera_info[friendly_name] = exif_data[exif_tag]
        except:
            pass

    def analyze_gps_info(self, image_path, result):
        try:
            with pyexiv2.Image(image_path) as img:
                exif_data = img.read_exif()
                gps_tags = {
                    'Exif.GPSInfo.GPSLatitude': 'Latitude',
                    'Exif.GPSInfo.GPSLongitude': 'Longitude',
                    'Exif.GPSInfo.GPSAltitude': 'Altitude'
                }
                for exif_tag, friendly_name in gps_tags.items():
                    if exif_tag in exif_data:
                        result.gps_info[friendly_name] = exif_data[exif_tag]
        except:
            pass

    def extract_exif_with_pil(self, image_path, result):
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        if "date" in tag_name.lower() or "time" in tag_name.lower():
                            try:
                                if isinstance(value, str):
                                    result.exif_dates[tag_name] = value
                            except:
                                pass
        except:
            pass

    def extract_exif_with_pyexiv2(self, image_path, result):
        try:
            with pyexiv2.Image(image_path) as img:
                exif_data = img.read_exif()
                date_tags = {
                    'Exif.Photo.DateTimeOriginal': 'DateTimeOriginal',
                    'Exif.Photo.DateTimeDigitized': 'DateTimeDigitized',
                    'Exif.Image.DateTime': 'DateTime',
                    'Exif.Image.ModifyDate': 'ModifyDate'
                }
                for exif_tag, friendly_name in date_tags.items():
                    if exif_tag in exif_data:
                        result.exif_dates[friendly_name] = exif_data[exif_tag]
        except:
            pass

    def determine_best_guess_date(self, result):
        priority_order = [
            "DateTimeOriginal",
            "DateTimeDigitized",
            "DateTime",
            "ModifyDate",
            "FileCreationTime",
            "FileModifiedTime"
        ]
        for date_type in priority_order:
            date_value = self.find_date_in_sources(date_type, result)
            if date_value:
                result.set_best_guess_date(f"{date_value} ({date_type})")
                return
        result.set_best_guess_date("No reliable date found")

    def find_date_in_sources(self, date_type, result):
        if date_type in result.exif_dates:
            return result.exif_dates[date_type]
        if date_type in result.file_system_dates:
            return result.file_system_dates[date_type]
        return None

    def compile_all_results(self, result):
        result.date_time_results["=== COMPREHENSIVE ANALYSIS ==="] = ""
        result.date_time_results["--- EXIF METADATA DATES ---"] = ""
        if not result.exif_dates:
            result.date_time_results["No EXIF dates found"] = ""
        else:
            for key, value in result.exif_dates.items():
                result.date_time_results[f"{key}:"] = value
        result.date_time_results["--- FILE SYSTEM DATES ---"] = ""
        for key, value in result.file_system_dates.items():
            result.date_time_results[f"{key}:"] = value
        result.date_time_results["--- CAMERA INFORMATION ---"] = ""
        if not result.camera_info:
            result.date_time_results["No camera info found"] = ""
        else:
            for key, value in result.camera_info.items():
                result.date_time_results[f"{key}:"] = value
        result.date_time_results["--- GPS INFORMATION ---"] = ""
        if not result.gps_info:
            result.date_time_results["No GPS info found"] = ""
        else:
            for key, value in result.gps_info.items():
                result.date_time_results[f"{key}:"] = value
        result.date_time_results["--- BEST GUESS DATE ---"] = ""
        result.date_time_results["Most reliable date:"] = result.best_guess_date


class PicForensicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PicForensics Professional")
        self.root.geometry("800x600")
        self.root.configure(bg="#e3f2fd")

        self.current_image = None
        self.image_label = None
        self.uploaded_images = []
        self.current_image_index = -1
        self.ai_detector = CombinedAIDetectorMinimal()
        self.timeline_analyzer = AdvancedImageTimelineAnalyzer()

        self.create_widgets()

    def create_widgets(self):
        nav_frame = tk.Frame(self.root, bg="#1976d2", height=40)
        nav_frame.pack(fill=tk.X, padx=10, pady=5)
        nav_frame.pack_propagate(False)

        nav_items = ["HOME", "About", "Help", "Exit"]
        for item in nav_items:
            btn = tk.Button(nav_frame, text=item, relief=tk.FLAT, bg="#1976d2",
                            fg="white", font=("Arial", 9),
                            command=lambda i=item: self.nav_action(i))
            btn.pack(side=tk.LEFT, padx=10)

        separator = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, padx=10, pady=5)

        main_content = tk.Frame(self.root, bg="#e3f2fd")
        main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        content_row = tk.Frame(main_content, bg="#e3f2fd")
        content_row.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(content_row, bg="#e3f2fd")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_frame = tk.Frame(left_frame, width=400, height=300, relief=tk.RAISED,
                                    bd=2, bg='white')
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_placeholder = tk.Label(self.image_frame,
                                          text="No Image Loaded\nClick 'Open New Image' to load an image",
                                          fg="gray", font=("Arial", 10), bg='white')
        self.image_placeholder.pack(expand=True)

        self.navigation_frame = tk.Frame(left_frame, bg="#e3f2fd")
        self.navigation_frame.pack(fill=tk.X, pady=5)

        self.prev_btn = tk.Button(self.navigation_frame, text="◀ Previous", command=self.previous_image,
                                  state="disabled", bg="#2196f3", fg="white", font=("Arial", 9))
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(self.navigation_frame, text="Next ▶", command=self.next_image,
                                  state="disabled", bg="#2196f3", fg="white", font=("Arial", 9))
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.verify_btn = tk.Button(self.navigation_frame, text="Verify Integrity", command=self.verify_integrity,
                                    state="disabled", bg="#ff9800", fg="white", font=("Arial", 9))
        self.verify_btn.pack(side=tk.LEFT, padx=5)

        self.image_counter = tk.Label(self.navigation_frame, text="No images", bg="#e3f2fd", font=("Arial", 9))
        self.image_counter.pack(side=tk.LEFT, padx=10)

        self.info_frame = tk.LabelFrame(left_frame, text="Image Info", padx=10, pady=10,
                                        bg="#e3f2fd", fg="#1976d2", font=("Arial", 10, "bold"))
        self.info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_data = {
            "File Size": "No image loaded",
            "File Format": "No image loaded",
            "File Path": "No image loaded"
        }

        self.info_labels = {}
        row = 0
        for key, value in self.info_data.items():
            tk.Label(self.info_frame, text=f"{key}:", font=("Arial", 9, "bold"),
                     bg="#e3f2fd", fg="#1976d2").grid(row=row, column=0, sticky=tk.W, pady=1)
            value_label = tk.Label(self.info_frame, text=value, font=("Arial", 9),
                                   bg="#e3f2fd")
            value_label.grid(row=row, column=1, sticky=tk.W, pady=1)
            self.info_labels[key] = value_label
            row += 1

        tools_frame = tk.Frame(content_row, width=200, bg="#e3f2fd")
        tools_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        tools_frame.pack_propagate(False)

        self.analysis_var = tk.StringVar(value="Analysis Results ▼")
        analysis_menu = tk.Menubutton(tools_frame, textvariable=self.analysis_var,
                                      relief=tk.RAISED, width=18, height=1,
                                      font=("Arial", 10), bg="#2196f3", fg="white")
        analysis_menu.pack(pady=5, fill=tk.X)

        analysis_dropdown = tk.Menu(analysis_menu, tearoff=0)
        analysis_dropdown.add_command(label="Meta Data", command=self.meta_data_analysis)
        analysis_dropdown.add_command(label="Time line", command=self.timeline_analysis)
        analysis_dropdown.add_command(label="AI detection", command=self.ai_detection_analysis)

        analysis_menu.configure(menu=analysis_dropdown)

        self.tools_var = tk.StringVar(value="Analysis Tools ▼")
        tools_menu = tk.Menubutton(tools_frame, textvariable=self.tools_var,
                                   relief=tk.RAISED, width=18, height=1,
                                   font=("Arial", 10), bg="#2196f3", fg="white")
        tools_menu.pack(pady=5, fill=tk.X)

        tools_dropdown = tk.Menu(tools_menu, tearoff=0)
        tools_dropdown.add_command(label="ELA", command=self.ela_analysis)
        tools_dropdown.add_command(label="Noise", command=self.noise_analysis)
        tools_dropdown.add_command(label="Histogram", command=self.histogram_analysis)

        tools_menu.configure(menu=tools_dropdown)

        spacer = tk.Frame(tools_frame, height=20, bg="#e3f2fd")
        spacer.pack(fill=tk.X)

        img_btn_frame = tk.Frame(main_content, bg="#e3f2fd")
        img_btn_frame.pack(pady=10)

        open_btn = tk.Button(img_btn_frame, text="Open New Image", command=self.open_image,
                             bg="#2196f3", fg="white", font=("Arial", 10),
                             relief=tk.RAISED, bd=2)
        open_btn.pack(side=tk.LEFT, padx=5)

        delete_btn = tk.Button(img_btn_frame, text="Delete Image", command=self.delete_image,
                               bg="#2196f3", fg="white", font=("Arial", 10),
                               relief=tk.RAISED, bd=2)
        delete_btn.pack(side=tk.LEFT, padx=5)

    def nav_action(self, item):
        if item == "Exit":
            self.root.quit()
        elif item == "HOME":
            self.show_home_dashboard()
        elif item == "About":
            self.show_about_info()
        elif item == "Help":
            self.show_help_info()

    def show_home_dashboard(self):
        home_window = tk.Toplevel(self.root)
        home_window.title("PicForensics - Dashboard")
        home_window.geometry("600x400")
        home_window.configure(bg="#f5f5f5")

        header = tk.Label(home_window, text="📷 PicForensics Professional",
                          font=("Arial", 16, "bold"), bg="#f5f5f5", fg="#1976d2")
        header.pack(pady=20)

        stats_frame = tk.Frame(home_window, bg="#f5f5f5")
        stats_frame.pack(pady=10)

        stats = [
            f"📁 Loaded Images: {len(self.uploaded_images)}",
            f"🔍 Current Analysis: {os.path.basename(self.current_image) if self.current_image else 'None'}",
            f"🛠️ Tools Available: 6 Analysis Methods",
            f"📊 Last Result: {getattr(self, 'last_ai_result', 'No analysis yet')}"
        ]

        for stat in stats:
            lbl = tk.Label(stats_frame, text=stat, font=("Arial", 11),
                           bg="#f5f5f5", fg="#333")
            lbl.pack(pady=5)

        quick_actions = tk.Frame(home_window, bg="#f5f5f5")
        quick_actions.pack(pady=20)

        action_btn = tk.Button(quick_actions, text="Quick Integrity Check",
                               command=self.quick_integrity_check,
                               bg="#4caf50", fg="white", font=("Arial", 10))
        action_btn.pack(side=tk.LEFT, padx=10)

        close_btn = tk.Button(home_window, text="Close Dashboard",
                              command=home_window.destroy,
                              bg="#2196f3", fg="white", font=("Arial", 10))
        close_btn.pack(pady=10)

    def quick_integrity_check(self):
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        try:
            result = self.ai_detector.analyze(self.current_image)
            messagebox.showinfo("Quick Integrity Check",
                                f"AI Probability: {result['ai_probability']}\n"
                                f"Confidence: {result['confidence']}")
        except Exception as e:
            messagebox.showerror("Error", f"Quick check failed: {str(e)}")

    def show_about_info(self):
        about_text = """PicForensics Professional

Version: 3.0 Enhanced
Advanced Image Forensics Tool

Features:
• Combined AI Detection with Multiple Models
• Comprehensive Metadata Analysis
• Advanced Timeline Reconstruction
• Professional Integrity Verification

Technology Stack:
• PyTorch & EfficientNet Models
• OpenCV Image Processing
• Advanced EXIF Analysis
• Machine Learning Ensemble"""
        messagebox.showinfo("About PicForensics", about_text)

    def show_help_info(self):
        help_text = """PicForensics Help Guide

Basic Operations:
• Open New Image: Load single or multiple images
• Navigation: Use Previous/Next buttons
• Verify Integrity: Comprehensive authenticity check

Analysis Tools:
• ELA: Error Level Analysis
• Noise: Pattern analysis
• Histogram: Color distribution

Analysis Results:
• Meta Data: EXIF and file information
• Time Line: Creation and modification dates
• AI Detection: AI-generated image detection

Professional forensic analysis tool"""
        messagebox.showinfo("Help Guide", help_text)

    def verify_integrity(self):
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        try:
            integrity_report = "Integrity Verification Report\n\n"

            file_size = os.path.getsize(self.current_image) / (1024 * 1024)
            integrity_report += f"File Size: {file_size:.2f} MB\n"

            image = Image.open(self.current_image)
            integrity_report += f"Dimensions: {image.width} x {image.height}\n"
            integrity_report += f"Format: {image.format}\n\n"

            timeline_result = self.timeline_analyzer.analyze_image(self.current_image)
            integrity_report += f"Best Guess Date: {timeline_result.best_guess_date}\n\n"

            ai_result = self.ai_detector.analyze(self.current_image)
            ai_prob = ai_result['ai_probability']
            confidence = ai_result['confidence']

            integrity_report += f"AI Detection: {ai_prob:.1%}\n"
            integrity_report += f"Confidence: {confidence}\n\n"

            if ai_prob > 0.7:
                integrity_report += "WARNING: High probability of AI generation\n"
            elif ai_prob > 0.5:
                integrity_report += "NOTE: Possible AI involvement\n"
            else:
                integrity_report += "Likely authentic image\n"

            if file_size < 0.1 and image.width * image.height > 1000000:
                integrity_report += "WARNING: High resolution with small file size\n"

            self.last_ai_result = f"{ai_prob:.1%} ({confidence})"
            messagebox.showinfo("Integrity Verification", integrity_report)

        except Exception as e:
            messagebox.showerror("Error", f"Integrity verification failed: {str(e)}")

    def update_navigation_buttons(self):
        if len(self.uploaded_images) <= 1:
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
        else:
            self.prev_btn.config(state="normal")
            self.next_btn.config(state="normal")

        if self.uploaded_images:
            self.image_counter.config(text=f"Image {self.current_image_index + 1} of {len(self.uploaded_images)}")
            self.verify_btn.config(state="normal")
        else:
            self.image_counter.config(text="No images")
            self.verify_btn.config(state="disabled")

    def previous_image(self):
        if self.uploaded_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.uploaded_images and self.current_image_index < len(self.uploaded_images) - 1:
            self.current_image_index += 1
            self.load_current_image()

    def load_current_image(self):
        if 0 <= self.current_image_index < len(self.uploaded_images):
            image_path = self.uploaded_images[self.current_image_index]
            self.load_and_display_image(image_path)
            self.update_image_info(image_path)
            self.update_navigation_buttons()

    def meta_data_analysis(self):
        self.analysis_var.set(f"Analysis Results ▼")
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        try:
            result = self.timeline_analyzer.analyze_image(self.current_image)
            meta_info = "Comprehensive Metadata Analysis\n\n"
            for key, value in result.date_time_results.items():
                meta_info += f"{key}: {value}\n"
            self.show_analysis_results("Meta Data Analysis", meta_info)
        except Exception as e:
            messagebox.showerror("Error", f"Meta data analysis failed: {str(e)}")

    def timeline_analysis(self):
        self.analysis_var.set(f"Analysis Results ▼")
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        try:
            result = self.timeline_analyzer.analyze_image(self.current_image)
            timeline_info = "Timeline Analysis\n\n"
            for key, value in result.date_time_results.items():
                if "DATE" in key.upper() or "TIME" in key.upper():
                    timeline_info += f"{key}: {value}\n"
            self.show_analysis_results("Timeline Analysis", timeline_info)
        except Exception as e:
            messagebox.showerror("Error", f"Timeline analysis failed: {str(e)}")

    def ai_detection_analysis(self):
        self.analysis_var.set(f"Analysis Results ▼")
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        try:
            result = self.ai_detector.analyze(self.current_image)
            ai_info = f"AI Detection Analysis\n\n"
            ai_info += f"AI Probability: {result['ai_probability']:.2%}\n"
            ai_info += f"Confidence Level: {result['confidence']}\n"

            if result['ai_probability'] > 0.7:
                ai_info += "\nHIGH PROBABILITY: This image is likely AI-generated\n"
            elif result['ai_probability'] > 0.5:
                ai_info += "\nMODERATE PROBABILITY: Possible AI involvement\n"
            else:
                ai_info += "\nLOW PROBABILITY: Likely authentic human-created image\n"

            self.last_ai_result = f"{result['ai_probability']:.1%} ({result['confidence']})"
            self.show_analysis_results("AI Detection Analysis", ai_info)

        except Exception as e:
            messagebox.showerror("Error", f"AI detection analysis failed: {str(e)}")

    def ela_analysis(self):
        self.tools_var.set(f"Analysis Tools ▼")
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        try:
            ela_image = compute_ela(self.current_image)
            if ela_image:
                self.display_analysis_image(ela_image, "ELA Analysis")
            else:
                messagebox.showerror("Error", "ELA analysis failed to generate image")
        except Exception as e:
            messagebox.showerror("Error", f"ELA analysis failed: {str(e)}")

    def noise_analysis(self):
        self.tools_var.set(f"Analysis Tools ▼")
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        try:
            noise_array = extract_noise(self.current_image)
            noise_normalized = (noise_array - noise_array.min()) / (noise_array.max() - noise_array.min() + 1e-8)
            noise_uint8 = (noise_normalized * 255).astype(np.uint8)
            noise_image = Image.fromarray(noise_uint8)
            self.display_analysis_image(noise_image, "Noise Analysis")
        except Exception as e:
            messagebox.showerror("Error", f"Noise analysis failed: {str(e)}")

    def histogram_analysis(self):
        self.tools_var.set(f"Analysis Tools ▼")
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        try:
            import matplotlib.pyplot as plt
            image = Image.open(self.current_image).convert("RGB")
            plt.figure(figsize=(6, 4))
            colors = ('red', 'green', 'blue')
            for i, color in enumerate(colors):
                histogram = image.getchannel(i).histogram()
                plt.plot(histogram, color=color, alpha=0.7)
            plt.title('RGB Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend(['Red', 'Green', 'Blue'])
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            hist_image = Image.open(buf)
            self.display_analysis_image(hist_image, "Histogram Analysis")
            plt.close()
        except Exception as e:
            messagebox.showerror("Error", f"Histogram analysis failed: {str(e)}")

    def show_analysis_results(self, title, results):
        results_window = tk.Toplevel(self.root)
        results_window.title(title)
        results_window.geometry("500x400")
        results_window.configure(bg="white")
        text_widget = tk.Text(results_window, wrap=tk.WORD, font=("Arial", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, results)
        text_widget.config(state=tk.DISABLED)
        close_btn = tk.Button(results_window, text="Close", command=results_window.destroy,
                              bg="#2196f3", fg="white", font=("Arial", 10))
        close_btn.pack(pady=10)

    def display_analysis_image(self, pil_image, title):
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title(title)
        analysis_window.geometry("600x500")
        image_frame = tk.Frame(analysis_window)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        display_image = pil_image.copy()
        display_image.thumbnail((550, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        image_label = tk.Label(image_frame, image=photo)
        image_label.image = photo
        image_label.pack(expand=True)
        close_btn = tk.Button(analysis_window, text="Close", command=analysis_window.destroy,
                              bg="#2196f3", fg="white", font=("Arial", 10))
        close_btn.pack(pady=10)

    def open_image(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )
        if file_paths:
            new_images = list(file_paths)
            self.uploaded_images.extend(new_images)
            if self.current_image_index == -1:
                self.current_image_index = 0
            self.load_current_image()

    def load_and_display_image(self, file_path):
        self.image_placeholder.pack_forget()
        image = Image.open(file_path)
        frame_width = 380
        frame_height = 280
        image.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        if self.image_label:
            self.image_label.destroy()
        self.image_label = tk.Label(self.image_frame, image=photo, bg='white')
        self.image_label.image = photo
        self.image_label.pack(expand=True)
        self.current_image = file_path

    def update_image_info(self, file_path):
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].upper().replace(".", "")
            self.info_labels["File Size"].config(text=f"{file_size:.2f}MB")
            self.info_labels["File Format"].config(text=file_ext)
            self.info_labels["File Path"].config(text=file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not get image info: {str(e)}")

    def delete_image(self):
        if not self.current_image:
            messagebox.showwarning("Delete Image", "No image to delete!")
            return
        result = messagebox.askyesno("Delete Image", "Are you sure you want to delete this image?")
        if result:
            if 0 <= self.current_image_index < len(self.uploaded_images):
                self.uploaded_images.pop(self.current_image_index)
                if self.uploaded_images:
                    if self.current_image_index >= len(self.uploaded_images):
                        self.current_image_index = len(self.uploaded_images) - 1
                    self.load_current_image()
                else:
                    if self.image_label:
                        self.image_label.destroy()
                        self.image_label = None
                    self.image_placeholder.pack(expand=True)
                    for key in self.info_labels:
                        self.info_labels[key].config(text="No image loaded")
                    self.current_image = None
                    self.current_image_index = -1
                    self.update_navigation_buttons()
            messagebox.showinfo("Delete Image", "Image deleted successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = PicForensicsApp(root)
    root.mainloop()
