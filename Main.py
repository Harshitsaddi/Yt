"""
🚀 ULTIMATE YouTube Shorts AI Automation
GPU-ACCELERATED with AESTHETIC MULTI-WORD CAPTIONS & BLURRED BORDERS
"""

import os
import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, VideoClip, concatenate_videoclips
from sklearn.preprocessing import StandardScaler
import whisper
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import pickle
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import warnings
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from collections import deque
import re
from dataclasses import dataclass
import time
import torch
warnings.filterwarnings('ignore')

# Check GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"🖥  Device: {DEVICE.upper()}")

try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
except ImportError:
    USE_FASTER_WHISPER = False

try:
    from ultralytics import YOLO
    USE_YOLO = True
except ImportError:
    USE_YOLO = False

try:
    from transformers import pipeline
    USE_SENTIMENT = True
except ImportError:
    USE_SENTIMENT = False


@dataclass
class ViralIndicators:
    hooks: List[Tuple[float, str]] = None
    peak_moments: List[float] = None
    emotional_peaks: List[Tuple[float, float]] = None
    speech_density: float = 0.0
    visual_variety: float = 0.0
    audio_dynamics: float = 0.0

    def _post_init_(self):
        if self.hooks is None:
            self.hooks = []
        if self.peak_moments is None:
            self.peak_moments = []
        if self.emotional_peaks is None:
            self.emotional_peaks = []


class KalmanTracker:
    def _init_(self, process_noise=0.05, measurement_noise=5):
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.R *= measurement_noise
        self.kf.P *= 100
        self.kf.Q *= process_noise
        self.initialized = False
        self.lost_track_count = 0

    def update(self, x, y, confidence=1.0):
        if not self.initialized:
            self.kf.x = np.array([x, y, 0, 0])
            self.initialized = True
            self.lost_track_count = 0

        self.kf.predict()
        original_R = self.kf.R.copy()
        self.kf.R *= (2.0 - confidence)
        self.kf.update(np.array([x, y]))
        self.kf.R = original_R
        return int(self.kf.x[0]), int(self.kf.x[1])

    def predict_only(self):
        self.kf.predict()
        self.lost_track_count += 1
        return int(self.kf.x[0]), int(self.kf.x[1])


class SmoothTracker:
    def _init_(self, detect_interval=8, smooth_window=12):
        self.detect_interval = detect_interval
        self.smooth_window = smooth_window
        self.positions = deque(maxlen=smooth_window)
        self.kalman = None
        self.frame_count = 0
        self.last_detection = None
        self.confidence_history = deque(maxlen=smooth_window)

        try:
            self.kalman = KalmanTracker(process_noise=0.01, measurement_noise=2)
        except ImportError:
            pass

    def smooth_position(self, x, y, confidence=1.0):
        if len(self.positions) == 0:
            self.positions.append((x, y))
            self.confidence_history.append(confidence)
            return x, y

        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 1.0
        alpha = 0.1 + (0.15 * avg_confidence)
        prev_x, prev_y = self.positions[-1]
        smooth_x = int(alpha * x + (1 - alpha) * prev_x)
        smooth_y = int(alpha * y + (1 - alpha) * prev_y)
        self.positions.append((smooth_x, smooth_y))
        self.confidence_history.append(confidence)
        return smooth_x, smooth_y

    def get_smooth_position(self, current_pos, confidence=1.0):
        if current_pos is None:
            if self.kalman and self.kalman.initialized:
                return self.kalman.predict_only()
            elif self.last_detection:
                return self.last_detection
            else:
                return None

        x, y = current_pos
        if self.kalman:
            x, y = self.kalman.update(x, y, confidence)
        x, y = self.smooth_position(x, y, confidence)
        self.last_detection = (x, y)
        return x, y


class AdvancedViralAnalyzer:
    def _init_(self, clip_duration=60):
        self.clip_duration = clip_duration

        if USE_FASTER_WHISPER:
            self.whisper_model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE_TYPE)
        else:
            self.whisper_model = whisper.load_model("base", device=DEVICE)

        if USE_YOLO:
            self.face_detector = YOLO('yolov8n-face.pt')
            if DEVICE == "cuda":
                self.face_detector.to('cuda')
        else:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        if USE_SENTIMENT:
            try:
                device_id = 0 if DEVICE == "cuda" else -1
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device_id
                )
            except:
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None

        self.hook_patterns = [
            r'\b(wait|hold on|stop|no way|omg|wow|insane|crazy|unbelievable)\b',
            r'\b(watch this|check this|look at|see this|this is)\b',
            r'\b(secret|reveal|truth|exposed|hidden|nobody)\b',
            r'\b(how to|tutorial|learn|guide|tip|trick|hack)\b',
            r'\b(best|worst|top|most|never|always|everyone)\b',
            r'\b(money|rich|poor|expensive|cheap|free)\b',
            r'\b(before|after|transformation|change|mistake)\b',
            r'\b(question|answer|why|what|how|when|where)\b',
        ]

    def download_video(self, url: str, output_path: str = "temp_video.mp4") -> str:
        import yt_dlp

        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4'
        }

        print(f"📥 Downloading...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'video')

        return output_path, title

    def extract_advanced_features(self, video_path: str) -> Dict:
        print("🔍 Analyzing video...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        sample_rate = max(1, int(fps / 2))
        motion_scores = []
        scene_changes = []
        face_counts = []
        color_variance = []
        prev_frame = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, (320, 240))

                if prev_frame is not None:
                    diff = cv2.absdiff(gray_resized, prev_frame)
                    motion_score = np.mean(diff)
                    motion_scores.append(motion_score)

                    if motion_score > 35:
                        scene_changes.append(frame_idx / fps)

                prev_frame = gray_resized

                if USE_YOLO:
                    results = self.face_detector(frame, verbose=False)
                    face_count = len(results[0].boxes) if len(results) > 0 else 0
                else:
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    face_count = len(faces)

                face_counts.append(face_count)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color_var = np.std(hsv)
                color_variance.append(color_var)

            frame_idx += 1

        cap.release()

        try:
            y, sr = librosa.load(video_path, sr=22050, duration=duration, mono=True)
            energy = librosa.feature.rms(y=y)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)

            time_steps = len(motion_scores)

            if len(energy) > 0 and time_steps > 0:
                audio_energy = np.interp(
                    np.linspace(0, len(energy) - 1, time_steps),
                    np.arange(len(energy)),
                    energy
                )
                audio_zcr = np.interp(
                    np.linspace(0, len(zcr) - 1, time_steps),
                    np.arange(len(zcr)),
                    zcr
                )
            else:
                audio_energy = np.zeros(time_steps)
                audio_zcr = np.zeros(time_steps)

            audio_dynamics = np.std(energy) / (np.mean(energy) + 1e-6)

        except Exception as e:
            time_steps = len(motion_scores)
            audio_energy = np.zeros(time_steps)
            audio_zcr = np.zeros(time_steps)
            audio_dynamics = 0
            tempo = 120
            beat_times = []

        print("🗣 Transcribing...")
        transcript_segments = []
        viral_indicators = ViralIndicators()

        try:
            if USE_FASTER_WHISPER:
                segments, info = self.whisper_model.transcribe(
                    video_path,
                    beam_size=5,
                    word_timestamps=True,
                    language='en'
                )

                for segment in segments:
                    text = segment.text.strip()
                    segment_data = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': text
                    }

                    for pattern in self.hook_patterns:
                        if re.search(pattern, text.lower()):
                            viral_indicators.hooks.append((segment.start, text))
                            break

                    if self.sentiment_analyzer:
                        try:
                            sentiment = self.sentiment_analyzer(text[:512])[0]
                            if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.9:
                                viral_indicators.emotional_peaks.append(
                                    (segment.start, sentiment['score'])
                                )
                        except:
                            pass

                    transcript_segments.append(segment_data)

            else:
                result = self.whisper_model.transcribe(video_path, word_timestamps=True)
                for segment in result['segments']:
                    text = segment['text'].strip()
                    segment_data = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text
                    }

                    for pattern in self.hook_patterns:
                        if re.search(pattern, text.lower()):
                            viral_indicators.hooks.append((segment['start'], text))
                            break

                    transcript_segments.append(segment_data)

            total_words = sum(len(seg['text'].split()) for seg in transcript_segments)
            viral_indicators.speech_density = total_words / duration if duration > 0 else 0

        except Exception as e:
            pass

        viral_indicators.visual_variety = len(scene_changes) / duration if duration > 0 else 0
        viral_indicators.audio_dynamics = audio_dynamics

        return {
            'duration': duration,
            'fps': fps,
            'motion_scores': motion_scores,
            'audio_energy': audio_energy,
            'audio_zcr': audio_zcr,
            'scene_changes': scene_changes,
            'transcript_segments': transcript_segments,
            'tempo': tempo,
            'beat_times': beat_times,
            'face_counts': face_counts,
            'color_variance': color_variance,
            'viral_indicators': viral_indicators
        }

    def calculate_viral_scores(self, features: Dict) -> np.ndarray:
        print("📊 Calculating scores...")

        motion = np.array(features['motion_scores'])
        audio = np.array(features['audio_energy'])
        zcr = np.array(features['audio_zcr'])
        faces = np.array(features['face_counts'])
        colors = np.array(features['color_variance'])

        min_length = min(len(motion), len(audio), len(zcr), len(faces), len(colors))

        motion = motion[:min_length]
        audio = audio[:min_length]
        zcr = zcr[:min_length]
        faces = faces[:min_length]
        colors = colors[:min_length]

        scaler = StandardScaler()
        motion_norm = scaler.fit_transform(motion.reshape(-1, 1)).flatten()
        audio_norm = scaler.fit_transform(audio.reshape(-1, 1)).flatten()
        zcr_norm = scaler.fit_transform(zcr.reshape(-1, 1)).flatten()

        engagement = (
                0.25 * motion_norm +
                0.25 * audio_norm +
                0.15 * zcr_norm +
                0.10 * (faces > 0).astype(float) +
                0.10 * (colors / np.max(colors) if np.max(colors) > 0 else colors)
        )

        viral_indicators = features['viral_indicators']

        for hook_time, hook_text in viral_indicators.hooks:
            idx = int(hook_time)
            if idx < len(engagement):
                engagement[idx:min(idx+5, len(engagement))] += 1.0

        for peak_time, score in viral_indicators.emotional_peaks:
            idx = int(peak_time)
            if idx < len(engagement):
                engagement[idx:min(idx+3, len(engagement))] += 0.5 * score

        for change_time in features['scene_changes']:
            idx = int(change_time)
            if idx < len(engagement):
                engagement[idx:min(idx+2, len(engagement))] += 0.3

        for beat_time in features['beat_times']:
            idx = int(beat_time)
            if idx < len(engagement):
                engagement[idx] += 0.2

        for segment in features['transcript_segments']:
            start_idx = int(segment['start'])
            end_idx = int(segment['end'])
            if start_idx < len(engagement):
                engagement[start_idx:min(end_idx, len(engagement))] += 0.4

        window = 7
        engagement_smoothed = np.convolve(engagement, np.ones(window)/window, mode='same')
        engagement_smoothed = (engagement_smoothed - engagement_smoothed.min()) / (
                engagement_smoothed.max() - engagement_smoothed.min() + 1e-6
        ) * 100

        return engagement_smoothed

    def find_viral_clips(self, engagement_scores: np.ndarray, features: Dict,
                         n_clips: int = 3) -> List[Tuple[int, int, float]]:
        clip_len = self.clip_duration
        viral_indicators = features['viral_indicators']
        candidates = []

        for start in range(0, max(1, len(engagement_scores) - clip_len), 3):
            end = min(start + clip_len, len(engagement_scores))
            avg_engagement = np.mean(engagement_scores[start:end])
            max_engagement = np.max(engagement_scores[start:end])
            hooks_in_range = sum(1 for t, _ in viral_indicators.hooks if start <= t < end)
            peaks_in_range = sum(1 for t, _ in viral_indicators.emotional_peaks if start <= t < end)

            viral_score = (
                    0.4 * avg_engagement +
                    0.3 * max_engagement +
                    0.2 * hooks_in_range * 20 +
                    0.1 * peaks_in_range * 15
            )

            candidates.append((start, end, viral_score, hooks_in_range, peaks_in_range))

        candidates.sort(key=lambda x: x[2], reverse=True)

        selected_clips = []
        for clip in candidates:
            start, end, score, hooks, peaks = clip
            overlap = False
            for selected in selected_clips:
                if not (end < selected[0] or start > selected[1]):
                    overlap = True
                    break

            if not overlap:
                selected_clips.append((start, end, score, hooks, peaks))

            if len(selected_clips) >= n_clips:
                break

        return sorted([(s, e, sc) for s, e, sc, _, _ in selected_clips])


class ProCaptionGenerator:
    def _init_(self):
        if USE_FASTER_WHISPER:
            self.whisper_model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE_TYPE)
        else:
            self.whisper_model = whisper.load_model("base", device=DEVICE)

        if USE_YOLO:
            self.face_detector = YOLO('yolov8n-face.pt')
            if DEVICE == "cuda":
                self.face_detector.to('cuda')
        else:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        self.caption_style = {
            'font_size': 80,
            'color': '#FFFFFF',
            'outline_color': '#000000',
            'outline_width': 8,
            'position': 400
        }

    def detect_face_confidence(self, frame):
        if USE_YOLO:
            results = self.face_detector(frame, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                largest_idx = np.argmax(areas)
                box = boxes[largest_idx]
                conf = confs[largest_idx]
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                return (center_x, center_y), float(conf)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                return (x + w // 2, y + h // 2), 0.8

        return None, 0.0

    def track_face_smooth(self, video_path: str, detect_interval: int = 8) -> List[Tuple[int, int, float]]:
        print("🎯 Tracking face...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps

        tracker = SmoothTracker(detect_interval=detect_interval, smooth_window=12)
        keyframes = []
        frame_idx = 0
        center_default = (width // 2, height // 2)

        while cap.isOpened() and frame_idx < frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            if frame_idx % detect_interval == 0:
                detected, confidence = self.detect_face_confidence(frame)

                if detected and confidence > 0.3:
                    smooth_pos = tracker.get_smooth_position(detected, confidence)
                    if smooth_pos:
                        keyframes.append((smooth_pos[0], smooth_pos[1], timestamp))
                elif tracker.last_detection:
                    keyframes.append((tracker.last_detection[0], tracker.last_detection[1], timestamp))
                else:
                    keyframes.append((center_default[0], center_default[1], timestamp))

            frame_idx += 1

        cap.release()

        interpolated = self.interpolate_cubic(keyframes, duration, fps)
        return interpolated

    def interpolate_cubic(self, keyframes: List[Tuple[int, int, float]],
                          duration: float, fps: float) -> List[Tuple[int, int, float]]:
        if len(keyframes) < 2:
            return keyframes

        from scipy.interpolate import CubicSpline

        timestamps = np.array([kf[2] for kf in keyframes])
        x_positions = np.array([kf[0] for kf in keyframes])
        y_positions = np.array([kf[1] for kf in keyframes])

        try:
            cs_x = CubicSpline(timestamps, x_positions, bc_type='natural')
            cs_y = CubicSpline(timestamps, y_positions, bc_type='natural')
            sample_times = np.arange(0, duration, 0.05)

            interpolated = []
            for t in sample_times:
                x = int(np.clip(cs_x(t), 0, 10000))
                y = int(np.clip(cs_y(t), 0, 10000))
                interpolated.append((x, y, t))

            return interpolated
        except:
            return keyframes

    def group_words(self, words: List[Dict], max_words: int = 3, max_duration: float = 2.0) -> List[Dict]:
        if not words:
            return []

        grouped = []
        current_group = []
        current_start = None

        for word in words:
            if not current_group:
                current_group = [word['text']]
                current_start = word['start']
                current_end = word['end']
            else:
                duration = word['end'] - current_start

                if len(current_group) < max_words and duration < max_duration:
                    current_group.append(word['text'])
                    current_end = word['end']
                else:
                    grouped.append({
                        'text': ' '.join(current_group),
                        'start': current_start,
                        'end': current_end
                    })
                    current_group = [word['text']]
                    current_start = word['start']
                    current_end = word['end']

        if current_group:
            grouped.append({
                'text': ' '.join(current_group),
                'start': current_start,
                'end': current_end
            })

        return grouped

    def generate_word_captions(self, video_path: str, clip_start: int,
                               clip_end: int) -> List[Dict]:
        print("💬 Generating captions...")

        try:
            words = []

            if USE_FASTER_WHISPER:
                segments, info = self.whisper_model.transcribe(
                    video_path,
                    beam_size=5,
                    word_timestamps=True,
                    language='en'
                )

                for segment in segments:
                    for word in segment.words:
                        if clip_start <= word.start <= clip_end:
                            words.append({
                                'text': word.word.strip(),
                                'start': word.start - clip_start,
                                'end': word.end - clip_start
                            })
            else:
                result = self.whisper_model.transcribe(video_path, word_timestamps=True)
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_data in segment['words']:
                            if clip_start <= word_data['start'] <= clip_end:
                                words.append({
                                    'text': word_data['word'].strip(),
                                    'start': word_data['start'] - clip_start,
                                    'end': word_data['end'] - clip_start
                                })

            captions = self.group_words(words, max_words=3, max_duration=2.0)
            return captions

        except Exception as e:
            return []

    def create_professional_short(self, video_clip: VideoFileClip, captions: List[Dict],
                                  output_path: str, focus_timeline: List[Tuple[int, int, float]] = None,
                                  add_effects: bool = True):
        print(f"🎬 Creating Short with blurred borders...")

        orig_w, orig_h = video_clip.size
        target_w, target_h = 1080, 1920

        # Calculate scale to fit video INSIDE 9:16 (no cropping)
        scale = min(target_w / orig_w, target_h / orig_h)
        video_resized = video_clip.resize(scale)
        new_w, new_h = video_resized.size

        # Create BLURRED background
        blur_scale = max(target_w / orig_w, target_h / orig_h)
        video_background = video_clip.resize(blur_scale)

        def make_blurred_bg(get_frame, t):
            frame = get_frame(t)
            # Crop to 9:16
            bg_h, bg_w = frame.shape[:2]
            x_crop = (bg_w - target_w) // 2
            y_crop = (bg_h - target_h) // 2
            cropped = frame[y_crop:y_crop + target_h, x_crop:x_crop + target_w]
            # Heavy blur
            blurred = cv2.GaussianBlur(cropped, (99, 99), 50)
            # Darken
            blurred = (blurred * 0.4).astype(np.uint8)
            return blurred

        background_clip = VideoClip(
            make_frame=lambda t: make_blurred_bg(video_background.get_frame, t),
            duration=video_clip.duration
        ).set_fps(min(video_clip.fps, 30))

        # Position main video in center
        x_pos = (target_w - new_w) // 2
        y_pos = (target_h - new_h) // 2
        video_resized = video_resized.set_position((x_pos, y_pos))

        # Create composite with background + main video
        base_composite = CompositeVideoClip([background_clip, video_resized], size=(target_w, target_h))
        base_composite = base_composite.set_audio(video_clip.audio)

        font = self.load_best_font(self.caption_style['font_size'])
        text_clips = []
        style = self.caption_style

        for i, caption in enumerate(captions):
            text = caption['text'].upper()

            img = Image.new('RGBA', (target_w, 300), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = (target_w - text_width) // 2
            y = 50

            outline_width = style['outline_width']
            for adj_x in range(-outline_width, outline_width + 1):
                for adj_y in range(-outline_width, outline_width + 1):
                    if adj_x != 0 or adj_y != 0:
                        draw.text(
                            (x + adj_x, y + adj_y),
                            text,
                            font=font,
                            fill=style['outline_color']
                        )

            draw.text((x, y), text, font=font, fill=style['color'])

            img_array = np.array(img)

            duration = caption['end'] - caption['start']

            caption_clip = ImageClip(img_array, duration=duration, transparent=True)
            caption_clip = caption_clip.set_start(caption['start']).set_position(('center', style['position']))

            text_clips.append(caption_clip)

        if add_effects:
            progress_clip = self.create_progress_bar(video_clip.duration, target_w, target_h)
            text_clips.append(progress_clip)

        final_video = CompositeVideoClip([base_composite] + text_clips)

        print("🎞 Rendering...")

        try:
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=30,
                preset='medium',
                bitrate='6000k',
                audio_bitrate='192k',
                threads=4,
                logger=None
            )
            print(f"✅ Saved: {output_path}")
        finally:
            final_video.close()
            base_composite.close()
            background_clip.close()
            video_background.close()
            video_resized.close()
            for clip in text_clips:
                try:
                    clip.close()
                except:
                    pass

    def load_best_font(self, size: int):
        font_paths = [
            "C:\\Windows\\Fonts\\impact.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except:
                    continue

        return ImageFont.load_default()

    def create_progress_bar(self, duration: float, width: int, height: int):
        def make_progress_frame(t):
            img = np.zeros((30, width, 3), dtype=np.uint8)
            progress = int((t / duration) * width)
            img[:, :] = [30, 30, 30]
            if progress > 0:
                img[:, :progress] = [255, 50, 50]
            return img

        progress_clip = VideoClip(make_frame=make_progress_frame, duration=duration)
        progress_clip = progress_clip.set_position((0, height - 30))
        return progress_clip


class YouTubeUploader:
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

    def _init_(self, credentials_file: str = 'client_secrets.json'):
        self.credentials_file = credentials_file
        self.youtube = self._get_authenticated_service()

    def _get_authenticated_service(self):
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)

            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return build('youtube', 'v3', credentials=creds)

    def upload_video(self, video_path: str, title: str, description: str,
                     tags: List[str], category_id: str = '22') -> str:
        print(f"📤 Uploading: {title}")

        if '#Shorts' not in title and '#shorts' not in title:
            title = f"{title} #Shorts"

        hashtags = ['#Shorts', '#YouTubeShorts', '#Viral', '#Trending']
        description = f"{description}\n\n{' '.join(hashtags)}"

        body = {
            'snippet': {
                'title': title[:100],
                'description': description[:5000],
                'tags': tags + ['shorts', 'youtubeshorts', 'viral', 'trending'],
                'categoryId': category_id
            },
            'status': {
                'privacyStatus': 'public',
                'selfDeclaredMadeForKids': False
            }
        }

        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        request = self.youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media
        )

        response = request.execute()
        video_id = response['id']

        print(f"✅ Uploaded!")
        print(f"🔗 https://youtube.com/shorts/{video_id}")

        return video_id


def main(video_url: str, n_clips: int = 3, clip_duration: int = 60,
         upload_to_youtube: bool = False, detect_interval: int = 8,
         add_effects: bool = True):

    print("=" * 80)
    print("🚀 YOUTUBE SHORTS AI - GPU ACCELERATED WITH BLURRED BORDERS")
    print("=" * 80)

    analyzer = AdvancedViralAnalyzer(clip_duration=clip_duration)
    caption_gen = ProCaptionGenerator()
    uploader = YouTubeUploader() if upload_to_youtube else None

    video_path, video_title = analyzer.download_video(video_url)
    features = analyzer.extract_advanced_features(video_path)
    viral_scores = analyzer.calculate_viral_scores(features)
    best_clips = analyzer.find_viral_clips(viral_scores, features, n_clips=n_clips)

    print(f"\n🎯 FOUND {len(best_clips)} VIRAL CLIPS")
    for i, (start, end, score) in enumerate(best_clips):
        print(f"  Clip {i+1}: {start}s-{end}s | Score: {score:.1f}/100")

    created_shorts = []

    for i, (start, end, score) in enumerate(best_clips):
        print(f"\n{'='*80}")
        print(f"CLIP {i+1}/{len(best_clips)} (Score: {score:.1f})")
        print('='*80)

        clip_path = f"clip_{i+1}_temp.mp4"

        video = VideoFileClip(video_path).subclip(start, end)
        video.write_videofile(clip_path, codec='libx264', audio_codec='aac',
                              logger=None, preset='fast')
        video.close()

        focus_timeline = caption_gen.track_face_smooth(clip_path, detect_interval)
        captions = caption_gen.generate_word_captions(video_path, start, end)

        output_path = f"short_{i+1}_final.mp4"
        video_clip = VideoFileClip(clip_path)
        caption_gen.create_professional_short(
            video_clip,
            captions,
            output_path,
            focus_timeline=focus_timeline,
            add_effects=add_effects
        )

        video_clip.close()
        created_shorts.append(output_path)

        if uploader:
            title = f"{video_title[:50]} - Viral Moment #{i+1}"
            description = (
                f"🔥 AI-extracted viral clip from: {video_title}\n\n"
                f"Engagement Score: {score:.1f}/100\n"
                f"Created with advanced AI tracking & captions!"
            )
            tags = ['viral', 'trending', 'ai', 'shorts', 'youtubeshorts']
            uploader.upload_video(output_path, title, description, tags)

        for attempt in range(5):
            try:
                os.remove(clip_path)
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.5)

    for attempt in range(5):
        try:
            os.remove(video_path)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(0.5)

    print(f"\n{'='*80}")
    print("🎉 ALL SHORTS CREATED!")
    print('='*80)
    print(f"Total: {len(created_shorts)}")
    for short in created_shorts:
        file_size = os.path.getsize(short) / (1024*1024)
        print(f"  • {short} ({file_size:.1f} MB)")
    print('='*80)


if _name_ == "_main_":
    print("\n" + "=" * 80)
    print("🎬 YOUTUBE SHORTS AI - GPU ACCELERATED WITH BLURRED BORDERS")
    print("=" * 80)
    print("\n🚀 FEATURES:")
    print("  ✓ GPU-ACCELERATED processing")
    print("  ✓ BLURRED BORDERS (modern TikTok/Instagram style)")
    print("  ✓ No black bars - video fits perfectly in 9:16")
    print("  ✓ Aesthetic multi-word captions (2-3 words)")
    print("  ✓ Viral moment detection with AI")
    print("  ✓ Ultra-smooth face tracking")
    print("=" * 80)

    VIDEO_URL = input("\n🎥 YouTube URL: ").strip()
    NUM_CLIPS = int(input("📊 Number of Shorts (default 3): ") or "3")
    CLIP_DURATION = int(input("⏱  Duration per Short in seconds (default 60): ") or "60")
    DETECT_INTERVAL = int(input("🎯 Face detection interval (default 8): ") or "8")

    effects_choice = input("✨ Add effects? (yes/no, default yes): ").lower()
    ADD_EFFECTS = effects_choice != 'no'

    upload_choice = input("📤 Upload to YouTube? (yes/no, default no): ").lower()
    UPLOAD_TO_YOUTUBE = upload_choice == 'yes'

    print(f"\n{'='*80}")
    print("⚙  CONFIGURATION:")
    print(f"  • Device: {DEVICE.upper()}")
    print(f"  • Clips: {NUM_CLIPS}")
    print(f"  • Duration: {CLIP_DURATION}s")
    print(f"  • Style: Blurred Borders + Multi-Word Captions")
    print(f"  • Upload: {'YES' if UPLOAD_TO_YOUTUBE else 'NO'}")
    print('='*80 + "\n")

    try:
        main(
            VIDEO_URL,
            n_clips=NUM_CLIPS,
            clip_duration=CLIP_DURATION,
            upload_to_youtube=UPLOAD_TO_YOUTUBE,
            detect_interval=DETECT_INTERVAL,
            add_effects=ADD_EFFECTS
        )
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()