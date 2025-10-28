"""
🧠 ULTIMATE AI-POWERED SHORTS CREATOR
Advanced Multi-Modal Context Understanding with Vision-Language Models
"""

import os
import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, VideoClip
from sklearn.preprocessing import StandardScaler
import whisper
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import pickle
from typing import List, Tuple, Dict, Optional
import warnings
from PIL import Image, ImageDraw, ImageFont
import re
from dataclasses import dataclass, field
import time
import torch
import gc
from scipy.signal import find_peaks
from collections import Counter, defaultdict
import json

warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"🖥  Device: {DEVICE.upper()}")

# Advanced ML Models
try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
except ImportError:
    USE_FASTER_WHISPER = False
    print("⚠  Install faster-whisper for speed: pip install faster-whisper")

try:
    from transformers import pipeline, CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
    USE_TRANSFORMERS = True
    print("✓ Transformers available - Advanced AI enabled")
except ImportError:
    USE_TRANSFORMERS = False
    print("⚠  Install transformers: pip install transformers")

try:
    from ultralytics import YOLO
    USE_YOLO = True
    print("✓ YOLO available - Object detection enabled")
except ImportError:
    USE_YOLO = False
    print("⚠  Install ultralytics: pip install ultralytics")


@dataclass
class ContextualMoment:
    """Rich contextual information about a moment"""
    timestamp: float
    duration: float

    # Speech analysis
    speech_text: str = ""
    speech_sentiment: str = ""
    speech_confidence: float = 0.0
    speech_emotion: str = ""
    entities: List[str] = field(default_factory=list)

    # Visual analysis
    visual_scene: str = ""
    visual_objects: List[str] = field(default_factory=list)
    visual_actions: List[str] = field(default_factory=list)
    face_emotions: List[str] = field(default_factory=list)

    # Audio analysis
    audio_intensity: float = 0.0
    audio_emotion: str = ""
    music_present: bool = False

    # Viral indicators
    viral_score: float = 0.0
    viral_triggers: List[str] = field(default_factory=list)
    hook_strength: float = 0.0
    retention_score: float = 0.0

    # Context
    content_type: str = ""  # tutorial, reaction, story, reveal, comedy, etc.
    narrative_role: str = ""  # setup, build, climax, resolution
    standalone_quality: float = 0.0


class AdvancedAIAnalyzer:
    """
    State-of-the-art multi-modal AI analysis system
    """

    def _init_(self, clip_duration=60):
        self.clip_duration = clip_duration

        # Whisper for transcription
        if USE_FASTER_WHISPER:
            print("Loading Whisper model...")
            self.whisper_model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE_TYPE)
        else:
            self.whisper_model = whisper.load_model("base", device=DEVICE)

        if USE_TRANSFORMERS:
            # CLIP for visual understanding
            print("Loading CLIP (visual-language model)...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            if DEVICE == "cuda":
                self.clip_model = self.clip_model.to("cuda")

            # BLIP for image captioning
            print("Loading BLIP (image captioning)...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            if DEVICE == "cuda":
                self.blip_model = self.blip_model.to("cuda")

            # Sentiment analysis
            print("Loading sentiment analyzer...")
            device_id = 0 if DEVICE == "cuda" else -1
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device_id
            )

            # Emotion detection
            print("Loading emotion detector...")
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=device_id
            )

            # Zero-shot classification for content type
            print("Loading content classifier...")
            self.content_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device_id
            )

        if USE_YOLO:
            print("Loading YOLO models...")
            self.object_detector = YOLO('yolov8n.pt')
            self.face_detector = YOLO('yolov8n-face.pt')
            if DEVICE == "cuda":
                self.object_detector.to('cuda')
                self.face_detector.to('cuda')

        # Viral content patterns (based on real viral shorts analysis)
        self.viral_patterns = {
            'hooks': [
                r'\b(wait|stop|hold on|hold up|pause)\b',
                r'\b(no way|omg|wow|insane|crazy|wild|unbelievable)\b',
                r'\b(you won\'?t believe|can\'?t believe)\b',
                r'\b(this is|this just)\b',
            ],
            'questions': [
                r'\?',
                r'\b(why|how|what|when|where|who)\b',
                r'\b(ever wondered|did you know|guess what)\b',
                r'\b(have you|do you|can you|will you)\b',
            ],
            'revelations': [
                r'\b(secret|truth|reveal|exposed|hidden|nobody knows)\b',
                r'\b(turns out|actually|realize|found out|discover)\b',
                r'\b(nobody tells? you|they don\'?t want|finally)\b',
            ],
            'intensity': [
                r'\b(best|worst|most|least|never|always|only|first|last)\b',
                r'\b(biggest|smallest|fastest|slowest|cheapest|expensive)\b',
                r'\b(perfect|terrible|amazing|horrible|incredible)\b',
            ],
            'engagement': [
                r'\b(watch|look|see|check|listen|follow)\b',
                r'\b(let me show you|i\'?ll show you|here\'?s)\b',
                r'\b(subscribe|like|comment|share)\b',
            ],
            'tutorial': [
                r'\b(how to|tutorial|learn|teach|step|tip|trick|hack)\b',
                r'\b(easy|simple|quick|fast way)\b',
            ],
            'story': [
                r'\b(so i|this happened|story time|one time)\b',
                r'\b(then|after|before|suddenly|finally)\b',
            ],
            'reaction': [
                r'\b(i can\'?t|i don\'?t|i just|i thought)\b',
                r'\b(reaction|respond|reply)\b',
            ],
        }

        # Content type labels for classification
        self.content_types = [
            "tutorial or how-to",
            "product review or unboxing",
            "reaction video",
            "storytelling or vlog",
            "comedy or entertainment",
            "before and after transformation",
            "surprising reveal",
            "educational or informative",
            "motivational or inspirational",
            "controversial or debate"
        ]

        # Visual action labels for CLIP
        self.action_labels = [
            "person talking to camera",
            "person reacting with surprise",
            "showing a product",
            "demonstrating something",
            "eating or tasting food",
            "dancing or moving",
            "cooking or preparing food",
            "outdoor scenery",
            "before and after comparison",
            "text or graphics on screen"
        ]

        print("✓ All AI models loaded successfully!\n")

    def download_video(self, url: str, output_path: str = "temp_video.mp4") -> Tuple[str, str]:
        import yt_dlp

        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }

        print(f"📥 Downloading video...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'video')

        print(f"✓ Downloaded: {title}\n")
        return output_path, title

    def analyze_visual_scene(self, frame: np.ndarray) -> Dict:
        """Deep visual understanding using CLIP and BLIP"""
        if not USE_TRANSFORMERS:
            return {'scene': 'unknown', 'actions': [], 'objects': []}

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Generate caption with BLIP
        inputs = self.blip_processor(pil_image, return_tensors="pt")
        if DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

        # Classify actions with CLIP
        inputs = self.clip_processor(
            text=self.action_labels,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        if DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0]

        # Get top actions
        top_actions = []
        for idx in torch.topk(probs, k=3).indices:
            if probs[idx] > 0.15:  # Confidence threshold
                top_actions.append(self.action_labels[idx])

        # Detect objects with YOLO
        objects = []
        if USE_YOLO:
            results = self.object_detector(frame, verbose=False)
            if len(results) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        obj_name = results[0].names[class_id]
                        objects.append(obj_name)

        return {
            'scene': caption,
            'actions': top_actions,
            'objects': list(set(objects))
        }

    def analyze_speech_deep(self, text: str) -> Dict:
        """Advanced NLP analysis of speech"""
        if not text or not USE_TRANSFORMERS:
            return {
                'sentiment': 'neutral',
                'emotion': 'neutral',
                'entities': [],
                'viral_triggers': [],
                'content_type': 'unknown'
            }

        # Sentiment
        sentiment_result = self.sentiment_analyzer(text[:512])[0]
        sentiment = sentiment_result['label'].lower()

        # Emotion
        emotion_results = self.emotion_analyzer(text[:512])[0]
        emotion = max(emotion_results, key=lambda x: x['score'])['label']

        # Extract entities (simple regex for common patterns)
        entities = []
        # Prices
        prices = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        entities.extend(prices)
        # Numbers
        numbers = re.findall(r'\b\d+(?:,\d{3})*\b', text)
        entities.extend(numbers[:3])  # Limit to first 3

        # Viral pattern matching
        viral_triggers = []
        for category, patterns in self.viral_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    viral_triggers.append(category)
                    break

        # Content type classification
        if len(text) > 10:
            content_result = self.content_classifier(
                text[:512],
                candidate_labels=self.content_types,
                multi_label=False
            )
            content_type = content_result['labels'][0]
        else:
            content_type = 'unknown'

        return {
            'sentiment': sentiment,
            'emotion': emotion,
            'entities': entities,
            'viral_triggers': list(set(viral_triggers)),
            'content_type': content_type
        }

    def analyze_video_contextual(self, video_path: str) -> Dict:
        """
        Deep contextual analysis of entire video
        """
        print("🧠 Performing deep contextual analysis...")
        print("=" * 80)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        # Sample frames strategically (beginning, middle, end, peaks)
        sample_rate = max(1, int(fps * 2))  # Every 2 seconds

        motion_scores = []
        visual_contexts = []
        scene_changes = []
        face_presence = []
        prev_frame = None
        frame_idx = 0

        print("  📹 Analyzing visual content with AI...")
        analyzed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps

                # Motion analysis
                gray = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    motion_score = np.mean(diff)
                    motion_scores.append(motion_score)

                    if motion_score > 45:
                        scene_changes.append(timestamp)

                prev_frame = gray

                # Deep visual analysis (sample less frequently for speed)
                if frame_idx % (sample_rate * 3) == 0 and USE_TRANSFORMERS:
                    visual_context = self.analyze_visual_scene(frame)
                    visual_contexts.append({
                        'timestamp': timestamp,
                        **visual_context
                    })
                    analyzed_frames += 1

                # Face detection
                if USE_YOLO:
                    results = self.face_detector(frame, verbose=False)
                    has_face = len(results) > 0 and len(results[0].boxes) > 0
                    face_presence.append(1 if has_face else 0)
                else:
                    face_presence.append(0)

            frame_idx += 1

        cap.release()
        print(f"    ✓ Analyzed {analyzed_frames} frames with AI vision")

        # Audio analysis
        print("  🔊 Analyzing audio with signal processing...")
        try:
            y, sr = librosa.load(video_path, sr=22050, duration=duration, mono=True)

            energy = librosa.feature.rms(y=y, hop_length=512)[0]
            energy_peaks, _ = find_peaks(energy, prominence=np.std(energy) * 0.5)
            energy_peak_times = librosa.frames_to_time(energy_peaks, sr=sr, hop_length=512)

            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]

            # Detect music vs speech
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]

            # Music detection (higher spectral complexity)
            music_indicators = (spectral_rolloff > np.percentile(spectral_rolloff, 70))

            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)

            # Silence detection
            silence_breaks = []
            silence_threshold = np.percentile(energy, 20)
            for i in range(1, len(energy)):
                if energy[i-1] < silence_threshold and energy[i] > np.percentile(energy, 60):
                    silence_breaks.append(librosa.frames_to_time(i, sr=sr, hop_length=512))

            # Interpolate
            time_steps = len(motion_scores)
            if len(energy) > 0 and time_steps > 0:
                audio_energy = np.interp(
                    np.linspace(0, len(energy) - 1, time_steps),
                    np.arange(len(energy)),
                    energy
                )
                audio_spectral = np.interp(
                    np.linspace(0, len(spectral_centroid) - 1, time_steps),
                    np.arange(len(spectral_centroid)),
                    spectral_centroid
                )
            else:
                audio_energy = np.zeros(time_steps)
                audio_spectral = np.zeros(time_steps)

            print(f"    ✓ Found {len(energy_peak_times)} audio peaks")
            print(f"    ✓ Found {len(silence_breaks)} dramatic pauses")

        except Exception as e:
            print(f"    ⚠  Audio analysis limited: {e}")
            audio_energy = np.zeros(len(motion_scores))
            audio_spectral = np.zeros(len(motion_scores))
            energy_peak_times = []
            silence_breaks = []
            beat_times = []
            music_indicators = np.zeros(100)

        # Speech transcription and NLP
        print("  🗣  Transcribing and analyzing speech with NLP...")
        transcript_segments = []
        contextual_moments = []

        try:
            if USE_FASTER_WHISPER:
                segments, _ = self.whisper_model.transcribe(
                    video_path,
                    beam_size=5,
                    word_timestamps=True,
                    language='en'
                )

                for segment in segments:
                    text = segment.text.strip()

                    # Deep NLP analysis
                    speech_analysis = self.analyze_speech_deep(text)

                    # Calculate speech rate
                    duration_seg = segment.end - segment.start
                    word_count = len(text.split())
                    speech_rate = word_count / duration_seg if duration_seg > 0 else 0

                    # Store segment
                    segment_data = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': text,
                        'words': [],
                        **speech_analysis,
                        'speech_rate': speech_rate
                    }

                    if hasattr(segment, 'words'):
                        for word in segment.words:
                            segment_data['words'].append({
                                'text': word.word.strip(),
                                'start': word.start,
                                'end': word.end
                            })

                    transcript_segments.append(segment_data)

                    # Create contextual moment if significant
                    viral_score = (
                            len(speech_analysis['viral_triggers']) * 25 +
                            (100 if speech_analysis['sentiment'] == 'positive' else 50) +
                            abs(speech_rate - 2.5) * 15
                    )

                    if viral_score > 50 or len(speech_analysis['viral_triggers']) > 0:
                        # Find matching visual context
                        visual_ctx = next(
                            (v for v in visual_contexts if abs(v['timestamp'] - segment.start) < 3),
                            {'scene': '', 'actions': [], 'objects': []}
                        )

                        moment = ContextualMoment(
                            timestamp=segment.start,
                            duration=duration_seg,
                            speech_text=text,
                            speech_sentiment=speech_analysis['sentiment'],
                            speech_emotion=speech_analysis['emotion'],
                            entities=speech_analysis['entities'],
                            visual_scene=visual_ctx['scene'],
                            visual_objects=visual_ctx['objects'],
                            visual_actions=visual_ctx['actions'],
                            viral_score=viral_score,
                            viral_triggers=speech_analysis['viral_triggers'],
                            content_type=speech_analysis['content_type']
                        )
                        contextual_moments.append(moment)

            else:
                result = self.whisper_model.transcribe(video_path, word_timestamps=True)
                for segment in result['segments']:
                    text = segment['text'].strip()
                    speech_analysis = self.analyze_speech_deep(text)

                    duration_seg = segment['end'] - segment['start']
                    word_count = len(text.split())
                    speech_rate = word_count / duration_seg if duration_seg > 0 else 0

                    segment_data = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text,
                        'words': segment.get('words', []),
                        **speech_analysis,
                        'speech_rate': speech_rate
                    }
                    transcript_segments.append(segment_data)

                    viral_score = (
                            len(speech_analysis['viral_triggers']) * 25 +
                            (100 if speech_analysis['sentiment'] == 'positive' else 50) +
                            abs(speech_rate - 2.5) * 15
                    )

                    if viral_score > 50 or len(speech_analysis['viral_triggers']) > 0:
                        visual_ctx = next(
                            (v for v in visual_contexts if abs(v['timestamp'] - segment['start']) < 3),
                            {'scene': '', 'actions': [], 'objects': []}
                        )

                        moment = ContextualMoment(
                            timestamp=segment['start'],
                            duration=duration_seg,
                            speech_text=text,
                            speech_sentiment=speech_analysis['sentiment'],
                            speech_emotion=speech_analysis['emotion'],
                            entities=speech_analysis['entities'],
                            visual_scene=visual_ctx['scene'],
                            visual_objects=visual_ctx['objects'],
                            visual_actions=visual_ctx['actions'],
                            viral_score=viral_score,
                            viral_triggers=speech_analysis['viral_triggers'],
                            content_type=speech_analysis['content_type']
                        )
                        contextual_moments.append(moment)

            print(f"    ✓ Transcribed {len(transcript_segments)} segments")
            print(f"    ✓ Identified {len(contextual_moments)} contextual viral moments")

        except Exception as e:
            print(f"    ⚠  Transcription limited: {e}")

        # Analyze overall content
        all_text = " ".join([seg['text'] for seg in transcript_segments])
        overall_content = self.analyze_speech_deep(all_text) if all_text else {}

        # Determine dominant visual themes
        all_objects = []
        all_actions = []
        for ctx in visual_contexts:
            all_objects.extend(ctx['objects'])
            all_actions.extend(ctx['actions'])

        object_counts = Counter(all_objects)
        action_counts = Counter(all_actions)

        print("\n  📊 Content Analysis Summary:")
        if overall_content.get('content_type'):
            print(f"    • Type: {overall_content['content_type']}")
        if object_counts:
            print(f"    • Common objects: {', '.join([obj for obj, _ in object_counts.most_common(3)])}")
        if action_counts:
            print(f"    • Common actions: {', '.join([act for act, _ in action_counts.most_common(2)])}")

        print("=" * 80 + "\n")

        return {
            'duration': duration,
            'fps': fps,
            'motion_scores': motion_scores,
            'audio_energy': audio_energy,
            'audio_spectral': audio_spectral,
            'face_presence': face_presence,
            'scene_changes': scene_changes,
            'energy_peaks': energy_peak_times,
            'silence_breaks': silence_breaks,
            'beat_times': beat_times,
            'music_present': np.mean(music_indicators) > 0.5,
            'transcript_segments': transcript_segments,
            'contextual_moments': contextual_moments,
            'visual_contexts': visual_contexts,
            'overall_content_type': overall_content.get('content_type', 'unknown'),
            'dominant_objects': [obj for obj, _ in object_counts.most_common(5)],
            'dominant_actions': [act for act, _ in action_counts.most_common(3)],
        }

    def calculate_contextual_engagement(self, features: Dict) -> np.ndarray:
        """
        Advanced engagement scoring with contextual understanding
        """
        print("📊 Calculating contextual engagement scores...")

        motion = np.array(features['motion_scores'])
        audio = np.array(features['audio_energy'])
        spectral = np.array(features['audio_spectral'])
        faces = np.array(features['face_presence'])

        min_len = min(len(motion), len(audio), len(spectral), len(faces))
        motion = motion[:min_len]
        audio = audio[:min_len]
        spectral = spectral[:min_len]
        faces = faces[:min_len]

        # Normalize
        scaler = StandardScaler()
        motion_norm = scaler.fit_transform(motion.reshape(-1, 1)).flatten()
        audio_norm = scaler.fit_transform(audio.reshape(-1, 1)).flatten()
        spectral_norm = scaler.fit_transform(spectral.reshape(-1, 1)).flatten()

        # BASE ENGAGEMENT (40% of score)
        base_engagement = (
                0.15 * motion_norm +
                0.25 * audio_norm +
                0.15 * spectral_norm +
                0.15 * faces +
                0.30 * (motion_norm * audio_norm)  # Audio-visual sync
        )

        # CONTEXTUAL BOOSTERS (60% of score)
        contextual_boost = np.zeros_like(base_engagement)

        # 1. Boost contextual moments (very high weight)
        for moment in features['contextual_moments']:
            idx = int(moment.timestamp)
            if idx < len(contextual_boost):
                # Stronger boost for better content types
                content_multiplier = 1.5 if any(x in moment.content_type.lower() for x in [
                    'reveal', 'tutorial', 'review', 'reaction'
                ]) else 1.0

                # Boost based on viral score and triggers
                boost_amount = (moment.viral_score / 100) * content_multiplier
                contextual_boost[idx:min(idx + 5, len(contextual_boost))] += boost_amount

        # 2. Boost speech segments with good sentiment
        for segment in features['transcript_segments']:
            start_idx = int(segment['start'])
            end_idx = int(segment['end'])
            if start_idx < len(contextual_boost):
                sentiment_boost = 0.4 if segment.get('sentiment') == 'positive' else 0.2
                emotion_boost = 0.3 if segment.get('emotion') in ['joy', 'surprise'] else 0.1
                contextual_boost[start_idx:min(end_idx, len(contextual_boost))] += (sentiment_boost + emotion_boost)

        # 3. Boost scene changes (fresh content)
        for change_time in features['scene_changes']:
            idx = int(change_time)
            if idx < len(contextual_boost):
                contextual_boost[idx:min(idx + 2, len(contextual_boost))] += 0.3

        # 4. Boost audio peaks
        for peak_time in features['energy_peaks']:
            idx = int(peak_time)
            if idx < len(contextual_boost):
                contextual_boost[idx:min(idx + 3, len(contextual_boost))] += 0.4

        # 5. Boost silence breaks (dramatic)
        for break_time in features['silence_breaks']:
            idx = int(break_time)
            if idx < len(contextual_boost):
                contextual_boost[idx:min(idx + 4, len(contextual_boost))] += 0.5

        # Combine base and contextual
        engagement = 0.4 * base_engagement + 0.6 * contextual_boost

        # Smooth
        window = 7
        engagement = np.convolve(engagement, np.ones(window)/window, mode='same')

        # Normalize to 0-100
        engagement = (engagement - engagement.min()) / (engagement.max() - engagement.min() + 1e-6) * 100

        return engagement

    def find_optimal_clips(self, engagement: np.ndarray, features: Dict,
                           n_clips: int = 3) -> List[Tuple[int, int, float, Dict]]:
        """
        Find clips with maximum contextual coherence and viral potential
        """
        print("🎯 Finding optimal contextual clips...")
        print("=" * 80)

        clip_len = self.clip_duration
        candidates = []

        for start in range(0, max(1, len(engagement) - clip_len), 3):
            end = min(start + clip_len, len(engagement))

            # ENGAGEMENT METRICS
            avg_engagement = np.mean(engagement[start:end])
            peak_engagement = np.max(engagement[start:end])
            consistency = 1.0 - (np.std(engagement[start:end]) / 100)

            # CONTEXTUAL ANALYSIS
            # Find moments in this clip
            clip_moments = [
                m for m in features['contextual_moments']
                if start <= m.timestamp < end
            ]

            # Aggregate viral triggers
            all_triggers = []
            all_content_types = []
            all_sentiments = []
            all_emotions = []
            entities_found = []

            for moment in clip_moments:
                all_triggers.extend(moment.viral_triggers)
                if moment.content_type:
                    all_content_types.append(moment.content_type)
                if moment.speech_sentiment:
                    all_sentiments.append(moment.speech_sentiment)
                if moment.speech_emotion:
                    all_emotions.append(moment.speech_emotion)
                entities_found.extend(moment.entities)

            # Determine clip content type (most common)
            content_type = max(set(all_content_types), key=all_content_types.count) if all_content_types else "unknown"
            dominant_sentiment = max(set(all_sentiments), key=all_sentiments.count) if all_sentiments else "neutral"
            dominant_emotion = max(set(all_emotions), key=all_emotions.count) if all_emotions else "neutral"

            # Count key indicators
            viral_moment_count = len(clip_moments)
            unique_triggers = len(set(all_triggers))
            scene_changes = sum(1 for t in features['scene_changes'] if start <= t < end)

            # SPEECH COVERAGE
            speech_coverage = 0
            complete_sentences = 0
            for seg in features['transcript_segments']:
                overlap_start = max(start, seg['start'])
                overlap_end = min(end, seg['end'])
                if overlap_start < overlap_end:
                    speech_coverage += (overlap_end - overlap_start)
                    # Check if segment is fully contained (complete thought)
                    if seg['start'] >= start and seg['end'] <= end:
                        complete_sentences += 1

            speech_ratio = speech_coverage / (end - start)

            # NARRATIVE STRUCTURE
            # Check for story arc (building engagement)
            third = (end - start) // 3
            first_third_eng = np.mean(engagement[start:start + third])
            middle_third_eng = np.mean(engagement[start + third:start + 2*third])
            final_third_eng = np.mean(engagement[start + 2*third:end])

            has_build = (middle_third_eng > first_third_eng * 1.1)
            has_climax = (final_third_eng > middle_third_eng * 1.1)
            has_arc = has_build and has_climax

            # STANDALONE QUALITY
            # Can this clip stand alone without context?
            standalone_score = 0.0

            # Has introduction (speech at beginning)
            has_intro = any(seg['start'] < start + 3 for seg in features['transcript_segments'] if seg['start'] >= start)
            standalone_score += 20 if has_intro else 0

            # Has clear content type
            if content_type != "unknown":
                standalone_score += 25

            # Has complete sentences
            standalone_score += min(complete_sentences * 5, 30)

            # Not cut off mid-sentence
            last_seg_in_clip = [seg for seg in features['transcript_segments']
                                if seg['start'] < end and seg['end'] > end]
            if not last_seg_in_clip or (last_seg_in_clip and end - last_seg_in_clip[0]['start'] > 3):
                standalone_score += 25

            # VIRAL POTENTIAL SCORE
            viral_potential = (
                    unique_triggers * 12 +
                    viral_moment_count * 15 +
                    (25 if dominant_sentiment == "positive" else 10) +
                    (20 if dominant_emotion in ["joy", "surprise"] else 5) +
                    (15 if content_type in ["surprising reveal", "tutorial or how-to", "product review"] else 0)
            )

            # COMPOSITE SCORE
            score = (
                    0.20 * avg_engagement +
                    0.15 * peak_engagement +
                    0.10 * consistency * 100 +
                    0.15 * viral_potential +
                    0.15 * speech_ratio * 100 +
                    0.10 * standalone_score +
                    0.10 * (50 if has_arc else 0) +
                    0.05 * scene_changes * 8
            )

            # Penalty for no speech (probably not interesting)
            if speech_ratio < 0.3:
                score *= 0.5

            metadata = {
                'content_type': content_type,
                'viral_moments': viral_moment_count,
                'viral_triggers': list(set(all_triggers)),
                'sentiment': dominant_sentiment,
                'emotion': dominant_emotion,
                'speech_ratio': speech_ratio,
                'standalone_score': standalone_score,
                'has_arc': has_arc,
                'entities': entities_found[:5],
                'avg_engagement': avg_engagement,
                'peak_engagement': peak_engagement,
                'viral_potential': viral_potential
            }

            candidates.append((start, end, score, metadata))

        # Sort by score
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Select non-overlapping clips with diverse content
        selected = []
        used_content_types = set()

        for clip in candidates:
            start, end, score, metadata = clip

            # Check overlap
            overlap = any(
                not (end <= s[0] + 5 or start >= s[1] - 5)
                for s in selected
            )

            # Prefer diverse content types
            content_diversity_bonus = 0
            if metadata['content_type'] not in used_content_types:
                content_diversity_bonus = 10

            adjusted_score = score + content_diversity_bonus

            if not overlap and metadata['standalone_score'] > 40:  # Quality threshold
                selected.append((start, end, adjusted_score, metadata))
                used_content_types.add(metadata['content_type'])

                print(f"✓ Clip {len(selected)}: {start}-{end}s | Score: {adjusted_score:.1f}")
                print(f"  • Type: {metadata['content_type']}")
                print(f"  • Viral triggers: {', '.join(metadata['viral_triggers'][:3]) if metadata['viral_triggers'] else 'None'}")
                print(f"  • Sentiment: {metadata['sentiment']} | Emotion: {metadata['emotion']}")
                print(f"  • Speech: {metadata['speech_ratio']*100:.0f}% | Standalone: {metadata['standalone_score']:.0f}/100")
                print(f"  • Story arc: {'Yes ✓' if metadata['has_arc'] else 'No'}")
                if metadata['entities']:
                    print(f"  • Key entities: {', '.join(metadata['entities'][:3])}")
                print()

            if len(selected) >= n_clips:
                break

        print("=" * 80 + "\n")
        return sorted(selected, key=lambda x: x[0])


class FastCaptionGenerator:
    """Optimized caption generation"""

    def _init_(self):
        if USE_FASTER_WHISPER:
            self.whisper_model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE_TYPE)
        else:
            self.whisper_model = whisper.load_model("base", device=DEVICE)

        self.caption_style = {
            'font_size': 85,
            'color': '#FFFFFF',
            'outline_color': '#000000',
            'outline_width': 10,
            'position': 420
        }

    def generate_captions(self, video_path: str, clip_start: int, clip_end: int) -> List[Dict]:
        print("💬 Generating captions...")

        try:
            words = []

            if USE_FASTER_WHISPER:
                segments, _ = self.whisper_model.transcribe(
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

            return self._group_words(words, max_words=2, max_duration=1.5)
        except:
            return []

    def _group_words(self, words: List[Dict], max_words: int = 2,
                     max_duration: float = 1.5) -> List[Dict]:
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

    def create_short_optimized(self, video_clip: VideoFileClip, captions: List[Dict],
                               output_path: str):
        print(f"🎬 Creating Short...")

        orig_w, orig_h = video_clip.size
        target_w, target_h = 1080, 1920

        scale = min(target_w / orig_w, target_h / orig_h)
        video_resized = video_clip.resize(scale)
        new_w, new_h = video_resized.size

        blur_scale = max(target_w / orig_w, target_h / orig_h)
        video_background = video_clip.resize(blur_scale)

        def make_blurred_bg(get_frame, t):
            frame = get_frame(t)
            bg_h, bg_w = frame.shape[:2]
            x_crop = (bg_w - target_w) // 2
            y_crop = (bg_h - target_h) // 2
            cropped = frame[y_crop:y_crop + target_h, x_crop:x_crop + target_w]
            blurred = cv2.GaussianBlur(cropped, (51, 51), 30)
            blurred = (blurred * 0.35).astype(np.uint8)
            return blurred

        background_clip = VideoClip(
            make_frame=lambda t: make_blurred_bg(video_background.get_frame, t),
            duration=video_clip.duration
        ).set_fps(30)

        x_pos = (target_w - new_w) // 2
        y_pos = (target_h - new_h) // 2
        video_resized = video_resized.set_position((x_pos, y_pos))

        base_composite = CompositeVideoClip(
            [background_clip, video_resized],
            size=(target_w, target_h)
        ).set_audio(video_clip.audio)

        font = self._load_font(self.caption_style['font_size'])
        text_clips = []

        for caption in captions:
            text = caption['text'].upper()

            img = Image.new('RGBA', (target_w, 300), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (target_w - text_width) // 2
            y = 50

            outline_w = self.caption_style['outline_width']
            for dx in range(-outline_w, outline_w + 1, 2):
                for dy in range(-outline_w, outline_w + 1, 2):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font,
                                  fill=self.caption_style['outline_color'])

            draw.text((x, y), text, font=font, fill=self.caption_style['color'])

            duration = caption['end'] - caption['start']
            caption_clip = ImageClip(np.array(img), duration=duration, transparent=True)
            caption_clip = caption_clip.set_start(caption['start']).set_position(
                ('center', self.caption_style['position'])
            )
            text_clips.append(caption_clip)

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
                preset='faster',
                bitrate='5000k',
                audio_bitrate='192k',
                threads=4,
                logger=None
            )
            print(f"✅ Saved: {output_path}\n")
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
            gc.collect()

    def _load_font(self, size: int):
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

        body = {
            'snippet': {
                'title': title[:100],
                'description': description[:5000],
                'tags': tags + ['shorts', 'youtubeshorts', 'viral'],
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

        print(f"✅ Uploaded! https://youtube.com/shorts/{video_id}\n")
        return video_id


def main(video_url: str, n_clips: int = 3, clip_duration: int = 60,
         upload_to_youtube: bool = False):

    print("\n" + "=" * 80)
    print("🧠 ULTIMATE AI-POWERED SHORTS CREATOR")
    print("=" * 80 + "\n")

    analyzer = AdvancedAIAnalyzer(clip_duration=clip_duration)
    caption_gen = FastCaptionGenerator()
    uploader = YouTubeUploader() if upload_to_youtube else None

    # Download
    video_path, video_title = analyzer.download_video(video_url)

    # Deep contextual analysis
    features = analyzer.analyze_video_contextual(video_path)

    # Calculate engagement with context
    engagement = analyzer.calculate_contextual_engagement(features)

    # Find optimal clips
    best_clips = analyzer.find_optimal_clips(engagement, features, n_clips=n_clips)

    if not best_clips:
        print("❌ No suitable clips found. Try a different video or adjust duration.")
        return

    print(f"🎬 Creating {len(best_clips)} Shorts...\n")
    print("=" * 80 + "\n")

    created_shorts = []

    for i, (start, end, score, metadata) in enumerate(best_clips):
        print(f"{'='*80}")
        print(f"PROCESSING CLIP {i+1}/{len(best_clips)}")
        print(f"{'='*80}")
        print(f"  Score: {score:.1f}/100")
        print(f"  Time: {start}s - {end}s ({end-start}s)")
        print(f"  Content Type: {metadata['content_type']}")
        print(f"  Viral Potential: {metadata['viral_potential']:.0f}/100")
        print(f"  Standalone Quality: {metadata['standalone_score']:.0f}/100")
        print(f"{'='*80}\n")

        # Extract clip
        clip_path = f"clip_{i+1}_temp.mp4"
        video = VideoFileClip(video_path).subclip(start, end)
        video.write_videofile(clip_path, codec='libx264', audio_codec='aac',
                              logger=None, preset='ultrafast')
        video.close()

        # Generate captions
        captions = caption_gen.generate_captions(video_path, start, end)

        # Create final short
        output_path = f"short_{i+1}_final.mp4"
        video_clip = VideoFileClip(clip_path)
        caption_gen.create_short_optimized(video_clip, captions, output_path)
        video_clip.close()

        created_shorts.append({
            'path': output_path,
            'metadata': metadata,
            'score': score
        })

        # Upload if requested
        if uploader:
            triggers_str = ', '.join(metadata['viral_triggers'][:3]) if metadata['viral_triggers'] else 'engaging'
            entities_str = ', '.join(metadata['entities'][:2]) if metadata['entities'] else ''

            title = f"{video_title[:40]} - {metadata['content_type'][:20]} #{i+1}"
            description = (
                f"🔥 AI-Selected Viral Moment\n\n"
                f"📊 Analytics:\n"
                f"• Viral Score: {score:.1f}/100\n"
                f"• Content Type: {metadata['content_type']}\n"
                f"• Sentiment: {metadata['sentiment'].title()}\n"
                f"• Triggers: {triggers_str}\n"
            )

            if entities_str:
                description += f"• Key Points: {entities_str}\n"

            description += f"\n#Shorts #Viral #Trending #{metadata['emotion'].title()}"

            tags = ['viral', 'trending', 'ai', 'shorts'] + metadata['viral_triggers'][:3]
            uploader.upload_video(output_path, title, description, tags)

        # Cleanup
        try:
            os.remove(clip_path)
        except:
            pass

        gc.collect()

    # Cleanup
    try:
        os.remove(video_path)
    except:
        pass

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 ALL SHORTS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nTotal Shorts: {len(created_shorts)}")
    print("\nRanked by Viral Potential:\n")

    ranked = sorted(created_shorts, key=lambda x: x['score'], reverse=True)
    for idx, short in enumerate(ranked, 1):
        size_mb = os.path.getsize(short['path']) / (1024*1024)
        print(f"{idx}. {short['path']} ({size_mb:.1f} MB)")
        print(f"   Score: {short['score']:.1f} | Type: {short['metadata']['content_type']}")
        print(f"   Viral: {short['metadata']['viral_potential']:.0f} | Standalone: {short['metadata']['standalone_score']:.0f}")
        if short['metadata']['viral_triggers']:
            print(f"   Triggers: {', '.join(short['metadata']['viral_triggers'][:3])}")
        print()

    print("=" * 80)
    print("\n💡 RECOMMENDATION: Upload clips in this order for maximum impact!")
    print("=" * 80 + "\n")


if _name_ == "_main_":
    print("\n" + "=" * 80)
    print("🧠 ULTIMATE AI-POWERED SHORTS CREATOR")
    print("=" * 80)
    print("\n🚀 ADVANCED AI FEATURES:")
    print("  ✓ CLIP (Visual-Language Understanding)")
    print("  ✓ BLIP (Image Captioning & Scene Description)")
    print("  ✓ Sentiment Analysis (Emotional Context)")
    print("  ✓ Emotion Detection (Joy, Surprise, Anger, etc.)")
    print("  ✓ Content Type Classification (10+ categories)")
    print("  ✓ Entity Recognition (Prices, Numbers, Products)")
    print("  ✓ Narrative Arc Detection (Setup→Build→Climax)")
    print("  ✓ Standalone Quality Scoring")
    print("  ✓ Multi-Modal Fusion (Video+Audio+Speech)")
    print("  ✓ Context-Aware Viral Detection")
    print("=" * 80)

    VIDEO_URL = input("\n🎥 YouTube URL: ").strip()
    NUM_CLIPS = int(input("📊 Number of Shorts (1-5, default 3): ") or "3")
    CLIP_DURATION = int(input("⏱  Duration per Short (30-60s, default 60): ") or "60")

    upload_choice = input("📤 Upload to YouTube? (yes/no, default no): ").lower()
    UPLOAD = upload_choice == 'yes'

    print(f"\n{'='*80}")
    print("⚙  CONFIGURATION:")
    print(f"  • Device: {DEVICE.upper()}")
    print(f"  • AI Models: CLIP + BLIP + Whisper + Transformers")
    print(f"  • Clips: {NUM_CLIPS} x {CLIP_DURATION}s")
    print(f"  • Analysis: Deep Contextual Understanding")
    print(f"  • Upload: {'YES' if UPLOAD else 'NO'}")
    print('='*80 + "\n")

    try:
        main(VIDEO_URL, n_clips=NUM_CLIPS, clip_duration=CLIP_DURATION,
             upload_to_youtube=UPLOAD)
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()