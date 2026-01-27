import os
import json
import base64
import httpx
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Prompt AI")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi API Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

SYSTEM_PROMPT = """Anda adalah ahli produksi video profesional. Analisis video yang diberikan secara objektif, hanya berdasarkan apa yang terlihat dan terdengar. JANGAN mengasumsikan, menebak, atau mengarang informasi yang tidak teramati langsung.

Ikuti struktur JSON berikut secara ketat. Untuk setiap field:
- Jika elemen TIDAK TERLIHAT atau TIDAK DAPAT DIKONFIRMASI dari video, isi dengan string kosong "" (untuk teks) atau array kosong [] (untuk list).
- Durasi harus dihitung dari timestamp aktual.
- Semua timestamp dalam format "MM:SS".
- Gunakan bahasa Indonesia untuk semua nilai teks.

Struktur output wajib:
{
  "video_title": "",
  "duration_seconds": 0,
  "scene_description": "",
  "key_elements": [],
  "mood_atmosphere": "",
  "camera_movement": "",
  "audio_suggestions": [],
  "branding_watermark": "",
  "target_audience": [],
  "tags": [],
  "video_analysis": {
    "basic_info": { "title": "", "source_url": "", "duration_seconds": 0, "format": "", "platform": "", "primary_objective": "", "core_message": "" },
    "audience_analysis": { "primary": [], "secondary": [], "pain_points_addressed": [], "value_provided": [] },
    "scene_breakdown": [ { "timestamp": "MM:SS", "description": "", "visual_elements": [], "audio_elements": "" } ],
    "ai_production_recommendations": [ { "tool_name": "", "use_case": "" } ]
  }
}"""


async def call_gemini_with_retry(video_base64: str, mime_type: str):
    retries = 5
    for i in range(retries):
        try:
            delay = 2 ** i
            if i > 0:
                await asyncio.sleep(delay)

            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": "Analisis video ini sesuai instruksi sistem yang diberikan."},
                            {"inlineData": {"mimeType": mime_type, "data": video_base64}}
                        ]
                    }],
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "generationConfig": {
                        "responseMimeType": "application/json"
                    }
                }

                response = await client.post(GEMINI_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    json_text = result['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(json_text)

        except Exception as e:
            if i == retries - 1:
                raise HTTPException(status_code=500, detail=f"Gagal memproses AI: {str(e)}")

    raise HTTPException(status_code=500, detail="Gagal menghubungi server AI setelah beberapa percobaan.")


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File harus berupa video.")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 20MB.")

    encoded_video = base64.b64encode(content).decode("utf-8")
    result = await call_gemini_with_retry(encoded_video, file.content_type)
    return result


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .font-mono { font-family: 'Fira Code', monospace; }

        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #475569; }

        .loader-spin {
            border: 3px solid rgba(99, 102, 241, 0.1);
            border-top: 3px solid #6366f1;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* Video Controls Styling */
        .video-container:hover .video-controls-overlay { opacity: 1; }
        .video-controls-overlay {
            opacity: 0;
            transition: opacity 0.3s ease;
            background: linear-gradient(transparent, rgba(0,0,0,0.7));
        }
    </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen">

    <div class="max-w-6xl mx-auto p-4 md:p-8">
        <header class="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div>
                <div class="flex items-center gap-2 mb-1 text-indigo-600 font-bold uppercase tracking-widest text-xs">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="15" y1="2" x2="15" y2="4"/><line x1="9" y1="2" x2="9" y2="4"/></svg>
                    Ai Video
                </div>
                <h1 class="text-4xl font-extrabold flex items-center gap-3 text-slate-900">
                    Prompt<span class="text-indigo-600"> AI</span>
                </h1>
                <p class="text-slate-500 mt-2 max-w-lg leading-relaxed">
                    Ubah konten video Anda menjadi prompt terstruktur dan metadata profesional secara otomatis.
                </p>
            </div>
            <div class="flex items-center gap-3 px-4 py-2 bg-white border border-slate-200 shadow-sm rounded-2xl w-fit">
                <div class="p-2 bg-indigo-50 rounded-lg text-indigo-600">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
                </div>
                <div>
                    <p class="text-[10px] text-slate-400 font-bold uppercase">Versi Sistem</p>
                    <p class="text-sm font-semibold text-slate-700">Doaibu 2.5 Flash</p>
                </div>
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <!-- Kolom Kiri: Input -->
            <div class="space-y-6">
                <div id="drop-zone" class="group relative border-2 border-dashed border-slate-300 bg-white rounded-3xl p-8 transition-all flex flex-col items-center justify-center text-center cursor-pointer min-h-[400px] hover:border-indigo-400 hover:shadow-xl hover:shadow-indigo-50">
                    <input type="file" id="file-input" class="hidden" accept="video/*">

                    <!-- Empty State -->
                    <div id="state-empty" class="flex flex-col items-center justify-center w-full">
                        <div class="w-20 h-20 bg-indigo-50 rounded-full flex items-center justify-center mb-6 text-indigo-600 group-hover:scale-110 transition-transform shadow-inner shadow-indigo-100/50">
                            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                        </div>
                        <h3 class="text-xl font-bold text-slate-800">Unggah Video</h3>
                        <p class="text-slate-500 text-sm mt-2 max-w-xs mx-auto">
                            Klik untuk mencari atau seret file Anda ke sini. Maksimal 20MB.
                        </p>
                    </div>

                    <!-- Preview State -->
                    <div id="state-preview" class="hidden w-full h-full flex flex-col">
                        <div class="flex items-center justify-between mb-4 bg-white p-3 rounded-2xl border border-indigo-100 shadow-sm">
                            <div class="flex items-center gap-3 text-left overflow-hidden">
                                <div class="bg-indigo-600 p-2 rounded-lg text-white">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.934a.5.5 0 0 0-.777-.416L16 11"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg>
                                </div>
                                <div class="overflow-hidden">
                                    <p id="file-name" class="font-bold text-slate-800 truncate text-sm">video.mp4</p>
                                    <p id="file-info" class="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">0.00 MB • READY</p>
                                </div>
                            </div>
                            <button id="remove-btn" class="p-2 text-slate-400 hover:text-red-500 transition-colors">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
                            </button>
                        </div>

                        <!-- Video Player Wrapper -->
                        <div id="video-container" class="video-container relative rounded-2xl overflow-hidden shadow-2xl bg-black aspect-video mt-auto mb-auto group">
                            <video id="video-player" class="w-full h-full object-contain"></video>

                            <!-- Video Controls Overlay (Modern Style) -->
                            <div id="video-overlay" class="video-controls-overlay absolute inset-0 flex flex-col justify-end p-4">
                                <!-- Progress Bar (Simple) -->
                                <div class="w-full bg-white/20 h-1 mb-4 rounded-full overflow-hidden">
                                    <div id="video-progress" class="bg-indigo-500 h-full w-0"></div>
                                </div>

                                <div class="flex items-center justify-between text-white">
                                    <!-- Play/Pause & Time -->
                                    <div class="flex items-center gap-4">
                                        <button id="play-pause-btn" class="hover:text-indigo-400 transition-colors">
                                            <svg id="play-svg" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                                            <svg id="pause-svg" class="hidden" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                                        </button>
                                        <span id="video-time" class="text-xs font-mono">00:00 / 00:00</span>
                                    </div>

                                    <!-- YouTube Style Features (Right Side) -->
                                    <div class="flex items-center gap-3">
                                        <!-- Picture in Picture -->
                                        <button id="pip-btn" title="Picture in Picture" class="p-1 hover:text-indigo-400 transition-colors">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 4.5v5H3m-1-6 6 6m13 0v-3c0-1.1-.9-2-2-2h-3m-5 13h7c1.1 0 2-.9 2-2v-7"/><path d="M14 14.5v5H9m-1-6 6 6"/></svg>
                                        </button>
                                        <!-- Fullscreen -->
                                        <button id="fullscreen-btn" title="Fullscreen" class="p-1 hover:text-indigo-400 transition-colors">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <button id="analyze-btn" disabled class="w-full py-5 rounded-2xl font-black text-lg flex items-center justify-center gap-3 transition-all relative overflow-hidden bg-slate-200 text-slate-400 cursor-not-allowed">
                    <svg id="btn-icon" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                    <span id="btn-text">Generate Prompt</span>
                </button>

                <div id="error-box" class="hidden p-4 bg-red-50 border border-red-100 text-red-600 rounded-2xl flex items-start gap-3">
                    <svg class="shrink-0 mt-0.5" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <p id="error-message" class="text-sm font-medium"></p>
                </div>
            </div>

            <!-- Kolom Kanan: Hasil -->
            <div class="flex flex-col min-h-[500px] lg:h-[700px]">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-xl font-bold flex items-center gap-2 text-slate-800">
                        <svg class="text-indigo-600" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><path d="M10 13a2 2 0 1 0 4 0 2 2 0 1 0-4 0Z"/><path d="m20 17-1.09-1.09a2 2 0 0 0-2.82 0L10 22"/></svg>
                        Hasil Prompt
                    </h2>
                    <button id="copy-btn" class="hidden flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider px-4 py-2 rounded-xl bg-white border border-slate-200 hover:bg-slate-50 transition-all shadow-sm active:scale-95">
                        <svg id="copy-icon" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                        <span id="copy-text">Salin JSON</span>
                    </button>
                </div>

                <div class="flex-grow bg-[#0f172a] rounded-[2rem] overflow-hidden relative border border-slate-800 shadow-2xl flex flex-col">
                    <!-- Loading Overlay -->
                    <div id="loading-overlay" class="hidden absolute inset-0 bg-slate-900/80 backdrop-blur-md z-10 flex flex-col items-center justify-center text-white">
                        <div class="relative mb-6">
                            <div class="loader-spin"></div>
                            <svg class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-indigo-400 animate-pulse" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                        </div>
                        <p class="font-bold text-lg text-center px-6">AI Memproses Konten...</p>
                        <p class="text-slate-400 text-sm mt-2">Ini mungkin memakan waktu 10-30 detik.</p>
                        <div class="flex gap-1 mt-3">
                            <span class="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-bounce" style="animation-delay: -0.3s"></span>
                            <span class="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-bounce" style="animation-delay: -0.15s"></span>
                            <span class="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-bounce"></span>
                        </div>
                    </div>

                    <!-- Placeholder -->
                    <div id="result-placeholder" class="flex-grow flex flex-col items-center justify-center text-slate-500 p-12 text-center">
                        <div class="w-20 h-20 border-2 border-slate-800 border-dashed rounded-3xl flex items-center justify-center mb-6">
                            <svg class="text-slate-700" xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                        </div>
                        <h4 class="text-slate-300 font-bold mb-2">Menunggu Data</h4>
                        <p class="text-sm leading-relaxed max-w-xs mx-auto">
                            Setelah video dianalisis via FastAPI, rincian scene akan muncul di sini dalam format JSON.
                        </p>
                    </div>

                    <!-- Result View -->
                    <div id="result-content" class="hidden flex-grow overflow-auto p-6 font-mono text-sm leading-relaxed custom-scrollbar">
                        <pre id="json-output" class="text-indigo-300 whitespace-pre-wrap"></pre>
                    </div>

                    <div class="p-4 bg-slate-900/50 border-t border-slate-800 flex items-center justify-between">
                        <div class="text-[9px] text-slate-500 uppercase tracking-[0.2em] font-black flex items-center gap-2">
                            <span id="status-indicator" class="w-2 h-2 rounded-full bg-slate-700"></span>
                            Sistem FastAPI <span id="status-text">Standby</span>
                        </div>
                        <div id="validation-label" class="hidden text-[9px] text-indigo-400 font-bold uppercase">
                            Struktur Output Tervalidasi
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // DOM Elements
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const stateEmpty = document.getElementById('state-empty');
        const statePreview = document.getElementById('state-preview');
        const fileNameLabel = document.getElementById('file-name');
        const fileInfoLabel = document.getElementById('file-info');
        const videoPlayer = document.getElementById('video-player');
        const videoOverlay = document.getElementById('video-overlay');
        const playPauseBtn = document.getElementById('play-pause-btn');
        const playSvg = document.getElementById('play-svg');
        const pauseSvg = document.getElementById('pause-svg');
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        const pipBtn = document.getElementById('pip-btn');
        const videoTime = document.getElementById('video-time');
        const videoProgress = document.getElementById('video-progress');
        const analyzeBtn = document.getElementById('analyze-btn');
        const btnText = document.getElementById('btn-text');
        const btnIcon = document.getElementById('btn-icon');
        const errorBox = document.getElementById('error-box');
        const errorMessage = document.getElementById('error-message');
        const loadingOverlay = document.getElementById('loading-overlay');
        const resultPlaceholder = document.getElementById('result-placeholder');
        const resultContent = document.getElementById('result-content');
        const jsonOutput = document.getElementById('json-output');
        const copyBtn = document.getElementById('copy-btn');
        const copyText = document.getElementById('copy-text');
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        const validationLabel = document.getElementById('validation-label');
        const removeBtn = document.getElementById('remove-btn');

        let isAnalyzing = false;

        // Listeners
        dropZone.onclick = (e) => { 
            // Only trigger click if not clicking the preview state elements
            if(!isAnalyzing && stateEmpty.contains(e.target)) fileInput.click(); 
        };

        fileInput.onchange = (e) => handleFile(e.target.files[0]);

        function handleFile(file) {
            if (!file || !file.type.startsWith('video/')) {
                showError("Format file tidak didukung. Harap pilih video.");
                return;
            }
            if (file.size > 20 * 1024 * 1024) {
                showError("File terlalu besar (Maks 20MB).");
                return;
            }

            fileNameLabel.innerText = file.name;
            fileInfoLabel.innerText = `${(file.size / (1024 * 1024)).toFixed(2)} MB • READY`;
            videoPlayer.src = URL.createObjectURL(file);

            stateEmpty.classList.add('hidden');
            statePreview.classList.remove('hidden');
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('bg-slate-200', 'text-slate-400', 'cursor-not-allowed');
            analyzeBtn.classList.add('bg-indigo-600', 'text-white', 'hover:bg-indigo-700', 'shadow-xl');
            btnIcon.classList.add('fill-white');
            errorBox.classList.add('hidden');
        }

        // --- Custom Video Player Logic ---
        function togglePlay() {
            if (videoPlayer.paused) {
                videoPlayer.play();
                playSvg.classList.add('hidden');
                pauseSvg.classList.remove('hidden');
            } else {
                videoPlayer.pause();
                playSvg.classList.remove('hidden');
                pauseSvg.classList.add('hidden');
            }
        }

        playPauseBtn.onclick = (e) => {
            e.stopPropagation();
            togglePlay();
        };

        // Toggle play on clicking the video itself
        videoPlayer.onclick = (e) => {
            e.stopPropagation();
            togglePlay();
        };

        // Fullscreen Feature (YouTube style)
        fullscreenBtn.onclick = (e) => {
            e.stopPropagation();
            if (!document.fullscreenElement) {
                if (videoPlayer.parentElement.requestFullscreen) {
                    videoPlayer.parentElement.requestFullscreen();
                } else if (videoPlayer.requestFullscreen) {
                    videoPlayer.requestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                }
            }
        };

        // Picture in Picture
        pipBtn.onclick = async (e) => {
            e.stopPropagation();
            try {
                if (document.pictureInPictureElement) {
                    await document.exitPictureInPicture();
                } else {
                    await videoPlayer.requestPictureInPicture();
                }
            } catch (err) {
                console.error("PiP failed", err);
            }
        };

        // Update Time and Progress
        videoPlayer.ontimeupdate = () => {
            const current = formatTime(videoPlayer.currentTime);
            const duration = formatTime(videoPlayer.duration || 0);
            videoTime.innerText = `${current} / ${duration}`;

            const progress = (videoPlayer.currentTime / videoPlayer.duration) * 100;
            videoProgress.style.width = `${progress}%`;
        };

        function formatTime(seconds) {
            const min = Math.floor(seconds / 60);
            const sec = Math.floor(seconds % 60);
            return `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
        }
        // ---------------------------------

        removeBtn.onclick = (e) => {
            e.stopPropagation();
            location.reload();
        };

        function showError(msg) {
            errorBox.classList.remove('hidden');
            errorMessage.innerText = msg;
        }

        analyzeBtn.onclick = async () => {
            const file = fileInput.files[0];
            if (!file || isAnalyzing) return;

            isAnalyzing = true;
            analyzeBtn.disabled = true;
            btnText.innerText = "Video to Prompt";
            loadingOverlay.classList.remove('hidden');
            errorBox.classList.add('hidden');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.detail || 'Terjadi kesalahan pada server.');
                }

                const data = await response.json();
                jsonOutput.innerText = JSON.stringify(data, null, 2);

                resultPlaceholder.classList.add('hidden');
                resultContent.classList.remove('hidden');
                copyBtn.classList.remove('hidden');
                statusIndicator.classList.replace('bg-slate-700', 'bg-green-500');
                statusText.innerText = 'Selesai';
                validationLabel.classList.remove('hidden');

            } catch (err) {
                showError(err.message);
            } finally {
                isAnalyzing = false;
                analyzeBtn.disabled = false;
                btnText.innerText = "Jalankan Analisis Lagi";
                loadingOverlay.classList.add('hidden');
            }
        };

        copyBtn.onclick = () => {
            const textArea = document.createElement("textarea");
            textArea.value = jsonOutput.innerText;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            copyText.innerText = 'Tersalin';
            setTimeout(() => copyText.innerText = 'Salin JSON', 2000);
        };
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
