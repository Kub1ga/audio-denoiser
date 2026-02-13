import torch
import numpy as np
from denoiser import pretrained
from pydub import AudioSegment
from scipy.io import wavfile
import os
import ssl

# Bypass SSL Error (Khusus macOS/Windows yang rewel)
ssl._create_default_https_context = ssl._create_unverified_context

def load_audio(path, target_sr=16000):
    print(f"📂 Membaca file: {os.path.basename(path)}...")
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    # Konversi ke array float32
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # Normalisasi ke range -1.0 s/d 1.0 (Bit depth handling)
    max_val = float(1 << (8 * audio.sample_width - 1))
    samples /= max_val
    return torch.from_numpy(samples), target_sr

def bersihkan_dengan_gpu(input_path):
    print("\n--- 🚀 MEMULAI PROSES AI (MODE GPU) ---")
    
    # 1. Cek GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Ditemukan: {torch.cuda.get_device_name(0)}")
        print("   (Mode Turbo Aktif!)")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU tidak terdeteksi. Menggunakan CPU (Mode Lambat).")

    # 2. Load Model ke GPU
    print("⏳ Sedang memuat model ke VRAM...")
    model = pretrained.dns64().to(device)
    model.eval()

    # 3. Baca Audio
    raw_audio, sr = load_audio(input_path)
    total_length = len(raw_audio)
    
    # 4. Strategi Chunking (Penting buat RTX 2050 4GB)
    # Kita potong per 30 detik (16000 * 30 = 480,000 samples)
    chunk_size = 16000 * 30 
    processed_chunks = []
    
    print(f"✂️ Memproses audio per 30 detik (Total: {total_length / sr / 60:.1f} menit)")

    with torch.no_grad():
        for i in range(0, total_length, chunk_size):
            # Ambil potongan
            end = min(i + chunk_size, total_length)
            chunk = raw_audio[i:end]
            
            # Pindah ke GPU & Tambah Dimensi [Batch, Channel, Time]
            chunk_tensor = chunk.unsqueeze(0).unsqueeze(0).to(device)
            
            # PROSES AI DISINI
            out_tensor = model(chunk_tensor)
            
            # Balikin ke CPU & Simpan
            processed_chunks.append(out_tensor.squeeze().cpu().numpy())
            
            # Progress Bar Sederhana
            persen = (end / total_length) * 100
            print(f"\r   Proses: {persen:.1f}% selesai...", end="")

    print("\n\n💾 Menggabungkan & Menyimpan file...")
    # Gabung semua potongan
    final_audio = np.concatenate(processed_chunks)
    
    # Simpan ke WAV
    nama_output = f"BERSIH_GPU_{os.path.basename(input_path)}.wav"
    folder_output = os.path.dirname(input_path)
    full_output_path = os.path.join(folder_output, nama_output)
    
    # Konversi balik ke format 16-bit PCM (Standar WAV)
    final_int16 = (final_audio * 32767).astype(np.int16)
    wavfile.write(full_output_path, sr, final_int16)
    
    print(f"✅ SELESAI! File disimpan di:\n   {full_output_path}")

if __name__ == "__main__":
    # Ganti ini dengan path file kajian kamu yang panjang
    file_kajian = r"C:\Path\Ke\File\Kajian_Panjang.mp3"
    
    if os.path.exists(file_kajian):
        bersihkan_dengan_gpu(file_kajian)
    else:
        print("❌ File tidak ditemukan. Cek path-nya lagi.")