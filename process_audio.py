import torch
import numpy as np
from denoiser import pretrained
from pydub import AudioSegment
from scipy.io import wavfile
import os

def load_audio_flexible(path):
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= (1 << (8 * audio.sample_width - 1))
    return torch.from_numpy(samples).unsqueeze(0), 16000

def bersihkan_kajian(input_path):
    print(f"--- Memulai Proses Pembersihan AI ---")
    
    print("Mengambil model AI...")
    model = pretrained.dns64().cpu()
    model.eval()

    print(f"Membaca file: {input_path}")
    try:
        out, sr = load_audio_flexible(input_path)
    except Exception as e:
        print(f"Error baca file: {e}. Pastikan FFmpeg sudah di PATH.")
        return

    print("Sedang memproses... (Mohon tunggu)")
    with torch.no_grad():
        out_batch = out.unsqueeze(0)
        denoised_batch = model(out_batch)
        denoised = denoised_batch.squeeze().cpu().numpy()
    
    nama_output = f"HASIL_BERSIH_{os.path.basename(input_path)}.wav"

    denoised_int = (denoised * 32767).astype(np.int16)
    wavfile.write(nama_output, sr, denoised_int)
    
    print(f"--- SELESAI ---")
    print(f"File jernih disimpan di: {os.path.abspath(nama_output)}")

if __name__ == "__main__":
    file_target = "kajian_original.mp3" 
    if os.path.exists(file_target):
        bersihkan_kajian(file_target)
    else:
        print(f"File {file_target} tidak ditemukan!")