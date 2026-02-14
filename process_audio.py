import argparse
import torch
import numpy as np
from denoiser import pretrained
from pydub import AudioSegment
import os
import sys
import ssl
import gc  

ssl._create_default_https_context = ssl._create_unverified_context

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Fungsi untuk membuat loading bar cantik di terminal"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

def bersihkan_kajian_cli(input_path, output_path=None):
    print(f"\n{'='*50}")
    print(f"   AR-RAUDA AI CLEANER")
    print(f"{'='*50}\n")
    
    print("[INIT] Sedang memuat model AI (DNS64)...")
    try:
        model = pretrained.dns64().cpu()
        model.eval()
    except Exception as e:
        print(f"❌ [ERROR] Gagal load model: {e}")
        return

    print(f"[READ] Membaca file: {os.path.basename(input_path)}...")
    try:
        audio_full = AudioSegment.from_file(input_path)
        audio_full = audio_full.set_frame_rate(16000).set_channels(1)
    except Exception as e:
        print(f"[ERROR] Gagal membaca file audio. Pastikan FFmpeg terinstall.\nDetail: {e}")
        return

    chunk_durasi_ms = 30 * 1000 
    total_durasi_ms = len(audio_full)
    hasil_bersih = []
    
    total_menit = total_durasi_ms / 1000 / 60
    print(f"[INFO] Total Durasi: {total_menit:.2f} menit")
    print(f"[PROCESS] Memulai pembersihan bertahap...")

    print_progress_bar(0, total_durasi_ms, prefix='Progress:', suffix='Selesai', length=40)

    with torch.no_grad():
        for i in range(0, total_durasi_ms, chunk_durasi_ms):
            chunk_audio = audio_full[i : i + chunk_durasi_ms]
            
            samples = np.array(chunk_audio.get_array_of_samples()).astype(np.float32)
            max_val = float(1 << (8 * chunk_audio.sample_width - 1))
            samples /= max_val
            
            chunk_tensor = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
            
            out_tensor = model(chunk_tensor)
            
            hasil_bersih.append(out_tensor.squeeze().cpu().numpy())
            
            current_progress = min(i + chunk_durasi_ms, total_durasi_ms)
            print_progress_bar(current_progress, total_durasi_ms, prefix='Progress:', suffix='Selesai', length=40)
            
            del chunk_tensor, out_tensor, samples
            gc.collect()

    print("\n\nMenyimpan file...")
    
    final_audio_data = np.concatenate(hasil_bersih)
    
    final_audio_data = np.clip(final_audio_data, -0.99, 0.99)
    
    final_int16 = (final_audio_data * 32767).astype(np.int16)
    
    if output_path is None:
        folder_asal = os.path.dirname(input_path)
        nama_file_asal = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(folder_asal, f"BERSIH_{nama_file_asal}.mp3")
    
    try:
        audio_export = AudioSegment(
            data=final_int16.tobytes(),
            sample_width=2,
            frame_rate=16000,
            channels=1
        )
        audio_export.export(output_path, format="mp3", bitrate="64k")
        
        print(f"[SUCCESS] Selesai! File disimpan di:")
        print(f"    {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ar-Rauda AI Audio Cleaner (CLI)')
    parser.add_argument('input_file', help='Path ke file audio (mp3/wav)')
    parser.add_argument('--output', help='Path output file (opsional)', default=None)
    
    if len(sys.argv) > 1:
        args = parser.parse_args()
        if os.path.exists(args.input_file):
            bersihkan_kajian_cli(args.input_file, args.output)
        else:
            print(f"File tidak ditemukan: {args.input_file}")
    else:
        print("💡 Tips: Jalankan lewat terminal: python process_cli.py namafile.mp3")
        target = input(">> Masukkan lokasi file audio (drag & drop kesini): ").strip().strip('"').strip("'")
        if os.path.exists(target):
            bersihkan_kajian_cli(target)
        else:
            print("File tidak ditemukan.")
            input("Tekan Enter untuk keluar...")