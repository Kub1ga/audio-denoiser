import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import torch
import numpy as np
from denoiser import pretrained
from pydub import AudioSegment
import ssl
import os
import gc  

ssl._create_default_https_context = ssl._create_unverified_context

def proses_audio_ai_chunking(input_path, update_status_callback):
    try:
        update_status_callback("Sedang memuat model AI (DNS64)...")
        model = pretrained.dns64().cpu()
        model.eval()

        update_status_callback(f"Membaca file audio...")
        try:
            audio_full = AudioSegment.from_file(input_path)
            audio_full = audio_full.set_frame_rate(16000).set_channels(1)
        except Exception as e:
            return False, f"Gagal membaca audio: {str(e)}"
        
        chunk_durasi_ms = 30 * 1000 
        total_durasi_ms = len(audio_full)
        
        hasil_bersih = []
        
        update_status_callback(f"Memulai pembersihan bertahap (Total: {total_durasi_ms/1000/60:.1f} menit)...")

        with torch.no_grad():
            for i in range(0, total_durasi_ms, chunk_durasi_ms):
                chunk_audio = audio_full[i : i + chunk_durasi_ms]
                
                samples = np.array(chunk_audio.get_array_of_samples()).astype(np.float32)
                max_val = float(1 << (8 * chunk_audio.sample_width - 1))
                samples /= max_val
                
                chunk_tensor = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
                
                out_tensor = model(chunk_tensor)
                
                hasil_bersih.append(out_tensor.squeeze().cpu().numpy())
                
                persen = min((i + chunk_durasi_ms) / total_durasi_ms * 100, 100)
                update_status_callback(f"Memproses: {persen:.1f}% selesai...")
                
                del chunk_tensor, out_tensor, samples
                gc.collect()

        update_status_callback("Menyatukan dan menyimpan file...")
        
        final_audio_data = np.concatenate(hasil_bersih)
        
        final_audio_data = np.clip(final_audio_data, -0.99, 0.99)
        
        final_int16 = (final_audio_data * 32767).astype(np.int16)
        
        folder_asal = os.path.dirname(input_path)
        nama_file_asal = os.path.splitext(os.path.basename(input_path))[0]
        nama_output = os.path.join(folder_asal, f"BERSIH_{nama_file_asal}.mp3")
        
        audio_export = AudioSegment(
            data=final_int16.tobytes(),
            sample_width=2,
            frame_rate=16000,
            channels=1
        )
        
        audio_export.export(nama_output, format="mp3", bitrate="64k")
        
        update_status_callback(f"Selesai! Disimpan di:\n{nama_output}")
        return True, nama_output

    except Exception as e:
        return False, str(e)

class AudioCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ar-Rauda AI Audio Cleaner (Mode Aman & Hemat)")
        self.root.geometry("500x400")
        self.root.resizable(False, False)

        lbl_judul = tk.Label(root, text="Pembersih Audio Kajian (AI)", font=("Helvetica", 16, "bold"))
        lbl_judul.pack(pady=20)

        self.btn_pilih = tk.Button(root, text="Pilih File Audio (MP3/WAV)", command=self.pilih_file, width=30, height=2)
        self.btn_pilih.pack(pady=10)

        self.lbl_file = tk.Label(root, text="Belum ada file dipilih", fg="gray", wraplength=400)
        self.lbl_file.pack(pady=5)

        self.btn_proses = tk.Button(root, text="Mulai Bersihkan", command=self.mulai_proses, state="disabled", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=20, height=2)
        self.btn_proses.pack(pady=20)

        self.lbl_status = tk.Label(root, text="Siap.", fg="blue", wraplength=450, font=("Consolas", 10))
        self.lbl_status.pack(pady=10)

        self.file_path = None

    def pilih_file(self):
        filename = filedialog.askopenfilename(title="Pilih Audio", filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.aac")])
        if filename:
            self.file_path = filename
            self.lbl_file.config(text=f"File: {os.path.basename(filename)}", fg="black")
            self.btn_proses.config(state="normal")
            self.lbl_status.config(text="File siap diproses.")

    def update_status(self, text):
        self.lbl_status.config(text=text)
        self.root.update_idletasks() 

    def task_proses(self):
        self.btn_pilih.config(state="disabled")
        self.btn_proses.config(state="disabled")

        sukses, pesan = proses_audio_ai_chunking(self.file_path, self.update_status)

        if sukses:
            messagebox.showinfo("Berhasil", f"Audio berhasil dibersihkan!\n\nLokasi:\n{pesan}")
            self.update_status("Proses Selesai.")
        else:
            messagebox.showerror("Error", f"Terjadi kesalahan:\n{pesan}")
            self.update_status("Gagal memproses.")

        self.btn_pilih.config(state="normal")
        self.btn_proses.config(state="normal")

    def mulai_proses(self):
        threading.Thread(target=self.task_proses).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioCleanerApp(root)
    root.mainloop()