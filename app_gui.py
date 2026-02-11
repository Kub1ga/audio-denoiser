import tkinter as no
from tkinter import filedialog, messagebox
import threading
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
    max_val = float(1 << (8 * audio.sample_width - 1))
    samples /= max_val
    return torch.from_numpy(samples).unsqueeze(0), 16000

def proses_audio_ai(input_path, update_status_callback):
    try:
        update_status_callback("Sedang memuat model AI (DNS64)...")
        model = pretrained.dns64().cpu()
        model.eval()

        update_status_callback(f"Membaca file: {os.path.basename(input_path)}...")
        out, sr = load_audio_flexible(input_path)

        update_status_callback("Sedang membersihkan noise... (Mohon tunggu)")
        
        with torch.no_grad():
            out_batch = out.unsqueeze(0)
            denoised_batch = model(out_batch)
            denoised = denoised_batch.squeeze().cpu().numpy()
        
        folder_asal = os.path.dirname(input_path)
        nama_file_asal = os.path.basename(input_path)
        nama_output = os.path.join(folder_asal, f"BERSIH_{nama_file_asal}.wav") # Output .wav agar aman

        denoised_int = (denoised * 32767).astype(np.int16)
        wavfile.write(nama_output, sr, denoised_int)
        
        update_status_callback(f"Selesai! Disimpan di:\n{nama_output}")
        return True, nama_output

    except Exception as e:
        return False, str(e)

class AudioCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ar-Rauda AI Audio Cleaner")
        self.root.geometry("500x350")
        self.root.resizable(False, False)

        lbl_judul = no.Label(root, text="Pembersih Audio Kajian (AI)", font=("Helvetica", 16, "bold"))
        lbl_judul.pack(pady=20)

        self.btn_pilih = no.Button(root, text="Pilih File Audio (MP3/WAV)", command=self.pilih_file, width=30, height=2)
        self.btn_pilih.pack(pady=10)

        self.lbl_file = no.Label(root, text="Belum ada file dipilih", fg="gray")
        self.lbl_file.pack(pady=5)

        self.btn_proses = no.Button(root, text="Mulai Bersihkan", command=self.mulai_proses, state="disabled", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=20, height=2)
        self.btn_proses.pack(pady=20)

        self.lbl_status = no.Label(root, text="Siap.", fg="blue", wraplength=450)
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
        self.root.update()

    def task_proses(self):
        self.btn_pilih.config(state="disabled")
        self.btn_proses.config(state="disabled")
        
        sukses, pesan = proses_audio_ai(self.file_path, self.update_status)

        if sukses:
            messagebox.showinfo("Berhasil", f"Audio berhasil dibersihkan!\nFile: {pesan}")
            self.update_status("Proses Selesai.")
        else:
            messagebox.showerror("Error", f"Terjadi kesalahan:\n{pesan}")
            self.update_status("Gagal memproses.")

        self.btn_pilih.config(state="normal")
        self.btn_proses.config(state="normal")

    def mulai_proses(self):
        threading.Thread(target=self.task_proses).start()

if __name__ == "__main__":
    root = no.Tk()
    app = AudioCleanerApp(root)
    root.mainloop()