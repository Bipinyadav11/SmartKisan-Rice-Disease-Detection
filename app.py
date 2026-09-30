import sys
import os
import subprocess
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ============================================================
# SMARTKISAN - Rice Leaf Disease Detection & Advisory System
# ============================================================

root = tk.Tk()
root.title("🌾 SmartKisan - Rice Disease Detection")
root.geometry("950x650")
root.configure(bg="#dff6dd")
root.resizable(False, False)

# ------------------ Header ------------------
header = tk.Frame(root, bg="#2e7d32", height=80)
header.pack(fill="x")

title = tk.Label(
    header,
    text="🌾 SMART KISAN",
    font=("Arial", 28, "bold"),
    fg="white",
    bg="#2e7d32"
)
title.pack(pady=5)

subtitle = tk.Label(
    header,
    text="Deep Learning Based Rice Leaf Disease Detection & Advisory System",
    font=("Arial", 12),
    fg="white",
    bg="#2e7d32"
)
subtitle.pack()

# ------------------ Left Display Frame ------------------
left = tk.Frame(root, bg="#dff6dd")
left.place(x=20, y=100)

image_label = tk.Label(
    left,
    text="No Image Selected",
    width=40,
    height=18,
    bg="white",
    relief="solid",
    font=("Arial", 12)
)
image_label.pack()

# ------------------ Right Results Frame ------------------
right = tk.Frame(root, bg="#dff6dd")
right.place(x=500, y=100)

tk.Label(
    right,
    text="Prediction Result",
    font=("Arial", 20, "bold"),
    bg="#dff6dd",
    fg="#1b5e20"
).pack(pady=10)

disease_box = tk.Label(
    right,
    text="Disease : ---",
    font=("Arial", 16),
    bg="white",
    width=28,
    anchor="w",
    relief="solid",
    padx=10
)
disease_box.pack(pady=10)

confidence_box = tk.Label(
    right,
    text="Confidence : ---",
    font=("Arial", 16),
    bg="white",
    width=28,
    anchor="w",
    relief="solid",
    padx=10
)
confidence_box.pack(pady=10)

treatment_box = tk.Label(
    right,
    text="Treatment : ---",
    font=("Arial", 14),
    bg="white",
    width=35,
    height=6,
    justify="left",
    wraplength=350,
    anchor="nw",
    relief="solid",
    padx=10,
    pady=10
)
treatment_box.pack(pady=10)

# ============================================================
# Treatments Dictionary
# ============================================================
treatments = {
    "bacterial_leaf_blight": (
        "Apply copper-based bactericide.\n"
        "Avoid excess nitrogen fertilizer."
    ),
    "brown_spot": (
        "Spray recommended fungicide (e.g., Mancozeb).\n"
        "Maintain proper soil nutrients & field drainage."
    ),
    "healthy": (
        "The rice leaf is healthy.\n"
        "No chemical treatment required. Keep monitoring."
    ),
    "leaf_blast": (
        "Apply systemic fungicide (e.g., Tricyclazole).\n"
        "Avoid excessive use of nitrogenous fertilizers."
    ),
    "leaf_scald": (
        "Apply recommended fungicide.\n"
        "Maintain proper field sanitation & clean seeds."
    ),
    "narrow_brown_spot": (
        "Apply potassium fertilizers and suitable fungicides.\n"
        "Ensure good field drainage."
    )
}

# ============================================================
# PROCESS IMAGE PIPELINE
# ============================================================
def process_image(file_path):
    try:
        # Display Selected Image
        img = Image.open(file_path)
        img.thumbnail((350, 350))
        photo = ImageTk.PhotoImage(img)

        image_label.config(image=photo, text="")
        image_label.image = photo

        # Call predict.py using sys.executable
        predict_script_path = os.path.join(os.path.dirname(__file__), "predict.py")
        
        result = subprocess.run(
            [sys.executable, predict_script_path, file_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            messagebox.showerror("Prediction Error", result.stderr)
            return

        output = result.stdout.strip().splitlines()

        if len(output) < 2:
            messagebox.showerror(
                "Prediction Error",
                "Unexpected output format from predict.py\n\n" + result.stdout
            )
            return

        disease = output[0].replace("Disease :", "").strip()
        confidence_str = output[1].replace("Confidence :", "").strip()

        try:
            confidence_val = float(confidence_str.replace("%", "").strip())
        except ValueError:
            confidence_val = 0.0

        # Invalid check or extremely low confidence handling
        if disease.lower() == "invalid image" or confidence_val < 40.0:
            disease_box.config(text="Disease : Invalid Image")
            confidence_box.config(text=f"Confidence : {confidence_str}")
            treatment_box.config(
                text=(
                    "Treatment :\n\n"
                    "Unable to identify as a valid rice leaf.\n"
                    "Please upload or capture a clear rice leaf image."
                )
            )
            return

        # Success Case
        disease_box.config(text=f"Disease : {disease}")
        confidence_box.config(text=f"Confidence : {confidence_str}")

        disease_key = disease.lower().strip()
        treatment = treatments.get(disease_key, "Treatment details not available for this disease.")

        treatment_box.config(text=f"Treatment :\n\n{treatment}")

    except subprocess.TimeoutExpired:
        messagebox.showerror("Timeout", "Prediction process timed out.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ============================================================
# GUI ACTIONS
# ============================================================
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select Rice Leaf Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        process_image(file_path)

def open_camera():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        messagebox.showerror("Camera Error", "Unable to access camera.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            messagebox.showerror("Camera Error", "Failed to capture frame.")
            break

        cv2.imshow("SmartKisan Camera - SPACE: Capture | ESC: Exit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE Key
            captured_path = os.path.join(os.getcwd(), "captured_leaf.jpg")
            cv2.imwrite(captured_path, frame)
            camera.release()
            cv2.destroyAllWindows()
            process_image(captured_path)
            return

        elif key == 27:  # ESC Key
            break

    camera.release()
    cv2.destroyAllWindows()

def reset():
    image_label.config(image="", text="No Image Selected")
    image_label.image = None
    disease_box.config(text="Disease : ---")
    confidence_box.config(text="Confidence : ---")
    treatment_box.config(text="Treatment : ---")

# ------------------ Buttons ------------------
button_frame = tk.Frame(root, bg="#dff6dd")
button_frame.place(x=150, y=500)

upload_btn = tk.Button(
    button_frame, text="📂 Upload Image", font=("Arial", 13, "bold"),
    bg="#4CAF50", fg="white", activebackground="#388E3C", cursor="hand2", width=16,
    command=upload_image
)
upload_btn.grid(row=0, column=0, padx=5)

camera_btn = tk.Button(
    button_frame, text="📷 Open Camera", font=("Arial", 13, "bold"),
    bg="#388E3C", fg="white", activebackground="#2E7D32", cursor="hand2", width=16,
    command=open_camera
)
camera_btn.grid(row=0, column=1, padx=5)

reset_btn = tk.Button(
    button_frame, text="🔄 Reset", font=("Arial", 13, "bold"),
    bg="#f44336", fg="white", activebackground="#d32f2f", cursor="hand2", width=12,
    command=reset
)
reset_btn.grid(row=0, column=2, padx=5)

# ------------------ Voice Advisory ------------------
def voice_advisory():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        text = treatment_box.cget("text")

        if "Treatment : ---" in text:
            engine.say("Please upload or capture a rice leaf image first.")
        else:
            engine.say(text)
        engine.runAndWait()

    except Exception as e:
        messagebox.showerror("Voice Advisory Error", str(e))

voice_btn = tk.Button(
    root, text="🎤 Voice Advisory", font=("Arial", 13, "bold"),
    bg="#2196F3", fg="white", cursor="hand2",
    command=voice_advisory
)
voice_btn.place(x=700, y=500)

# ------------------ Footer ------------------
footer = tk.Label(
    root,
    text="Developed by Team SmartKisan | Deep Learning Based Rice Leaf Disease Detection System",
    font=("Arial", 10), bg="#2e7d32", fg="white"
)
footer.pack(side="bottom", fill="x")

if __name__ == "__main__":
    root.mainloop()