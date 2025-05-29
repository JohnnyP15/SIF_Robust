import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import os


class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.labels = {}
        self.entries = {}
        self.window.title(window_title)

        # Set video source to the camera (0 is default camera)
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        # Set a smaller resolution for faster processing and better window fit
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Change to a smaller resolution like 640x480
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        x = (screen_width / 2) - (self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
        y = (screen_height / 2) - (self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2) - 100

        self.window.geometry('%dx%d+%d+%d' % (
            self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 300, x, y))

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=0, column=0, columnspan=4, pady=20)

        self.create_sliders()

        self.btn_capture = tk.Button(window, text="Capture", command=self.capture_image)
        self.btn_reset = tk.Button(window, text="Reset to Default", command=self.reset_to_default)
        self.btn_filter = tk.Button(window, text="Toggle Grayscale", command=self.toggle_grayscale)

        self.btn_capture.grid(row=8, column=0, pady=20, padx=10, sticky=tk.W)
        self.btn_reset.grid(row=8, column=1, pady=20, padx=10, sticky=tk.W)
        self.btn_filter.grid(row=8, column=2, pady=20, padx=10, sticky=tk.W)

        self.grayscale = False
        self.delay = 10
        self.scale_factor = 1  # Scale factor to maintain original image size
        self.image_counter = 0

        # Create directory for saving images
        self.image_dir = "calibration_images"
        os.makedirs(self.image_dir, exist_ok=True)

        self.update()
        self.window.mainloop()

    def create_sliders(self):
        self.contrast_slider = self.create_slider("Contrast", self.vid.get(cv2.CAP_PROP_CONTRAST), row=1, column=0,
                                                  from_=0, to_=255)
        self.brightness_slider = self.create_slider("Brightness", self.vid.get(cv2.CAP_PROP_BRIGHTNESS), row=2,
                                                    column=0, from_=0, to_=255)
        self.focus_slider = self.create_slider("Focus", self.vid.get(cv2.CAP_PROP_FOCUS), row=3, column=0, from_=0,
                                               to_=255)
        self.saturation_slider = self.create_slider("Saturation", self.vid.get(cv2.CAP_PROP_SATURATION), row=4,
                                                    column=0, from_=0, to_=255)
        self.gain_slider = self.create_slider("Gain", self.vid.get(cv2.CAP_PROP_GAIN), row=5, column=0, from_=0,
                                              to_=255)
        self.exposure_slider = self.create_slider("Exposure", self.vid.get(cv2.CAP_PROP_EXPOSURE), row=1, column=1,
                                                  from_=-8, to_=8)
        self.white_balance_slider = self.create_slider("White Balance", self.vid.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
                                                       row=2, column=1, from_=2800, to_=6500)
        self.sharpness_slider = self.create_slider("Sharpness", 0, row=3, column=1, from_=0, to_=255)
        self.gamma_slider = self.create_slider("Gamma", 0, row=4, column=1, from_=0, to_=255)
        self.zoom_slider = self.create_slider("Zoom", self.vid.get(cv2.CAP_PROP_ZOOM), row=5, column=1, from_=0,
                                              to_=10)  # Assuming zoom range 0 to 10

    def create_slider(self, label_text, initial_value, row, column, from_, to_):
        label = ttk.Label(self.window, text=label_text)
        label.grid(row=row, column=column * 4, pady=5, padx=5, sticky=tk.W)

        var = tk.DoubleVar()
        var.set(initial_value)
        var.trace("w", lambda name, index, mode, var=var, label_text=label_text: self.update_label(var, label_text))

        slider = ttk.Scale(self.window, from_=from_, to_=to_, orient=tk.HORIZONTAL, variable=var,
                           command=self.update_camera_settings)
        slider.grid(row=row, column=column * 4 + 1, pady=5, padx=5)

        value_label = ttk.Label(self.window, text=str(initial_value))
        value_label.grid(row=row, column=column * 4 + 2, pady=5, padx=5, sticky=tk.W)
        self.labels[label_text] = value_label

        entry = ttk.Entry(self.window, textvariable=var, width=5)
        entry.grid(row=row, column=column * 4 + 3, pady=5, padx=5, sticky=tk.W)
        self.entries[label_text] = entry

        return slider

    def capture_image(self):
        ret, frame = self.vid.read()
        if ret:
            self.image_counter += 1
            filename = os.path.join(self.image_dir, f"calibration_image_{self.image_counter}.jpg")
            cv2.imwrite(filename, frame)

            with open("settings.csv", "a") as f:
                f.write(
                    "Image,{},Contrast,{},Brightness,{},Focus,{},Saturation,{},Gain,{},Exposure,{},White Balance,{},Sharpness,{},Gamma,{},Zoom,{}\n".format(
                        filename,
                        self.contrast_slider.get(),
                        self.brightness_slider.get(),
                        self.focus_slider.get(),
                        self.saturation_slider.get(),
                        self.gain_slider.get(),
                        self.exposure_slider.get(),
                        self.white_balance_slider.get(),
                        self.sharpness_slider.get(),
                        self.gamma_slider.get(),
                        self.zoom_slider.get()
                    ))
            print(f"Captured {filename}")
        else:
            print("Failed to capture image")

    def reset_to_default(self):
        self.contrast_slider.set(self.vid.get(cv2.CAP_PROP_CONTRAST))
        self.brightness_slider.set(self.vid.get(cv2.CAP_PROP_BRIGHTNESS))
        self.focus_slider.set(self.vid.get(cv2.CAP_PROP_FOCUS))
        self.saturation_slider.set(self.vid.get(cv2.CAP_PROP_SATURATION))
        self.gain_slider.set(self.vid.get(cv2.CAP_PROP_GAIN))
        self.exposure_slider.set(self.vid.get(cv2.CAP_PROP_EXPOSURE))
        self.white_balance_slider.set(self.vid.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))
        self.sharpness_slider.set(0)
        self.gamma_slider.set(0)
        self.zoom_slider.set(self.vid.get(cv2.CAP_PROP_ZOOM))

    def update_camera_settings(self, value=None):
        self.vid.set(cv2.CAP_PROP_CONTRAST, self.contrast_slider.get())
        self.vid.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness_slider.get())
        self.vid.set(cv2.CAP_PROP_FOCUS, self.focus_slider.get())
        self.vid.set(cv2.CAP_PROP_SATURATION, self.saturation_slider.get())
        self.vid.set(cv2.CAP_PROP_GAIN, self.gain_slider.get())
        self.vid.set(cv2.CAP_PROP_EXPOSURE, self.exposure_slider.get())
        self.vid.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.white_balance_slider.get())
        self.vid.set(cv2.CAP_PROP_SHARPNESS, self.sharpness_slider.get())
        self.vid.set(cv2.CAP_PROP_GAMMA, self.gamma_slider.get())
        self.vid.set(cv2.CAP_PROP_ZOOM, self.zoom_slider.get())

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Resize the frame if needed
            frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for display
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    def update_label(self, var, label_text):
        value = var.get()
        self.labels[label_text].config(text=str(int(value)))

    def toggle_grayscale(self):
        self.grayscale = not self.grayscale


# Main execution
root = tk.Tk()
app = CameraApp(root, "Tkinter Camera UI")