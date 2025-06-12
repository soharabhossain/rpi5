import tkinter as tk
from PIL import Image, ImageTk

class TrafficGUI:
    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Intelligent Traffic Light Management")
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.label_1 = tk.Label(self.root, text="Traffic Light 1: RED", font=('Arial', 14))
        self.label_1.pack()
        self.label_2 = tk.Label(self.root, text="Traffic Light 2: GREEN", font=('Arial', 14))
        self.label_2.pack()

        self.wait_label_1 = tk.Label(self.root, text="Timer 1: 90s")
        self.wait_label_1.pack()
        self.wait_label_2 = tk.Label(self.root, text="Timer 2: 90s")
        self.wait_label_2.pack()

        self.update_gui()

    def update_frame(self, frame):
        image = Image.fromarray(frame)
        image = image.resize((640, 480))
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def update_labels(self, state1, state2, time1, time2):
        self.label_1.config(text=f"Traffic Light 1: {state1}")
        self.label_2.config(text=f"Traffic Light 2: {state2}")
        self.wait_label_1.config(text=f"Timer 1: {time1}s")
        self.wait_label_2.config(text=f"Timer 2: {time2}s")

    def update_gui(self):
        self.controller.process_frame()
        self.root.after(1000, self.update_gui)

    def run(self):
        self.root.mainloop()
