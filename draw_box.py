import tkinter as tk
from PIL import Image, ImageDraw
class DrawingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Drawing Canvas")

        # Initialize drawing variables
        self.drawing = False
        self.last_x = None
        self.last_y = None

        self.canvas_size = 256
        self.new_canvas()

        self.updated = False

        self.root.bind("<Return>", self.save_image)
        self.add_text_box('')

    def update(self):
        self.root.update()

    def set_text(self, text):
        self.add_text_box(text)

    def new_canvas(self):
        # Remove the existing canvas if it exists
        if hasattr(self, 'canvas'):
            self.canvas.destroy()

        # Create a new canvas with the specified background color
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="#f9f7ea")
        self.canvas.pack()

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Create a new Image object to draw on
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "#f9f7ea")
        self.draw = ImageDraw.Draw(self.image)

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def on_draw(self, event):
        if self.drawing:
            radius = 5.647058823529412 # Radius of the circle
            x0, y0 = event.x - radius, event.y - radius
            x1, y1 = event.x + radius, event.y + radius
            self.canvas.create_oval(x0, y0, x1, y1, fill="black", outline="black")
            self.draw.ellipse((x0, y0, x1, y1), fill="black", outline="black")

        self.last_x = event.x
        self.last_y = event.y

    def save_image(self, event):
        self.saved_image = self.image.copy()
        self.updated = True
        self.canvas.destroy()
        self.new_canvas()

    def get_new_image(self):
        if not self.updated:
            return None
        ret = self.saved_image.copy()
        self.updated = False
        return ret

    def add_text_box(self, text):
        # Create a rounded rectangle on the canvas
        text_box_width = 100
        text_box_height = 30
        corner_radius = 10
        x0 = self.canvas_size - text_box_width - 10
        y0 = self.canvas_size - text_box_height - 10
        x1 = self.canvas_size - 10
        y1 = self.canvas_size - 10

        self.canvas.create_arc(x0, y0, x0 + 2 * corner_radius, y0 + 2 * corner_radius, start=90, extent=90,
                               style=tk.PIESLICE, fill="white", outline="white")
        self.canvas.create_arc(x1 - 2 * corner_radius, y0, x1, y0 + 2 * corner_radius, start=0, extent=90,
                               style=tk.PIESLICE, fill="white", outline="white")
        self.canvas.create_arc(x0, y1 - 2 * corner_radius, x0 + 2 * corner_radius, y1, start=180, extent=90,
                               style=tk.PIESLICE, fill="white", outline="white")
        self.canvas.create_arc(x1 - 2 * corner_radius, y1 - 2 * corner_radius, x1, y1, start=270, extent=90,
                               style=tk.PIESLICE, fill="white", outline="white")

        self.canvas.create_rectangle(x0 + corner_radius, y0, x1 - corner_radius, y1, fill="white", outline="white")
        self.canvas.create_rectangle(x0, y0 + corner_radius, x1, y1 - corner_radius, fill="white", outline="white")

        # Add a label inside the rounded rectangle
        self.text_label = tk.Label(self.canvas, text=text, bg="white", fg="black", font=("Arial", 10))
        self.canvas.create_window(self.canvas_size - text_box_width // 2 - 10,
                                  self.canvas_size - text_box_height // 2 - 10, window=self.text_label)