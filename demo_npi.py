import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
 
import cv2

class MaskApp(ctk.CTk):
 
    def __init__(self):
        super().__init__()
 
        self.title("Smart Selection Application")
        self.geometry(f"{1280}x{720}")
       
        # Configure grid layout:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=6)
        self.grid_rowconfigure(0, weight=1)
 
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, padx=10, pady=10, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)
 
        self.button_frame = ctk.CTkFrame(self.sidebar_frame)
        self.button_frame.grid(row=0, column=0, rowspan=3, columnspan=2, padx=10, pady=10, sticky="ew")
        self.button_frame.grid_columnconfigure(0, weight=1)
 
        self.canvas_frame = ctk.CTkFrame(self, corner_radius=0)
        self.canvas_frame.grid(row=0, column=1, rowspan=6, padx=10, pady=10, sticky="nsew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
 
        self.logo_label = ctk.CTkLabel(self.button_frame, text="Masking Options", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
 
        # BUTTONS
        self.debug_btn = ctk.CTkButton(self.button_frame, text="Debug", command=self.debug_display)
        self.debug_btn.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
   
        self.load_btn = ctk.CTkButton(self.button_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
 
        self.save_btn = ctk.CTkButton(self.button_frame, text="Save Masked Area", command=self.save_mask)
        self.save_btn.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
 
        self.clear_btn = ctk.CTkButton(self.button_frame, text="Clear Image", command=self.clear_data)
        self.clear_btn.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
 
        self.mask_btns = ctk.CTkSegmentedButton(self.button_frame, command=self.switch_mask_type)
        self.mask_btns.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
        self.mask_btns.configure(values=["Point", "Lasso"])
        self.mask_btns.set("Point")

        # self.predictor = load_predictor()
 
        # Storage attributes:
        self.filename = None
        self.image = None
        self.canvas = None
        self.bbox = None
        self.mask = None
 
        # Hyperparameters:
        self.pointer_color = "blue"
        self.bbox_color = "blue"
        self.seg_color = (0, 255, 0) # Green
        self.expand_per = 0.15
        self.alpha = 0.18
        self.back_alpha = 0.30
        self.bbox_width = 2
 
        self.select_width = 2 # Selection width
 
        self.coords = []
        self.mask_coords = []
 
   
    @property
    def mask_type(self):
        """Return 1 if point type, 0 if lasso.
        """
        value = self.mask_btns.get()
        if value == "Point":
            return 1
        else:
            return 0
       
       
    def set_icon(self, icon_filepath):
 
        # ico = Image.open(icon_filepath)
        #photo = tk.PhotoImage(file=icon_filepath)
        # self.wm_iconphoto(True, photo)
        self.iconbitmap(icon_filepath)
 
 
    def select(self, event):
        """While holding down left-click, obtains the mouse coordinates.
        """
 
        # Gets direct coordinates:
        x, y = event.x, event.y
        self.coords.append((x, y))
 
        # Draw (only on lasso mode):
        x1, y1 = x - 1, y -1
        x2, y2 = x + 1, y + 1
        self.canvas.create_oval(x1, y1, x2, y2,
                                     fill=self.pointer_color,
                                     outline=self.pointer_color,
                                     width=1)
 
 
    def collect(self, event):
        # Only apply collection if there are actual coordinates:
        if len(self.coords) == 0:
            return None
       
        # Reset image:
        self.reset_image()
 
        # Transfer the coordinates:
        self.mask_coords = self.coords[:]
        # Reset:
        self.coords = []
 
        # Get and display bounding box:
        self.bbox, self.mask = self.predict_mask_bbox()
 
        # Render everything:
        self.render_mask()
 
   
    def select_point(self, event):
        """Apply for point masking option.
        """
 
        # Only add the initial mouse point (and neighbouring points):
        if len(self.coords) == 0:
            x = event.x
            y = event.y
 
            dims = (self.image.shape[1], self.image.shape[0])
            # Create a cross-neighbourhood around the point:
            self.coords = get_neighbour_points(x, y, self.select_width, dims)
 
 
    def predict_mask_bbox(self):
        """Get the bounding box using the collected sample points.
        """
 
        coords = self.mask_coords
        # Process coords
        input_points = np.array(coords)
 
        # Get mask points:
        # masks, scores, logits = predict_mask(self.predictor, input_points)
        # ind = np.argmax(scores)
        # mask = masks[ind]
        mask = predict_mask_refined(self.predictor, input_points)
 
        bbox = bounding_box(mask)
        x0, y0, w, h = bbox
        x1, y1 = x0 + w, y0 + h
 
        # Expand bounding box:
        shape = (self.image.shape[1], self.image.shape[0])
        x0, y0, x1, y1 = expand_bbox((x0, y0, x1, y1), shape, self.expand_per)
        x0 = x0 + self.bbox_width // 2
        x1 = x1 - self.bbox_width // 2
        y0 = y0 + self.bbox_width // 2
        y1 = y1 - self.bbox_width // 2
 
        return (x0, y0, x1, y1), mask
   
 
    def render_mask(self):
        """Display the current predicted mask and bounding box. For
        illustrative and user display purposes.
        """
 
        # Rendering bug unless you specify global variables:
        global rect, mask_image, dim_image
 
        (x0, y0, x1, y1) = self.bbox
        mask = self.mask
 
        # Draw the rectangular bbox:
        alpha = int(255 * self.alpha)
        fill = self.winfo_rgb(self.bbox_color) + (alpha,)
        rect = Image.new('RGBA', (x1-x0, y1-y0), fill)
        rect = ImageTk.PhotoImage(rect)
        self.canvas.create_image(x0, y0, anchor="nw", image=rect)
        self.canvas.create_rectangle(x0, y0, x1, y1,
                                     outline=self.bbox_color,
                                     width=self.bbox_width)
       
        # Draw the segmentation mask:
        mask_pil = render_mask(mask, self.seg_color, self.alpha)
        mask_image = ImageTk.PhotoImage(mask_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=mask_image)
 
        # Dim around the segmentation:
        # back_pil = render_back(mask, self.back_alpha)
        # dim_image = ImageTk.PhotoImage(back_pil)
        # self.canvas.create_image(0, 0, anchor="nw", image=dim_image)
 
 
    def load_image(self):
        filename = tk.filedialog.askopenfilename(
            initialdir="/home/daniel/Pictures",
            title="Select an image file",
            filetypes=(("PNG files", "*.png*"),
                       ("JPEG files", "*.jpg*"),
                       ("All files", "*.*"))
        )
 
        if len(filename) == 0:
            return None
       
        # Store filename:
        self.filename = filename
 
        # Already loaded an image: Clear it.
        if self.image is not None:
            self.clear_canvas()
 
        # Load file:
        image = load_image(filename)
        self.canvas_frame.update()
        canvas_dims = (self.canvas_frame.winfo_width(),
                       self.canvas_frame.winfo_height())
 
        # Get resized image dimensions:
        rs_image = rescale_image(image, canvas_dims)
        self.place_image(rs_image)
 
        # Set image:
        self.predictor.set_image(rs_image)
 
 
    def reset_image(self):
        self.clear_canvas()
        self.place_image(self.image)
 
 
    def clear_canvas(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
 
 
    def clear_data(self):
        """Clears everything, including loaded files.
        """
 
        self.clear_canvas()
 
        # Clear all attributes (to prevent saves)
        self.filename = None
        self.image = None
        self.bbox = None
        self.mask = None
        self.canvas = None
 
 
    def place_image(self, image):
        # Must use global here, there is a bug with canvas.
        global img
 
        dims = (image.shape[1], image.shape[0])
        self.image = image
 
        # Convert image:
        img = photo_image(image)
        # Get canvas:
        self.canvas = tk.Canvas(self.canvas_frame, height=dims[1], width=dims[0])
        self.canvas.grid(row=0, column=0)# , sticky=sticky)
        self.canvas.create_image(0, 0, anchor="nw", image=img)
 
        # Bind based on current mask type:
        if self.mask_type == 1:
            self.canvas.bind("<Button-1>", self.select_point)
        elif self.mask_type == 0:
            self.canvas.bind("<B1-Motion>", self.select)
       
        # Always get the release command:
        self.canvas.bind("<ButtonRelease-1>", self.collect)
 
        # if dims[0] > dims[1]:
           #  sticky = "ew"
        # else:
           #  sticky = "ns"
        # bg_img = ctk.CTkImage(Image.fromarray(image), size=dims)
        # self.image_label = ctk.CTkLabel(self.canvas_frame, text="", image=bg_img)
        # self.image_label.grid(row=0, column=0, sticky=sticky)
        # self.image_label.bind("<B1-Motion>", self.select)
        # self.image_label.bind("<ButtonRelease-1>", self.collect)
 
 
    def save_mask(self):
        """Saves the masked portion of the image.
        """
 
        # Button doesn't do anything if there is no bbox.
        if self.bbox is None or self.image is None:
            return None
       
        # Save location:
        f = tk.filedialog.asksaveasfile(
            mode='w',
            defaultextension=".png",
            initialdir="/home/daniel/Pictures",
            title="Choose a save location",
            filetypes=(("PNG files", "*.png*"),
                       ("JPEG files", "*.jpg*"),
                       ("All files", "*.*"))
        )
 
        if f is None:
            return None
       
        # Save cropped portion. Maybe scale up to original image?
        x0, y0, x1, y1 = self.bbox
        img = cv2.cvtColor(self.image[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f.name, img)
 
 
    def switch_mask_type(self, *args):
 
        # There is no canvas:
        if self.image is None:
            return None
 
        self.place_image(self.image)
 
 
    def debug_display(self):
        print(self.mask_btns.get())


if __name__ == "__main__":
    # Custom TK parameters:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
 
    app = MaskApp()
    app.mainloop()