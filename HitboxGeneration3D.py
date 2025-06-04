import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import trimesh
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import json
import time

plt.style.use('dark_background')


class AdvancedVoxelGenerator:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Voxel Generator")
        self.master.geometry("1600x1000")
        self.setup_style()

        # Инициализация переменных
        self.mesh = None
        self.current_hitboxes = []
        self.progress = 0
        self.progress_running = False
        self.cancel_processing = False
        self.camera_state = {
            'elev': 30,
            'azim': 45,
            'distance': 5.0,
            'x_center': 0.0,
            'y_center': 0.0
        }
        self.precision_levels = {
            'super low': {'clusters': 50, 'iterations': 5},
            'low': {'clusters': 150, 'iterations': 10},
            'medium': {'clusters': 300, 'iterations': 15},
            'high': {'clusters': 600, 'iterations': 20},
            'ultra': {'clusters': 1200, 'iterations': 25}
        }

        self.setup_ui()
        self.setup_bindings()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#444444', foreground='#00FF00')
        style.configure('TFrame', background='#444444')
        style.configure('TLabel', background='#444444', foreground='white')
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.map('TButton', background=[('active', '#555555')])
        style.configure('red.Horizontal.TProgressbar',
                        background='#00FF00',
                        troughcolor='#333333',
                        lightcolor='#00FF00',
                        darkcolor='#00FF00')

    def setup_ui(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Область предпросмотра
        self.fig = plt.figure(figsize=(14, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#2a2a2a')

        # Панель управления
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Прогресс-бар
        self.progress_bar = ttk.Progressbar(
            control_frame,
            orient=tk.HORIZONTAL,
            length=280,
            mode='determinate',
            style='red.Horizontal.TProgressbar'
        )
        self.progress_bar.pack(pady=10)

        # Кнопки управления
        self.load_btn = ttk.Button(control_frame, text="Load Model", command=self.load_model)
        self.precision_btn = ttk.Button(control_frame, text="Precision Settings", command=self.show_precision_menu)
        self.export_btn = ttk.Button(control_frame, text="Export Model", command=self.show_export_menu)
        self.cancel_btn = ttk.Button(control_frame, text="Cancel", command=self.cancel_processing)

        buttons = [self.load_btn, self.precision_btn, self.export_btn, self.cancel_btn]
        for btn in buttons:
            btn.pack(fill=tk.X, pady=3)

        # Консоль
        console_frame = ttk.LabelFrame(control_frame, text="Console")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.console = scrolledtext.ScrolledText(
            console_frame,
            height=12,
            state='disabled',
            bg='#e0e0e0',
            font=('Consolas', 9)
        )
        self.console.pack(fill=tk.BOTH, expand=True)

        self.cmd_entry = ttk.Entry(console_frame)
        self.cmd_entry.pack(fill=tk.X, pady=5)
        self.cmd_entry.bind("<Return>", self.process_command)

    def setup_bindings(self):
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.master.bind("<Left>", lambda e: self.move_camera('left'))
        self.master.bind("<Right>", lambda e: self.move_camera('right'))
        self.master.bind("<Up>", lambda e: self.move_camera('up'))
        self.master.bind("<Down>", lambda e: self.move_camera('down'))

    def log(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"> {message}\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def update_progress(self, value):
        self.progress_bar['value'] = value
        self.master.update_idletasks()

    def start_processing_thread(self, func, *args):
        threading.Thread(target=func, args=args, daemon=True).start()

    def cancel_processing(self):
        self.cancel_processing = True
        self.log("Operation cancelled")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("3D Files", "*.obj *.stl *.glb *.fbx *.ply")])
        if path:
            self.start_processing_thread(self.process_model, path)

    def process_model(self, path):
        try:
            self.progress_running = True
            self.cancel_processing = False
            self.update_progress(10)

            self.mesh = trimesh.load(path, force='mesh')
            if isinstance(self.mesh, trimesh.Scene):
                self.mesh = self.mesh.dump(concatenate=True)

            self.update_progress(30)
            self.generate_hitboxes('high')

            self.update_progress(90)
            self.master.after(0, self.update_viewport)
            self.log(f"Model loaded: {len(self.mesh.faces)} faces")

        except Exception as e:
            self.log(f"Error: {str(e)}")
        finally:
            self.progress_running = False
            self.update_progress(0)

    def show_precision_menu(self):
        if self.mesh is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return

        menu = tk.Menu(self.master, tearoff=0)
        for level in self.precision_levels:
            menu.add_command(
                label=level.title(),
                command=lambda l=level: self.set_precision(l)
            )
        menu.post(self.precision_btn.winfo_rootx(),
                  self.precision_btn.winfo_rooty() + self.precision_btn.winfo_height())

    def set_precision(self, level):
        if self.mesh is None:
            return
        self.start_processing_thread(self.generate_hitboxes, level)

    def generate_hitboxes(self, precision_level):
        try:
            self.progress_running = True
            self.cancel_processing = False
            self.update_progress(10)

            params = self.precision_levels[precision_level]
            vertices = self.mesh.vertices

            if precision_level == 'super low':
                self.current_hitboxes = [self.get_bounding_box(vertices)]
                return

            kmeans = KMeans(
                n_clusters=min(params['clusters'], len(vertices)),
                init='k-means++',
                n_init=1,
                max_iter=params['iterations'],
                verbose=0,
                random_state=42
            )

            kmeans.fit(vertices)
            self.update_progress(50)

            for i in range(3):
                if self.cancel_processing: return
                closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vertices)
                kmeans.cluster_centers_ = vertices[closest]
                self.update_progress(50 + i * 10)
                time.sleep(0.1)

            labels = kmeans.predict(vertices)
            self.current_hitboxes = [self.get_bounding_box(vertices[labels == i]) for i in range(kmeans.n_clusters)]
            self.update_progress(80)

            self.master.after(0, self.update_viewport)
            self.log(f"Generated {len(self.current_hitboxes)} hitboxes ({precision_level} precision)")

        except Exception as e:
            self.log(f"Processing Error: {str(e)}")
        finally:
            self.progress_running = False
            self.update_progress(0)

    def get_bounding_box(self, points):
        return np.array([np.min(points, axis=0), np.max(points, axis=0)])

    def update_viewport(self):
        self.ax.clear()

        if self.mesh:
            self.ax.plot_trisurf(
                *self.mesh.vertices.T,
                triangles=self.mesh.faces,
                alpha=0.1,
                color='#555555'
            )

            for minc, maxc in self.current_hitboxes:
                self.draw_voxel(minc, maxc)

        self.update_camera()
        self.canvas.draw()

    def draw_voxel(self, minc, maxc):
        edges = np.array([
            [minc[0], minc[1], minc[2]],
            [maxc[0], minc[1], minc[2]],
            [maxc[0], maxc[1], minc[2]],
            [minc[0], maxc[1], minc[2]],
            [minc[0], minc[1], maxc[2]],
            [maxc[0], minc[1], maxc[2]],
            [maxc[0], maxc[1], maxc[2]],
            [minc[0], maxc[1], maxc[2]]
        ])

        segments = [[edges[s], edges[e]] for s, e in [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]]

        self.ax.add_collection3d(Line3DCollection(
            segments, colors='#00FF00', linewidths=0.8, alpha=0.7
        ))

    def update_camera(self):
        self.ax.view_init(
            elev=self.camera_state['elev'],
            azim=self.camera_state['azim']
        )
        self.ax.dist = self.camera_state['distance']
        self.ax.set_xlim3d(
            self.camera_state['x_center'] - self.camera_state['distance'],
            self.camera_state['x_center'] + self.camera_state['distance']
        )
        self.ax.set_ylim3d(
            self.camera_state['y_center'] - self.camera_state['distance'],
            self.camera_state['y_center'] + self.camera_state['distance']
        )

    def show_export_menu(self):
        menu = tk.Menu(self.master, tearoff=0)
        formats = [
            ('Wavefront OBJ', '.obj'),
            ('Blender File', '.blend'),
            ('STL Model', '.stl'),
            ('JSON Data', '.json'),
            ('GLTF Format', '.gltf')
        ]

        for text, ext in formats:
            menu.add_command(
                label=text,
                command=lambda e=ext: self.export_model(e)
            )

        menu.post(
            self.export_btn.winfo_rootx(),
            self.export_btn.winfo_rooty() + self.export_btn.winfo_height()
        )

    def export_model(self, fmt):
        path = filedialog.asksaveasfilename(
            defaultextension=fmt,
            filetypes=[(f"{fmt.upper()} Files", f"*{fmt}")]
        )

        if path:
            try:
                if fmt == '.obj':
                    self.export_obj(path)
                elif fmt == '.blend':
                    self.export_blend(path)
                elif fmt == '.stl':
                    self.export_stl(path)
                elif fmt == '.json':
                    self.export_json(path)
                elif fmt == '.gltf':
                    self.export_gltf(path)

                self.log(f"Exported as {fmt}")
            except Exception as e:
                self.log(f"Export Error: {str(e)}")

    def export_stl(self, path):
        boxes = trimesh.util.concatenate([
            trimesh.primitives.Box(
                extents=maxc - minc,
                transform=trimesh.transformations.translation_matrix((maxc + minc) / 2)
            )
            for minc, maxc in self.current_hitboxes
        ])
        boxes.export(path)

    def export_obj(self, path):
        with open(path, 'w') as f:
            for minc, maxc in self.current_hitboxes:
                box = trimesh.primitives.Box(extents=maxc - minc)
                box.apply_translation((maxc + minc) / 2)
                f.write(box.export(file_type='obj'))

    def export_json(self, path):
        data = {'hitboxes': [[h[0].tolist(), h[1].tolist()] for h in self.current_hitboxes]}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def export_blend(self, path):
        try:
            import bpy

            # Очистка сцены
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Создание коллекции
            collection = bpy.data.collections.new("Hitboxes")
            bpy.context.scene.collection.children.link(collection)

            # Создание материала
            material = bpy.data.materials.new(name="Hitbox_Material")
            material.diffuse_color = (0.0, 1.0, 0.0, 1.0)  # RGBA

            for i, (minc, maxc) in enumerate(self.current_hitboxes):
                if self.cancel_processing: return

                # Создание меша
                box = trimesh.primitives.Box(extents=maxc - minc)
                mesh = bpy.data.meshes.new(f'Hitbox_{i}')
                mesh.from_pydata(
                    vertices=box.vertices.tolist(),
                    edges=[],
                    faces=box.faces.tolist()
                )

                # Создание объекта
                obj = bpy.data.objects.new(f'Hitbox_{i}', mesh)
                obj.location = (maxc + minc) / 2
                obj.scale = (1, 1, 1)

                # Назначение материала
                mesh.materials.append(material)

                # Добавление в коллекцию
                collection.objects.link(obj)
                obj.hide_set(False)

                # Добавление модификатора Wireframe
                wireframe = obj.modifiers.new("Wireframe", 'WIREFRAME')
                wireframe.thickness = 0.01

                self.update_progress(90 + (i / len(self.current_hitboxes)) * 10)

            # Сохранение файла
            bpy.ops.wm.save_as_mainfile(filepath=path)
            self.log(f"Exported {len(self.current_hitboxes)} visible hitboxes to Blender")

        except Exception as e:
            self.log(f"Blender Export Error: {str(e)}")

    def export_gltf(self, path):
        scene = trimesh.Scene()
        for minc, maxc in self.current_hitboxes:
            box = trimesh.primitives.Box(extents=maxc - minc)
            box.apply_translation((maxc + minc) / 2)
            scene.add_geometry(box)
        scene.export(path)

    def process_command(self, event):
        cmd = self.cmd_entry.get().strip().lower()
        self.cmd_entry.delete(0, tk.END)

        commands = {
            'clear': self.clear_console,
            'reset': self.reset_camera,
            'help': self.show_help,
            'info': self.show_info
        }

        if cmd in commands:
            commands[cmd]()
        else:
            self.log(f"Unknown command: {cmd}")

    def clear_console(self):
        self.console.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.config(state=tk.DISABLED)

    def reset_camera(self):
        self.camera_state = {
            'elev': 30,
            'azim': 45,
            'distance': 5.0,
            'x_center': 0.0,
            'y_center': 0.0
        }
        self.update_camera()
        self.canvas.draw()
        self.log("Camera reset")

    def show_help(self):
        help_text = """
        Available commands:
        clear    - Clear console
        reset    - Reset camera
        info     - Show model info
        help     - Show this help
        """
        self.log(help_text.strip())

    def show_info(self):
        if self.mesh:
            info = f"""
            Vertices: {len(self.mesh.vertices)}
            Faces: {len(self.mesh.faces)}
            Hitboxes: {len(self.current_hitboxes)}
            Bounds: {self.mesh.bounds}
            """
            self.log(info.strip())
        else:
            self.log("No model loaded")

    def on_mouse_press(self, event):
        self.last_mouse_pos = (event.x, event.y)

    def on_mouse_move(self, event):
        if event.button == 1:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]

            self.camera_state['azim'] -= dx * 0.5
            self.camera_state['elev'] += dy * 0.5

            self.update_camera()
            self.canvas.draw_idle()
            self.last_mouse_pos = (event.x, event.y)

    def on_scroll(self, event):
        zoom_factor = 0.1 if event.button == 'up' else -0.1
        self.camera_state['distance'] = max(0.5, self.camera_state['distance'] - zoom_factor)
        self.update_camera()
        self.canvas.draw_idle()

    def move_camera(self, direction):
        move_speed = 0.1 * self.camera_state['distance']

        if direction == 'left':
            self.camera_state['x_center'] += move_speed
        elif direction == 'right':
            self.camera_state['x_center'] -= move_speed
        elif direction == 'up':
            self.camera_state['y_center'] -= move_speed
        elif direction == 'down':
            self.camera_state['y_center'] += move_speed

        self.update_camera()
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedVoxelGenerator(root)
    root.mainloop()