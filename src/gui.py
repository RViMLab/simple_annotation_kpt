"""Folder picker and process launcher for the annotation tool."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SETTINGS_PATH = Path(__file__).resolve().parents[1] / '.gui_settings.json'


def load_settings():
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def initial_directory(value):
    path = Path(value) if value else Path(__file__).resolve().parents[1]
    while not path.is_dir() and path != path.parent:
        path = path.parent
    return str(path)


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def dataset_ready(scene):
    try:
        scene = Path(scene)
        images = scene / 'images'
        names = {p.name for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
        metadata = json.loads((scene / 'center_points_masks_annotations.json').read_text(encoding='utf-8'))
        tracks = json.loads((scene / 'keypoints.json').read_text(encoding='utf-8'))
        listed = [item['file_name'] for item in metadata['images']]
        return bool(names) and bool(listed) and len(listed) == len(set(listed)) and set(listed) == names and isinstance(tracks['tracks'], list)
    except (OSError, ValueError, KeyError, TypeError):
        return False


class AnnotationApp:
    def __init__(self, root):
        self.root = root
        root.title('Simple Annotation KPT')
        root.geometry('850x520')
        settings = load_settings()
        self.input = tk.StringVar(value=str(settings.get('input', '')))
        self.output = tk.StringVar(value=str(settings.get('output', Path(__file__).resolve().parents[1] / 'outputs')))
        self.step = tk.StringVar(value=str(settings.get('frame_step', '1')))
        self.status = tk.StringVar()
        self.busy = False
        self.process = None
        self.events = queue.Queue()
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill='both', expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text='Input image folder / video').grid(row=0, column=0, sticky='w')
        self.input_entry = ttk.Entry(frame, textvariable=self.input)
        self.input_entry.grid(row=0, column=1, sticky='ew', padx=8)
        self.folder_button = ttk.Button(frame, text='Choose folder', command=self.choose_input)
        self.folder_button.grid(row=0, column=2)
        self.video_button = ttk.Button(frame, text='Choose video', command=self.choose_video)
        self.video_button.grid(row=0, column=3, padx=5)
        ttk.Label(frame, text='Output root folder').grid(row=1, column=0, sticky='w', pady=12)
        self.output_entry = ttk.Entry(frame, textvariable=self.output)
        self.output_entry.grid(row=1, column=1, sticky='ew', padx=8)
        self.output_button = ttk.Button(frame, text='Choose output', command=self.choose_output)
        self.output_button.grid(row=1, column=2)
        ttk.Label(frame, text='Keep every Nth frame').grid(row=2, column=0, sticky='w')
        self.step_entry = ttk.Spinbox(frame, from_=1, to=10000, textvariable=self.step, width=8)
        self.step_entry.grid(row=2, column=1, sticky='w', padx=8)
        ttk.Label(frame, textvariable=self.status, wraplength=790).grid(row=3, column=0, columnspan=4, sticky='w', pady=12)
        controls = ttk.Frame(frame)
        controls.grid(row=4, column=0, columnspan=4, sticky='w')
        self.extract_button = ttk.Button(controls, text='Extract', command=self.extract)
        self.extract_button.pack(side='left', padx=(0, 10))
        self.review_button = ttk.Button(controls, text='Start annotation', command=self.review)
        self.review_button.pack(side='left', padx=(0, 10))
        self.open_button = ttk.Button(controls, text='Open result folder', command=self.open_result)
        self.open_button.pack(side='left')
        self.log = tk.Text(frame, height=15, wrap='word', state='disabled')
        self.log.grid(row=5, column=0, columnspan=4, sticky='nsew', pady=12)
        frame.rowconfigure(5, weight=1)
        self.input.trace_add('write', self.paths_changed)
        self.output.trace_add('write', self.paths_changed)
        self.step.trace_add('write', self.paths_changed)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()
        root.after(100, self.poll)

    def save_settings(self):
        data = {'input': self.input.get(), 'output': self.output.get(), 'frame_step': self.step.get()}
        try:
            temporary = SETTINGS_PATH.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
            temporary.replace(SETTINGS_PATH)
        except OSError as exc:
            self.write_log(f'Could not save folder preferences: {exc}\n')

    def paths_changed(self, *_):
        self.refresh()
        # Debounce manual typing; picker selections and launch/close also save.
        pending = getattr(self, '_settings_after', None)
        if pending is not None:
            self.root.after_cancel(pending)
        self._settings_after = self.root.after(300, self.persist_paths)

    def persist_paths(self):
        self._settings_after = None
        self.save_settings()

    def scene(self):
        if not self.input.get().strip() or not self.output.get().strip():
            return None
        # Same naming rule as extractor._scene_name; normalize trailing separators.
        name = os.path.splitext(Path(self.input.get().strip()).name)[0]
        return Path(self.output.get().strip()) / name

    def refresh(self):
        scene = self.scene()
        ready = scene is not None and dataset_ready(scene)
        self.extract_button.configure(state='disabled' if self.busy else 'normal')
        for button in (self.review_button, self.open_button):
            button.configure(state='normal' if ready and not self.busy else 'disabled')
        for widget in (self.input_entry, self.output_entry, self.folder_button, self.video_button, self.output_button, self.step_entry):
            widget.configure(state='disabled' if self.busy else 'normal')
        state = 'Processing...' if self.busy else ('Ready to annotate' if ready else 'Extract required')
        self.status.set(f'{state}\nResult: {scene if scene else "Choose input and output folders"}')

    def choose_input(self):
        value = filedialog.askdirectory(title='Choose an image folder', initialdir=initial_directory(self.input.get()))
        if value:
            self.input.set(str(Path(value)))

    def choose_video(self):
        value = filedialog.askopenfilename(title='Choose video', initialdir=initial_directory(self.input.get()), filetypes=[('Videos', '*.mp4 *.avi *.mov *.mkv *.wmv'), ('All files', '*.*')])
        if value:
            self.input.set(value)

    def choose_output(self):
        value = filedialog.askdirectory(title='Choose output root folder', initialdir=initial_directory(self.output.get()), mustexist=False)
        if value:
            self.output.set(value)

    def write_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        self.log.see('end')
        self.log.configure(state='disabled')

    def launch(self, args):
        self.save_settings()
        self.busy = True
        self.refresh()
        script = Path(__file__).resolve().parents[1] / 'main.py'
        def worker():
            try:
                self.process = subprocess.Popen([sys.executable, '-u', str(script), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
                for line in self.process.stdout:
                    self.events.put(('log', line))
                self.events.put(('done', self.process.wait()))
            except Exception as exc:
                self.events.put(('log', str(exc) + '\n'))
                self.events.put(('done', -1))
        threading.Thread(target=worker, daemon=True).start()

    def extract(self):
        source = Path(self.input.get().strip())
        if not self.input.get().strip() or not source.exists() or self.scene() is None:
            messagebox.showerror('Input required', 'Choose an existing input and an output root folder.')
            return
        try:
            step = int(self.step.get())
            if step < 1:
                raise ValueError()
        except ValueError:
            messagebox.showerror('Invalid frame step', 'Frame step must be a positive integer.')
            return
        scene = self.scene()
        if scene.exists() and any(scene.iterdir()):
            messagebox.showinfo('Existing result', 'This result folder already contains files. Choose a new output root for extraction to protect existing annotations. If extraction is complete, click Start annotation.')
            return
        self.launch(['extract', '--input', str(source), '--output', self.output.get().strip(), '--frame-step', str(step)])

    def review(self):
        if self.scene() is not None and dataset_ready(self.scene()):
            self.launch(['review', str(self.scene())])

    def open_result(self):
        if self.scene() is not None and dataset_ready(self.scene()):
            os.startfile(str(self.scene()))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'log':
                    self.write_log(value)
                else:
                    self.busy = False
                    self.process = None
                    self.refresh()
                    if value:
                        messagebox.showerror('Operation failed', 'See the log for details. Run this tool in a Python environment with OpenCV, NumPy and Matplotlib installed.')
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def close(self):
        if self.busy:
            messagebox.showinfo('Operation running', 'Finish extraction or close the annotation window before closing the launcher.')
            return
        self.save_settings()
        pending = getattr(self, '_settings_after', None)
        if pending is not None:
            self.root.after_cancel(pending)
        self.root.destroy()


def run_gui():
    root = tk.Tk()
    AnnotationApp(root)
    root.mainloop()
