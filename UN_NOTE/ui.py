import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from tkinter import messagebox
#import tensorflow.keras as keras
import numpy as np
import subprocess


# 创建Tkinter窗口
window = tk.Tk()
window.title("心脏病检测")

# 设置窗口大小
window_width = 800
window_height = 600
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# 设置背景图片
background_image = Image.open("t1.png")
background_image = background_image.resize((window_width, window_height), Image.ANTIALIAS)
background_image_tk = ImageTk.PhotoImage(background_image)
background_label = tk.Label(window, image=background_image_tk)
background_label.pack()

# 创建按钮事件处理函数
def open_image():
    # 打开文件对话框并获取用户选择的图像文件
    file_path = filedialog.askopenfilename(filetypes=[("图像文件", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # 创建新窗口
        image_window = tk.Toplevel(window)
        image_window.title("上传图像")

        # 加载图像
        img = Image.open(file_path)
        img = img.resize((window_width, window_height))  # 调整图像大小以适应窗口
        img_tk = ImageTk.PhotoImage(img)

        # 显示图像
        image_panel = tk.Label(image_window, image=img_tk)
        image_panel.image = img_tk  # 保持图像对象的引用，避免被垃圾回收
        image_panel.pack(expand=True)

        # 进行心脏病检测
        result = detect_heart_disease(file_path)

        # 创建标签用于显示预测结果
        result_label = tk.Label(image_window, text="预测结果: " + result, font=("Helvetica", 16), bg='white')
        result_label.pack()

# 加载训练好的模型
model_path = None
model = None

def load_model():
    global model
    if model_path is not None:
        model = keras.models.load_model(model_path)

def choose_resnet_model():
    global model_path
    model_path = 'XIN-Res.h5'
    load_model()

def choose_alexnet_model():
    global model_path
    model_path = 'XIN-AlexNet.h5'
    load_model()

choose_resnet_button = tk.Button(window, text="选择 Res 模型", command=choose_resnet_model, font=("Helvetica", 16))
choose_resnet_button.place(relx=0.5, rely=0.3, anchor='center')

choose_alexnet_button = tk.Button(window, text="选择 Alex 模型", command=choose_alexnet_model, font=("Helvetica", 16))
choose_alexnet_button.place(relx=0.5, rely=0.4, anchor='center')

# 创建按钮用于打开图像
open_button = tk.Button(window, text="心脏病预测", command=open_image, font=("Helvetica", 16))
open_button.place(relx=0.5, rely=0.5, anchor='center')

def detect_heart_disease(image_path):
    # 加载图像并进行预处理
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))####加载图片
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # 使用模型进行预测
    prediction = model.predict(img)
    if prediction > 0.5:
        return "心脏病"
    else:
        return "健康"

# 创建退出按钮事件处理函数
def exit_program():
    if messagebox.askokcancel("退出", "确定要退出程序吗？"):
        window.destroy()

# 创建退出按钮
exit_button = tk.Button(window, text="退出", command=exit_program, font=("Helvetica", 16))
exit_button.place(relx=0.5, rely=0.9, anchor='center')

def open_image_viewer():
    subprocess.Popen(["python", "hands.py"])

open_image_viewer_button = tk.Button(window, text="查看图片", command=open_image_viewer, font=("Helvetica", 16))
open_image_viewer_button.place(relx=0.5, rely=0.7, anchor='center')
# 运行Tkinter窗口主循环
window.mainloop()