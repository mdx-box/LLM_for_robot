from PIL import Image
import os

def images_to_gif(input_folder, output_gif, duration=500, scale=0.5):
    """
    将文件夹中的图片加载并转换为 GIF 动画，同时降低分辨率。

    参数:
    - input_folder: 包含图片的文件夹路径。
    - output_gif: 输出 GIF 文件路径 (如: "output.gif")。
    - duration: 每帧之间的间隔时间（单位：毫秒）。
    - scale: 分辨率缩放比例（如 0.5 表示降低为原来的 50%）。
    """
    # 获取文件夹中的所有图片文件（按名称排序）
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])

    # 加载图片并调整分辨率
    images = []
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = Image.open(img_path)
        
        # 调整分辨率
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img_resized = img.resize((new_width, new_height))
        images.append(img_resized)

    if not images:
        print("No images found in the folder!")
        return

    # 将图片保存为 GIF 动画
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 设置循环次数，0 表示无限循环
    )
    print(f"GIF 动画已保存到 {output_gif}")

# 使用示例
input_folder = "F:/PHD/4 task/79 国资委项目/DTAssembly"  # 替换为图片所在文件夹路径
output_gif = "output.gif"  # 替换为输出 GIF 文件名
images_to_gif(input_folder, output_gif, duration=50, scale=0.3)
