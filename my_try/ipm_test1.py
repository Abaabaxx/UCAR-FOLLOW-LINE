import cv2
import numpy as np

# --- 1. 参数配置区 (从您的C代码中移植) ---

# C代码中的变换矩阵
# C代码中的变换矩阵
# C代码中的变换矩阵 (使用测试矩阵1)
C_CODE_MATRIX = np.array([
    [-42.689601, 31.316015, 10088.624179],
    [-0.409502, 22.081064, -3077.007244],
    [-0.000118, 0.033758, -3.878394]
], dtype=np.float64)

# C代码中定义的目标俯视图(结果图)的尺寸 (宽度, 高度)
# 将输出尺寸改为更适合PC查看的大小，例如 480x640
OUTPUT_SHAPE = (480, 640)  # (height, width)

# 输入视频的路径 (请确保视频文件与此脚本在同一目录下, 或提供完整路径)
VIDEO_PATH = '/home/lby/CURSOR/follow_line/follow_line_mp4/line_left_2.mp4' 

# --- 2. 核心功能函数 (模拟C代码逻辑) ---

def create_lookup_table(source_shape, output_shape, matrix):
    """
    完全模拟C代码的ImagePerspective_Init()函数, 生成一个查找表(LUT)。
    
    参数:
        source_shape (tuple): 源图像的(高度, 宽度)
        output_shape (tuple): 目标俯视图的(高度, 宽度)
        matrix (np.array): 3x3变换矩阵
        
    返回:
        np.array: 一个形状为(out_h, out_w, 2)的查找表, 存储(x, y)坐标
    """
    source_h, source_w = source_shape
    output_h, output_w = output_shape
    
    # 创建一个空的查找表, 用-1作为无效标记
    lookup_table = np.full((output_h, output_w, 2), -1, dtype=np.int32)
    
    print("正在生成查找表 (LUT)...")
    # 遍历目标图像的每一个像素 (i对应列/宽度, j对应行/高度)
    for j in range(output_h):
        for i in range(output_w):
            # C代码中的核心计算公式
            denominator = matrix[2, 0] * i + matrix[2, 1] * j + matrix[2, 2] + 1e-9 # 加1e-9防止除以0
            
            local_x = (matrix[0, 0] * i + matrix[0, 1] * j + matrix[0, 2]) / denominator
            local_y = (matrix[1, 0] * i + matrix[1, 1] * j + matrix[1, 2]) / denominator
            
            # 边界检查, 确保坐标在源图像范围内
            if 0 <= local_x < source_w and 0 <= local_y < source_h:
                # 将有效的源坐标存入查找表
                lookup_table[j, i] = [int(local_x), int(local_y)]
                
    print("查找表生成完毕。")
    return lookup_table

def apply_lookup_table(frame, lut):
    """
    使用预先计算好的LUT来高效生成IPM图像。
    """
    output_h, output_w, _ = lut.shape
    
    # 创建一个黑色的空白输出图像
    ipm_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    # 使用NumPy的高级索引功能一次性完成像素复制, 效率极高
    # 1. 找到LUT中所有有效的位置 (坐标不是-1)
    valid_mask = lut[:, :, 0] != -1
    
    # 2. 获取这些有效位置在LUT中存储的源坐标(x, y)
    src_coords = lut[valid_mask]
    src_x = src_coords[:, 0]
    src_y = src_coords[:, 1]
    
    # 3. 直接从源图像(frame)中抓取所有需要的像素
    pixels = frame[src_y, src_x]
    
    # 4. 将抓取到的像素填充到目标图像的对应位置
    ipm_frame[valid_mask] = pixels
    
    return ipm_frame

# --- 3. 主程序 ---

def main():
    """
    主函数: 读取视频, 应用变换, 显示结果
    """
    # 打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{VIDEO_PATH}'")
        return

    # 获取视频的原始尺寸和帧率
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    delay = int(1000 / fps)

    # 在循环开始前, 一次性生成查找表
    lut = create_lookup_table(source_shape=(source_h, source_w), 
                              output_shape=OUTPUT_SHAPE, 
                              matrix=C_CODE_MATRIX)

    print("\n视频处理中... 按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束。")
            break
        
        # 在每一帧中, 应用查找表来生成IPM图像
        ipm_frame = apply_lookup_table(frame, lut)
        
        # 显示原始视频和IPM结果
        cv2.imshow('Original Video', frame)
        cv2.imshow('IPM View (from C-code LUT logic)', ipm_frame)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出。")

if __name__ == '__main__':
    main()