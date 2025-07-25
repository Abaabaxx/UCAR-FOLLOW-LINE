import cv2
import numpy as np

# 定义逆透视变换矩阵
# 注意：这里使用的是示例矩阵，您需要替换为您实际的变换矩阵
IPM_MATRIX = np.array([
    [1.2, 0.0, -100],
    [0.0, 1.2, -50],
    [0.0, 0.002, 1.0]
], dtype=np.float64)

# 定义输出图像的尺寸
OUTPUT_SIZE = (640, 480)

def run_ipm_test(video_path):
    """
    读取视频文件，应用逆透视变换，并实时显示原始和变换后的视频
    
    Args:
        video_path: 输入视频的路径
    """
    # 创建视频捕获对象
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取失败，则退出循环
        if not ret:
            print("视频读取完毕或发生错误")
            break
        
        # 应用逆透视变换
        ipm_frame = cv2.warpPerspective(frame, IPM_MATRIX, OUTPUT_SIZE)
        
        # 显示原始帧和变换后的帧
        cv2.imshow('原始视图', frame)
        cv2.imshow('IPM视图', ipm_frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 替换为您的视频文件路径
    input_video_file = 'follow_line_mp4/line_left_1.mp4'
    
    # 运行测试
    run_ipm_test(input_video_file) 