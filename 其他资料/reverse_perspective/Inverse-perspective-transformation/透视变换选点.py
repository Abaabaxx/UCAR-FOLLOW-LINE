# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:28:40 2025

@author: 寻木
"""

import cv2
import os
import sys

# 跨平台 getch 函数实现，用于替代仅限 Windows 的 msvcrt.getch()
try:
    # Windows 平台
    import msvcrt
    def _getch():
        # 在Windows上，方向键是两个字节，第一个是 b'\xe0'
        key = msvcrt.getch()
        if key == b'\xe0':
            # 读取第二个字节来判断是哪个方向键
            key = msvcrt.getch()
            if key == b'H': return b'H' # 上
            if key == b'P': return b'P' # 下
            if key == b'K': return b'K' # 左
            if key == b'M': return b'M' # 右
        return key # 返回普通按键
except ImportError:
    # Linux/macOS 平台
    import tty
    import termios
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # 一次性读取最多3个字节
            ch_bytes = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        # 将读取到的字节序列转换为脚本期望的格式
        if ch_bytes == '\x1b[A': return b'H' # 上
        if ch_bytes == '\x1b[B': return b'P' # 下
        if ch_bytes == '\x1b[D': return b'K' # 左
        if ch_bytes == '\x1b[C': return b'M' # 右
        if ch_bytes == '\r' or ch_bytes == '\n': return b'\r' # 回车
        if ch_bytes == '\x1b': return b'\x1b' # ESC
        return ch_bytes.encode() # 其他按键

# 全局 getch 函数引用
getch = _getch

img=cv2.imread("/home/lby/CURSOR/follow_line/reverse_perspective/Inverse-perspective-transformation/1.png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
imgori=img.copy()
all_point=[[0,0],[0,0],[0,0],[0,0]]
img=cv2.resize(img,(int(img.shape[1]*0.5),int(img.shape[0]*0.5)))
img1=img.copy()
step=10
print("使用方法：\n\n1.输入点的坐标，然后按任意方向键查看图中的位置\n\n2.位置不满意即可使用方向键移动\n(若发现按下方向键没反应可以鼠标点一下图片的窗口再试试)\n\n3.若满意，按回车即可保存\n\n4.若想换一个位置和移动的步长，按esc即可\n\n\n")
for k in range (0,4):
    
    while True:
        print("请输入第"+str(k+1)+"个点的像素点的位置：")
        x0 = (input("请输入x坐标\n\n"))
        y0= (input ("请输入y坐标\n\n"))
        
        if len(x0)!=0 and len(y0)!=0 and str(int(x0))==x0 and str(int(y0))==y0:
            x0 = int(x0)
            y0= int(y0)
            
            break
    
    while True:
       key1=getch() 
       
       # --- 简化后的按键处理逻辑 ---
       if key1 in (b'\r', b'\n'): #回车键 (兼容 Windows 和 Linux/macOS)
           all_point[k]=[x0,y0]
           print("cv2坐标(左上角为原点)为\n")
           print (x0,y0)
           break
       elif key1==b'\x1b': #输入键，esc
           
           img1=img.copy()

           while True:
               print("请输入第"+str(k+1)+"个点的像素点的位置：")
               x0 = (input("请输入x坐标\n\n"))
               y0= (input ("请输入y坐标\n\n"))
               step=int(input("请输入方向键步长\n\n"))
               if 0 < step <200 and len(x0)!=0 and len(y0)!=0 and str(int(x0))==x0 and str(int(y0))==y0:
                   x0 = int(x0)
                   y0= int(y0)
                   break
           x=int(x0/2)
           y=int(y0/2)
           # --- 使用安全的 cv2.rectangle 替换手动绘图 ---
           # 使用 cv2.rectangle 绘制一个 10x10 的中心标记，这比手动操作像素更安全
           marker_size = 5
           # cv2.rectangle 接受 (x, y) 格式的点
           pt1 = (x - marker_size, y - marker_size)
           pt2 = (x + marker_size, y + marker_size)
           cv2.rectangle(img1, pt1, pt2, (0, 0, 255), -1) # -1 表示填充矩形
           
           cv2.imshow("1", img1)
           cv2.waitKey(1)
           # --- 绘图逻辑结束 ---
           print (x0,y0)
           
       elif key1==b'H' or key1==b"P" or key1==b"K" or key1==b"M":
           
           img1=img.copy()
           
           if key1==b'H' : #方向键上
               y0-=step
           elif key1==b"P" : #方向键下
               y0+=step
           elif key1==b"K" : #方向键左
               x0-=step
           elif key1==b"M" : #方向键右
               x0+=step
           x=int(x0/2)
           y=int(y0/2)
           # --- 使用安全的 cv2.rectangle 替换手动绘图 ---
           # 使用 cv2.rectangle 绘制一个 10x10 的中心标记，这比手动操作像素更安全
           marker_size = 5
           # cv2.rectangle 接受 (x, y) 格式的点
           pt1 = (x - marker_size, y - marker_size)
           pt2 = (x + marker_size, y + marker_size)
           cv2.rectangle(img1, pt1, pt2, (0, 0, 255), -1) # -1 表示填充矩形
           
           cv2.imshow("1", img1)
           cv2.waitKey(1)
           # --- 绘图逻辑结束 ---
           print (x0,y0)
           
       
       
       
       
for k in range (0,4):
    #print("自然坐标系下：")
    print(all_point[k][0],all_point[k][1])
    print("\n\n\n")
    x=all_point[k][0]
    y=all_point[k][1]
    for i in range (0,10):
        for j in range (0,10):
            imgori[y+i,x+j]=(0,0,255)
            imgori[y-i,x-j]=(0,0,255)
    

    
cv2.imshow("1",imgori)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 移除 cv2.imwrite，因为它在循环的最后，会导致图片窗口闪退
# cv2.imwrite("2.png",imgori)

# 跨平台替代 os.system("pause")
print("处理完成，请按回车键退出...")
input()
cv2.imwrite("2.png",imgori) # 将保存图片的操作移到最后
cv2.destroyAllWindows()