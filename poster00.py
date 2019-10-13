# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 18:52:53 2019
增加了小图对接的色彩平滑效果
@author: hongwei
"""

from PIL import Image,ImageFilter
import time
from glob import glob
import numpy as np

Image.MAX_IMAGE_PIXELS = 1000000000
 
#取得图片色素均值矩阵
def get_mean(picarray,pieces):
    array1 = np.zeros(piecesnum*piecesnum*3)
    i = 0
    for x in range(0,piecesnum):
        for y in range(0,piecesnum):          
            R_mean = np.mean(picarray[x*pieces : x*pieces + pieces,y*pieces : y*pieces + pieces,0])
            G_mean = np.mean(picarray[x*pieces : x*pieces + pieces,y*pieces : y*pieces + pieces,1])
            B_mean = np.mean(picarray[x*pieces : x*pieces + pieces,y*pieces : y*pieces + pieces,2])
            array1[i] = R_mean
            i+=1
            array1[i] = G_mean
            i+=1
            array1[i] = B_mean
            i+=1
    return array1

#读取素材库
def load_image(picspath):    
    file_name=glob(picspath + '\\*jpg')
    sample = [] 
    meanarray = []
    i = 0
    for file in file_name:  
        picar =np.array(Image.open(file))
        mean1 = get_mean(picar,newpieces)
        sample.append(picar)  
        meanarray.append(mean1)
        i+=1
        if i%1000 == 0:
            print('已读取',i,'张素材')
    sample = np.array(sample)  
    meanarray = np.array(meanarray)  
    return sample , meanarray

#素描风格
def chaoyuemode1(img):
    a = np.asarray(img.convert('L')).astype('float')

    depth = 17.                     # (0-100)
    grad = np.gradient(a)            #取图像灰度的梯度值
    grad_x, grad_y = grad              #分别取横纵图像梯度值
    grad_x = grad_x*depth/100.
    grad_y = grad_y*depth/100.
    A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
    uni_x = grad_x/A
    uni_y = grad_y/A
    uni_z = 1./A
     
    vec_el = np.pi/2.2                  # 光源的俯视角度，弧度值
    vec_az = np.pi/4.                   # 光源的方位角度，弧度值
    dx = np.cos(vec_el)*np.cos(vec_az)  #光源对x 轴的影响
    dy = np.cos(vec_el)*np.sin(vec_az)  #光源对y 轴的影响
    dz = np.sin(vec_el)             #光源对z 轴的影响
     
    b = 255*(dx*uni_x + dy*uni_y + dz*uni_z)    #光源归一化
    b = b.clip(0,255)
     
    im = Image.fromarray(b.astype('uint8')).convert('RGB') #重构图像
    return im

#浮雕风格
def chaoyuemode3(img):
    a = img.filter(ImageFilter.EMBOSS)

    return a

#油画风格
def chaoyuemode2(img,templateSize=4, bucketSize=8, step=1):
    gray = np.asarray(img.convert('L')).astype('float')
    img = np.array(img)

    gray = ((gray/256)*bucketSize).astype(int)      
    h,w = img.shape[:2]
     
    oilImg = np.zeros(img.shape, np.uint8)          
     
    for i in range(0,h,step):        
        top = i-templateSize 
        bottom = i+templateSize+1
        if top < 0:
            top = 0
        if bottom >= h:
            bottom = h-1
            
        for j in range(0,w,step):            
            left = j-templateSize
            right = j+templateSize+1
            if left < 0:
                left = 0
            if right >= w:
                right = w-1                
           
            buckets = np.zeros(bucketSize,np.uint8)                    
            bucketsMean = [0,0,0]                                   
            
            buckets = np.bincount(gray[top:bottom,left:right].reshape(-1))
            maxBucketIndex = np.argmax(buckets)
            maxBucket = np.max(buckets)                 
            t = gray[top:bottom,left:right] == maxBucketIndex
            bucketsMean = np.sum(img[top:bottom,left:right][t],axis=0)

            bucketsMean = (bucketsMean/maxBucket).astype(int)        
            # 油画图
            oilImg[i:i+step,j:j+step,:] = bucketsMean

    oilImg = Image.fromarray(oilImg) #重构图像
    return  oilImg

#取绝对值范数
def getimg4(a1):
    a1 = get_mean(a1,oldpieces)
    temp =  a1 - means
    e = np.linalg.norm(temp,ord=1,axis=1,keepdims=True)
    index = np.argmin(e)
    return index 

#取欧氏距离范数
def getimg3(a1):
    a1 = get_mean(a1,oldpieces)
    temp =  a1 - means
    e = np.linalg.norm(temp,ord=2,axis=1,keepdims=True)
    b = np.argsort(e, axis=0)
    return b[np.random.randint(0,5)]

#原图放大处理
def optimize(imp,imq,w_dev,h_dev,diapha):
    pic_width = imq.width
    pic_hight = imq.height
    region = (0,0,imp.width-w_dev,imp.height-h_dev)    
    #裁切图片
    imp = imp.crop(region)
    imp = imp.resize((pic_width,pic_hight),Image.BILINEAR)
#    imp.save('模糊效果前.jpg')
    #高斯模糊
    imp = imp.filter(ImageFilter.GaussianBlur(radius=4))
#    imp.save('模糊效果后.jpg')
    imp = imp.convert('RGBA')
    #分离通道
    r,g,b,a = imp.split()    
    a = a.point(lambda i: i>0 and diapha)
    imp.putalpha(a)
    imq.paste(imp,(0,0),mask = a)
    return imq

#平滑边界
def Smoothing(img,w):
    f = np.asarray([1/(2*w)*i for i in range(1,2*w+1)])
    f1 = np.asarray([1/(w)*i for i in range(1,w+1)])
    f2 = f1[::-1]
    fn = np.r_[f1,f2]
    f1_n = 1 - fn
    for y in range(1,hstep):
        gap = img[y*newpic-w:y*newpic+w,:,:]
        R1 =  np.mean(gap[0:w,:,0],0) 
        R2 =  np.mean(gap[w+1:2*w,:,0],0) 
        R = R2-R1
        RM = R1 + np.dot(f[:,None],R[None,:])  
        
        G1 =  np.mean(gap[0:w,:,1],0) 
        G2 =  np.mean(gap[w+1:2*w,:,1],0) 
        G = G2-G1               
        GM = G1 + np.dot(f[:,None],G[None,:])        

        B1 =  np.mean(gap[0:w,:,2],0) 
        B2 =  np.mean(gap[w+1:2*w,:,2],0) 
        B = B2-B1               
        BM = B1 + np.dot(f[:,None],B[None,:])

        img[y*newpic-w:y*newpic+w,:,0] = img[y*newpic-w:y*newpic+w,:,0]*f1_n[:,None] + RM*fn[:,None]
        img[y*newpic-w:y*newpic+w,:,1] = img[y*newpic-w:y*newpic+w,:,1]*f1_n[:,None] + GM*fn[:,None]
        img[y*newpic-w:y*newpic+w,:,2] = img[y*newpic-w:y*newpic+w,:,2]*f1_n[:,None] + BM*fn[:,None]
    
    for x in range(1,wstep):
        gap = img[:,x*newpic-w:x*newpic+w,:]
        R1 =  np.mean(gap[:,0:w,0],1) 
        R2 =  np.mean(gap[:,w+1:2*w,0],1) 
        R = R2-R1
        RM = R1[:,None] + np.dot(R[:,None],f[None,:])        

        G1 =  np.mean(gap[:,0:w,1],1) 
        G2 =  np.mean(gap[:,w+1:2*w,1],1) 
        G = G2-G1
        GM = G1[:,None] + np.dot(G[:,None],f[None,:])        

        B1 =  np.mean(gap[:,0:w,2],1) 
        B2 =  np.mean(gap[:,w+1:2*w,2],1) 
        B = B2-B1
        BM = B1[:,None] + np.dot(B[:,None],f[None,:])
        
        img[:,x*newpic-w:x*newpic+w,0] = img[:,x*newpic-w:x*newpic+w,0]*f1_n[None,:] + RM*fn[None,:]
        img[:,x*newpic-w:x*newpic+w,1] = img[:,x*newpic-w:x*newpic+w,1]*f1_n[None,:] + GM*fn[None,:]
        img[:,x*newpic-w:x*newpic+w,2] = img[:,x*newpic-w:x*newpic+w,2]*f1_n[None,:] + BM*fn[None,:]                
    return img


#设定划分格数n（总数n*n）
piecesnum = 4

#最小块像素块的边长（像素单位）
oldpieces = 4
newpieces = 30 #piecesnum*newpieces需要等于素材图片边长

#图库路径
picspath = "square120"  #"square120"

#大图路径
pic = Image.open('原图\\'+'timg.jpg')
savapath = ''+'123.jpg'

#特效类型 0：正常  1：素描  2：油画  3:浮雕
m = 0

#渲染程度
diapha = 0
diapha = int(diapha*255/100)

#计算时间
time_start=time.time()

oldpic = oldpieces*piecesnum
newpic = newpieces*piecesnum
print('读取素材图片..............')

#取得小图及其均值矩阵的list
files , means = load_image(picspath)

time_mid=time.time()
print('读图用时 {:.0f}分 {:.0f}秒'.format(
       ( time_mid - time_start )// 60,  ( time_mid - time_start ) % 60))

if m == 1:
    pic = chaoyuemode1(pic)#素描
    time_mid=time.time()
    print('素描特效用时 {:.0f}分 {:.0f}秒'.format(
       ( time_mid - time_start )// 60,  ( time_mid - time_start ) % 60))
elif m == 2:    
    pic = chaoyuemode2(pic)#油画
    time_mid=time.time()
    print('油画特效用时 {:.0f}分 {:.0f}秒'.format(
       ( time_mid - time_start )// 60,  ( time_mid - time_start ) % 60))


#取得原图长宽
pic_width = pic.width
pic_hight = pic.height

#计算遍历的步数
wstep = int(pic_width/oldpic)
hstep = int(pic_hight/oldpic)

w_dev = pic_width%oldpic
h_dev = pic_hight%oldpic

#新建图矩阵
N = np.zeros([int(newpic*hstep),int(newpic*wstep),3])
#大图矩阵
picarray = np.array(pic)
print(picarray.shape)
print(N.shape)
for x in range(0,wstep):
    for y in range(0,hstep): 
        getf = files[getimg3(picarray[y*oldpic : y*oldpic + oldpic,x*oldpic : x*oldpic + oldpic,:])] 
        N[y*newpic : y*newpic + newpic,x*newpic : x*newpic + newpic,:] = getf
    print('已完成-------',round(x/wstep/0.01,2),'%')
print('平滑边界.................')
N = Smoothing(N,10)
toImage = Image.fromarray(N.astype('uint8')).convert('RGB')
print('整体渲染..................')
toImage = optimize(pic,toImage,w_dev,h_dev,diapha)
toImage.save(savapath)

time_end=time.time()
print('拼图完成用时 {:.0f}分 {:.0f}秒'.format(
       ( time_end - time_mid )// 60,  ( time_end - time_mid ) % 60))
print('总共用时 {:.0f}分 {:.0f}秒'.format(
       ( time_end - time_start )// 60,  ( time_end - time_start ) % 60))