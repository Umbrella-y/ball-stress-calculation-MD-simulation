import sys
import time
import os
import pandas as pd
import numpy as np
import math as math
import warnings
import matplotlib.pyplot as plt
from sympy import init_printing, Matrix
init_printing(use_unicode=True)
warnings.filterwarnings("ignore")
##_____________________________________________________
filepath = r'E:\test'
filepath = filepath + '/'
##_____________________________________________________
def find_basic (filename):##输出文件的基本信息，时间步，尺寸和原子数目
    dict_parameters = {}#'TIMESTEP':'','NUMBER OF ATOMS':'','BOX':''
    with open (filename,'r', encoding='utf-8') as d:
        contents = d.readlines()
        i=0
        for line in contents[:6]:
            judge = 'ITEM'
            flag = judge in line
            if flag is True:
                continue
            else:
                i+=1
                dict_parameters[i] = line
                print(line)
        print(dict_parameters)
    return dict_parameters
#find_basic(r'E:\test/30.cor.dump.1355000')
##______________________________________________________
def findI(files):#查找文件中是否还包含文件头
    str = "ITEM:"
    files = os.listdir(files)#创建路径下所有文件的列表
    files.sort()#初始化排序文件
    files.sort(key = lambda x: float(x[:-4]))#按照数字大小排序
    for file in files:
        file = filepath + '/'+ file
        with open(file, 'r', encoding = 'utf-8') as d:
            contents = d.readlines()
            firstline = contents[0]
            flag = str in firstline
            if flag is True:
                return flag
                break
##______________________________________________________ 
def delete_top(files):#删除每个txt文件的头9行
    files = os.listdir(files)#创建路径下所有文件的列表
    files.sort()#初始化排序文件
    files.sort(key = lambda x: float(x[:-4]))#按照数字大小排序
    for file in files:
        print("\t" + file)#打印输出排序结果
    for file in files:
        file = filepath + '/'+ file
        with open(file, 'r', encoding = 'utf-8') as d:
            contents = d.readlines()#遍历所有的文件，除了代码输出文件以外的所有文件的每一行
        with open(file,'w') as d:
            d.write(''.join(contents[9:]))#删除每个txt文件的头9行
    
##______________________________________________________            
def find_center(filename):##寻找球体的质心在哪里
    with open (filename,'r', encoding='utf-8') as d:
        contents = d.readlines()
        firstline = contents[0]
        flag = 'ITEM' in firstline
        if flag is True:
            contentsnew = contents[8:]
        else:
            contentsnew = contents
    contentsnew[0] = contentsnew[0][11:-1]
    #print(contentsnew[0])
    res=contentsnew[0].strip(' ')
    res=res.strip(' ')
    res=res.split(' ')
    df = pd.read_table(filename ,sep='\s+',skiprows=9 ,names=res)
    #print(df)
    centerx = (df['x'].mean())
    centery = (df['y'].mean())
    centerz = (df['z'].mean())
    print(centerx,centery,centerz)
    df.loc[:,'nx'] = df.loc[:,'x']-centerx
    df.loc[:,'ny'] = df.loc[:,'y']-centery
    df.loc[:,'nz'] = df.loc[:,'z']-centerz
    print('the mass center is',centerx,centery,centerz,'!!new coordinates is calculated!!')
    print(df)
    return df#找到质心后返回一个已经重新计算好了坐标值的数据库

##______________________________________________________
def Cartesian2Spherical(p):
    # p = (x,y,z)
    # theta  in (0,pi) and phi in (0,2pi)
    x,y,z = p
    r = np.sqrt(x*x+y*y+z*z)
    phi = np.arctan2(y,x)  # Inclination
    theta = np.arccos(z/r)  # Azimuth
    q = np.array([r,theta,phi])
    return q
##______________________________________________________
def Spherical2Cartesian(q):
    # q = (r,theta,phi)
    # theta  in (0,pi) and phi in (0,2pi)
    r,theta,phi = q
    SinTheta = np.sin(theta)
    CosTheta = np.cos(theta)
    SinPhi = np.sin(phi)
    CosPhi = np.cos(phi)
    rSinTheta = r*SinTheta
    x = rSinTheta*CosPhi
    y = rSinTheta*SinPhi
    z = r*CosTheta
    p  = np.array([x,y,z])
    return p
##______________________________________________________
def xyz_to_rphitheta(df):
    m = np.array([df.loc[:,'nx'],df.loc[:,'ny'],df.loc[:,'nz']])
    m = Cartesian2Spherical(m)
    df.loc[:,'r'] = m[0]
    df.loc[:,'theta'] = m[1]
    df.loc[:,'phi'] = m[2]
    print(df)
    return df
##______________________________________________________________

def eulerAngles2rotationMat(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format is 'degree':
        theta = [i * math.pi / 180.0 for i in theta]
 
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
 
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
##______________________________________________________________
def calculate_RRPP(df):
    bash = pd.DataFrame()#columns= ['RRstress','PhiPhistress']
    a=0
    for i in df.index:
        time_start = time.time()
        df.loc[i,'rrr'] = math.sqrt((df.loc[i,'nx'])**2+(df.loc[i,'ny'])**2+(df.loc[i,'nz'])**2)
        nx = float(df.loc[i,'nx'])
        ny = float(df.loc[i,'ny'])
        nz = float(df.loc[i,'nz'])
        rr_vector = np.array([nx,ny,nz])
        degree_z = df.loc[i,'phi']
        degree_y = df.loc[i,'theta']
        degree_x = np.pi
        rotate_degree = np.array([degree_x,degree_y,degree_z])
        rotate_matrix = eulerAngles2rotationMat(rotate_degree, format='notdgree')
        y = np.matrix([[df.loc[i,'c_peratom[1]'],df.loc[i,'c_peratom[4]'],df.loc[i,'c_peratom[5]']], 
                        [df.loc[i,'c_peratom[4]'],df.loc[i,'c_peratom[2]'],df.loc[i,'c_peratom[6]']], 
                        [df.loc[i,'c_peratom[5]'],df.loc[i,'c_peratom[6]'],df.loc[i,'c_peratom[3]']]])
        z = np.dot(rotate_matrix,y)
        z = np.dot(z,rotate_matrix.T)
        df.loc[i,'sigmaRR'] = z[0,0]
        df.loc[i,'sigmaPhiPhi'] = z[2,2]
        df.loc[i,'sigmatheta'] = z[1,1]
        time_end = time.time()
        
        print('Current step =',a,'calculate stress time =',time_end-time_start,end='\r')
        #time.sleep(1)
        a+=1
    df = df.sort_values(by='rrr',ascending=True)
    bash.loc[:,'rrr'] = df.loc[:,'rrr']
    bash.loc[:,'sigmaRR'] = df.loc[:,'sigmaRR']
    bash.loc[:,'sigmaPhiPhi'] = df.loc[:,'sigmaPhiPhi']
    bash.loc[:,'sigmatheta'] = df.loc[:,'sigmatheta']
    bash.sort_values(by='rrr',ascending=True)
    #df = df.to_csv('RR&PhiPhi_stress.csv')
    return bash,df
##______________________________________________________ 
def pingjun_hang(df):
    time_start = time.time()
    my_dict = dict()
    df_data = globals()
    boxlow = float((filename)[3].split()[0])
    boxhigh = float(find_basic(filename)[3].split()[1])
    binsize =  0.1##################################################################################切分的球壳的厚度#########################——————————————————————————————————
    huafen = int((boxhigh - boxlow)/binsize)
    for i in range(0,huafen-1):
        my_dict['rrr' + str(i)] = list()
    radiuslist = list()
    for i in range(0,huafen-1):
        radiuslist.append(int(binsize*i))
    for d in df.index:
        #print(d)
        for i in range(0,huafen-1):
            x1 = 0 + binsize*i
            x2 = 10 + binsize*i 
            if df.loc[d,'rrr']>= x1 and df.loc[d,'rrr'] < x2:
                my_dict['rrr' + str(i)].append(list(df.loc[d]))
    zongshuju = pd.DataFrame(columns=['rrr','sigmaRR','sigmaPhiPhi','sigmatheta'])
    listRadius = list()
    listRR = list()
    listPhiPhi =list()
    listtheta =list()
    for i in range(0,huafen-1):
        data1 = my_dict['rrr' + str(i)]
        df_data['rrr' + str(i)] = pd.DataFrame(data1,columns=['rrr','sigmaRR','sigmaPhiPhi','sigmatheta'])
        #print(df_data['rrr' + str(i)])
        zonghe = df_data['rrr' + str(i)].apply(lambda x:x.sum(),axis=0)
        ##zonghe____<class 'pandas.core.series.Series'>
                    # r             -3.598678e+01
                    # sigmaRR       -3.499415e+07
                    # sigmaPhiPhi    3.146590e+07
        listRadius.append(zonghe[0])
        listRR.append(zonghe[1])
        listPhiPhi.append(zonghe[2])
        listtheta.append(zonghe[3])
    zongshuju.loc[:,'rrr'] = pd.DataFrame(radiuslist)
    x = 0
    for i in zongshuju.index:
        print(i)
        zongshuju.loc[x,'sigmaRR'] = pd.DataFrame(listRR).loc[x][0]
        zongshuju.loc[x,'sigmaPhiPhi'] = pd.DataFrame(listPhiPhi).loc[x][0]
        zongshuju.loc[x,'sigmatheta'] = pd.DataFrame(listtheta).loc[x][0]
        x = x + 1
    time_end = time.time()
    print('calculate stress time =',time_end-time_start,end='\r')
    return zongshuju

##______________________________________________________ 
##计算完成后需添加文件头（自动）
def add_header(filename, header):
    with open(filename, "r+") as file:
        old = file.readlines()[1:]
        #print(old)
        file.seek(0)
        file.writelines(header)
        add = "ITEM: ATOMS id type x y z vx vy vz c_peratom[1] c_peratom[2] c_peratom[3] c_peratom[4] c_peratom[5] c_peratom[6] nx ny nz r theta phi rrr sigmaRR sigmaPhiPhi sigmatheta\n"
        file.write(add)
        file.writelines(old)
##______________________________________________________ 
filepath = r'E:\test\铝3k保存'   
files = os.listdir(r'E:\test\铝3K/')#创建路径下所有文件的列表
fileresource = r'E:\test\铝3K/'
files.sort()#初始化排序文件
files.sort(key = lambda x: float(x[-6:]))#按照数字大小排序   
startpoint = 1       
for file in files:
    file = fileresource +'/'+file
    find_basic(file)
    df = find_center(file)
    df = xyz_to_rphitheta(df)
    h1,h2 = calculate_RRPP(df)
    h2.to_csv(filepath +'/'+ 'Aftercalculation{}.txt'.format(startpoint),sep=' ',index=False)
    with open(file,'r') as w:
        head = w.readlines()[:8]
    add_header(filepath +'/'+ 'Aftercalculation{}.txt'.format(startpoint),head)
    startpoint+=1

