import numpy as np
import matplotlib.pyplot as plt
from utils import DLT
import cv2 as cv
plt.style.use('seaborn-v0_8')


pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Finds angle between two vectors"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector)

def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)

def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)
    
def visualize_3d(p3ds,p3dsf,capF,capS):

    """Now visualize in 3D"""
    torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']


    counter=1

    p3ds = p3ds-p3ds[0,10,:]

    ## commpute left knee angle
    #6-8 vs 8-10
    tmp=p3ds.shape
    Nframes=tmp[0]
    l_knee=np.zeros((Nframes,))
    r_knee=np.zeros((Nframes,))
    for i in np.arange(0,Nframes):
        l_knee[i]=angle_between(p3ds[i,6,:]-p3ds[i,8,:], p3ds[i,8,:]-p3ds[i,10,:])
        r_knee[i]=angle_between(p3ds[i,7,:]-p3ds[i,9,:], p3ds[i,9,:]-p3ds[i,11,:])

    fig0 = plt.figure()
    plt.plot(r_knee*180/np.pi,'r',label='right knee')
    plt.plot(l_knee*180/np.pi,'k',label='left knee')
    plt.xlabel('Frame Number')
    plt.ylabel('Angle [o]')
    plt.legend(loc="upper left")
    plt.ylim(0, 150)
    plt.pause(10)
    #plt.plot(p3ds[:,10,0],'b')
    #plt.plot(p3ds[:,10,1],'g')
    #plt.plot(p3ds[:,10,2],'r')
    



    from mpl_toolkits.mplot3d import Axes3D
    #startF=1
    #capF.set(cv.CAP_PROP_POS_FRAMES)
    #print('First frame of F = ',capF.get(cv.CAP_PROP_POS_FRAMES))   
    #capS.set(cv.CAP_PROP_POS_FRAMES)
    #print('First frame of S = ',capS.get(cv.CAP_PROP_POS_FRAMES))
    
    fig = plt.figure()
    axF = fig.add_subplot(221)
    axS = fig.add_subplot(222)
    ax = fig.add_subplot(223, projection='3d')
    ax2 = fig.add_subplot(224, projection='3d')

    
    #new_z = np.zeros((2,3))
    new_z = ((p3ds[0,0,:]-p3ds[0,10,:])+(p3ds[0,1,:]-p3ds[0,11,:]))/2
    alpha_z = angle_between(new_z, np.array([0.0, 0.0, 1.0]))
    new_z = z_rotation(new_z, -alpha_z)

    
    Mins = np.min(np.min(p3ds,0),0)-500
    Maxs = np.max(np.max(p3ds,0),0)+500
   
    
    for framenum, kpts3d in enumerate(p3ds):
        ret, frameF = capF.read()
        ret, frameS = capS.read()
        if framenum%2 == 0: continue #skip every 2nd frame
        axF.imshow(cv.cvtColor(frameF, cv.COLOR_BGR2RGB))
        axF.axis('off')
        axS.imshow(cv.cvtColor(frameS, cv.COLOR_BGR2RGB))
        axS.axis('off')

        ax.plot(np.linspace(0,new_z[0]),np.linspace(0,new_z[1]),np.linspace(0,new_z[2]))
        #ax.plot(np.linspace(0,new_vect[0]),np.linspace(0,new_vect[1]),np.linspace(0,new_vect[2]))


        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)
                ax2.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)

        #uncomment these if you want scatter plot of keypoints and their indices.
        for i in range(12):
            ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])
            #ax2.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
            ax2.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])
        
        #for _c in connections:
           # ax.plot(xs = [p3dsf[_c[0],0], p3dsf[_c[1],0]], ys = [p3dsf[_c[0],1], p3dsf[_c[1],1]], zs = [p3dsf[_c[0],2], p3dsf[_c[1],2]], c = 'red')
           # ax2.plot(xs = [p3dsf[_c[0],0], p3dsf[_c[1],0]], ys = [p3dsf[_c[0],1], p3dsf[_c[1],1]], zs = [p3dsf[_c[0],2], p3dsf[_c[1],2]], c = 'red')

        #ax.set_axis_off()
        #ax2.set_axis_off()
        ax2.set_xlim3d([Mins[0], Maxs[0]])
        ax2.set_ylim3d([Mins[1], Maxs[1]])
        ax2.set_zlim3d([Mins[2], Maxs[2]])
        ax2.view_init(elev=-78, azim=-91,roll=-0)
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_zticks([])

        ax.set_xlim3d([Mins[0], Maxs[0]])
        ax.set_ylim3d([Mins[1], Maxs[1]])
        ax.set_zlim3d([Mins[2], Maxs[2]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.view_init(elev=-160, azim=20,roll=-100)
        figname='.\Frames\Fig_'+str(counter)+'.png'
        plt.savefig(figname, bbox_inches='tight')
        plt.pause(0.05)
        ax.cla()
        ax2.cla()
        axF.cla()
        axS.cla()
        
        counter=counter+1

if __name__ == '__main__':
    Nvideo = '4'
    capF = cv.VideoCapture('.\Videos\Example_videoA'+Nvideo+'.avi')
    capS = cv.VideoCapture('.\Videos\Example_videoB'+Nvideo+'.avi')
    
    #npz = np.load('Floor_points.npz')
    #connections=npz['connections']
    #p3dsf=npz['p3dsf']
    p3ds = read_keypoints('.\Tracking\kpts_3d_'+Nvideo+'.dat')
    p3dsf = read_keypoints('.\Tracking\kpts_3d_'+Nvideo+'.dat')
    visualize_3d(p3ds,p3dsf,capF,capS)
    capF.release() 
    capS.release()