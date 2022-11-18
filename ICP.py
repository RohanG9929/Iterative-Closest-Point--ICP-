import open3d as o3d
import copy
import numpy as np
from sklearn.neighbors import KDTree
from scipy.linalg import orthogonal_procrustes


demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

kitti_source = o3d.io.read_point_cloud('./Task2/kitti_frame1.pcd')
kitti_target = o3d.io.read_point_cloud('./Task2/kitti_frame2.pcd')


def findCorrespondances(src,dst):

    srTree = KDTree(src)
    dist, idxs = srTree.query(dst)

    a = np.array([(range(len(idxs)))])
    corr = np.hstack((idxs,a.T))
    
    #Indexes[i] provides the index  of the closest
    #point to dst[i] in src
    return corr,dist


def myICP(src,dst,corr):

    #Trimming src so that the number of points I am working
    #with is equal in the source and destination
    src = src[(corr[:,0])]
    dst = dst[(corr[:,1])]

    #Finding the Sum of D and S correspondance points
    n = corr.shape[0]
    ES = sum(src)
    ED = sum(dst)

    #Define new sets of corresponding points
    #Demeaning
    myS = src - ES/n
    myD = dst - ED/n

    # Finding R using Orthogonal Procrustes Problem 
    #Removing t from the cost function and using the 
    #Demeaned source and destination points to reduce the 
    #procrustes problem into an orthogonal procrustes problem
    #SUMi (di@si.transpose())
    R,scale = orthogonal_procrustes(myD,myS)

    #t = (SUM(Di) - R*SUM(Si))/n
    t = (ED-R@ES)/n

    transformation = np.vstack((np.hstack((R,np.array([t]).T)),np.zeros((1,4))))
    transformation[3,3] = 1

    #Computing the cost function with the updated R and t
    cost= np.power((np.linalg.norm(dst - (R@src.T).T - t)),2)
  
    return transformation,cost



def draw_registration_result(source, target, transformation):
    """
    param: source - source point cloud
    param: target - target point cloud
    param: transformation - 4 X 4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])


    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def performICPIterations(source,target,numIterations,threshold = 0.01):
    #Initialising variables
    initT = np.eye(4)                                   #Inital Transformation
    src = np.asarray(source.points)                     #Converting points to np arrays
    dst = np.asarray(target.points)
    runningSource = copy.deepcopy(source)               #Copying source to work on
    runningSource.transform(initT)                      #Applying initial T

    draw_registration_result(source, target, initT)     #Visualing oringal source
    oldCost = 0                                         #Setting init cost to 0
    runnningT = np.eye(4)                               #Transformation that will encompass all T's
                                                        #not just the recently applied T
    for i in range(numIterations):
        # print(i)
        #Finding the index of points in src(Source) that are the closest to the 
        #points in dst(Target)
        corr,_ = findCorrespondances(src,dst)

        #Finding the optimal translation and rotation
        trans,newCost = myICP(src,dst,corr)

        #Transforming the Point cloud with the new translation and rotation
        runningSource.transform(trans)
        src = np.asarray(runningSource.points)
        runnningT = trans@runnningT

        #Checking if the cost is below the threshold and breaking if its below the thresold
        # print(np.absolute(oldCost-newCost))

        if np.absolute(oldCost-newCost)<threshold:
            print("break")
            print(np.absolute(oldCost-newCost))
            print(i)
            break
        else:
            oldCost = (newCost)
        

    draw_registration_result(source, target, runnningT)
    return runnningT



homoT_after_refinement = performICPIterations(source,target,100)
print("Transformation from Original Source to Target")
print(homoT_after_refinement)