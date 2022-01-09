import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle as pkl

import geopandas as gpd

from geopandas import GeoSeries
from shapely.geometry import LineString,MultiLineString
import scipy.sparse as sp
import pandas as pd
import os

from scipy.sparse.linalg.eigen.arpack import eigsh

seed = 300
np.random.seed(seed)

def distance(p1, p2):
    return ((p1.X - p2.X) ** 2 + (p1.Y - p2.Y) ** 2) ** 0.5

def angle(p1,p2,mode=1):
    dx=p1.X-p2.X
    dy=p1.Y-p2.Y
    if dx==0.0 :
        if mode == 2:
            return 90.0
        else:
            if dy>0:
                return 90.0
            elif dy<0:
                return -90.0
            else:
                return 0.0
    else :
        if mode==1:
            return  np.arctan(dy / dx) / np.pi * 180.0
        elif mode==2:
            _angle = np.arctan(dy/dx)/np.pi*180.0
            if _angle>0:
                return _angle
            else:
                return _angle+180.0
        else:
            _angle = np.arctan(dy / dx) / np.pi * 180.0
            if (dx>0 and dy>0) or (dx>0 and dy<0):
                return _angle
            elif dx<0 and (dy<0 or dy ==0.0):
                return _angle-180.0
            elif dx<0 and (dy>0 or dy ==0.0):
                return _angle+180.0

def angle3(p1,p2,p3):
    a=distance(p1, p2)
    b=distance(p1, p3)
    c=distance(p2, p3)
    if (a*b==0.0):
        return 0
    return np.arccos((a**2+b**2-c**2)/(2*a*b))

class Point:
    def __init__(self, x, y):
        self.X = x
        self.Y = y

def get_linePointList (shapefilePath):
    lineShape = gpd.read_file (shapefilePath, encode='utf-8')
    lineList = lineShape['geometry']
    lineLengthList = lineList.length
    lineCoorList=[]
    firstLastPoints=[]
    lineSegmentCenterPoint = []
    for i in range(0,lineList.shape[0]):
        lineSegment = np.array(lineList.iat[i].xy).T
        firstPoint = Point(lineSegment[0][0], lineSegment[0][1])
        lastPoint = Point(lineSegment[-1][0], lineSegment[-1][1])
        lineCoorList.append(lineSegment)
        firstLastPoints.append((firstPoint,lastPoint))
        _centerXY = np.array(LineString(lineSegment).centroid.coords.xy)
        lineSegmentCenterPoint.append(Point(_centerXY[0],_centerXY[1]))
    centerXY = np.array(MultiLineString(lineCoorList).centroid.coords.xy)
    centerPoint = Point(centerXY[0],centerXY[1])
    return lineCoorList,lineLengthList.values,firstLastPoints,centerPoint,lineSegmentCenterPoint

def sinuousIndex(lineSegmentCoor):
    straightLength = LineString([(lineSegmentCoor[0][0], lineSegmentCoor[0][1]),
                                 (lineSegmentCoor[-1][0], lineSegmentCoor[-1][1])]).length
    if straightLength==0:
        straightLength = 0.00001
    curveLength = LineString(lineSegmentCoor).length
    return curveLength,curveLength/straightLength

def calMeander(lineSegmentCoor):
    if lineSegmentCoor.shape[0]<3:
        return 0
    rect = LineString(lineSegmentCoor).minimum_rotated_rectangle
    try:
        rectPoints = rect.exterior.coords.xy
    except:
        return 0
    length01 = LineString([(rectPoints[0][0],rectPoints[1][0]),(rectPoints[0][1], rectPoints[1][1])]).length
    length02 = LineString([(rectPoints[0][0],rectPoints[1][0]),(rectPoints[0][3], rectPoints[1][3])]).length
    if length01>length02:
        length=length01
        width=length02+0.00000001
    else:
        length=length02
        width=length01+0.00000001
    radius = ((length/2)**2+width**2)/(2*width)
    angleA = 2*np.arcsin((length/2)/radius)
    arcLength = angleA*radius
    return (LineString(lineSegmentCoor).length-arcLength)/arcLength

def calMinRectParam(lineSegmentCoor):
    if lineSegmentCoor.shape[0]<3:
        rectDirection = angle(Point(lineSegmentCoor[0][0],lineSegmentCoor[0][1]),Point(lineSegmentCoor[1][0], lineSegmentCoor[1][1]))
        lengthWidthRatio = 0
        return rectDirection,lengthWidthRatio
    rect = LineString(lineSegmentCoor).minimum_rotated_rectangle
    try:
        rectPoints = rect.exterior.coords.xy
    except:
        rectDirection = angle(Point(lineSegmentCoor[0][0],lineSegmentCoor[0][1]),Point(lineSegmentCoor[-1][0], lineSegmentCoor[-1][1]))
        lengthWidthRatio = 0
        return rectDirection,lengthWidthRatio
    length01 = LineString([(rectPoints[0][0],rectPoints[1][0]),(rectPoints[0][1], rectPoints[1][1])]).length
    length02 = LineString([(rectPoints[0][0],rectPoints[1][0]),(rectPoints[0][3], rectPoints[1][3])]).length
    if length01>length02:
        rectDirection = angle(Point(rectPoints[0][0],rectPoints[1][0]),Point(rectPoints[0][1], rectPoints[1][1]))
        lengthWidthRatio = length02/length01
    else:
        rectDirection = angle(Point(rectPoints[0][0],rectPoints[1][0]),Point(rectPoints[0][2], rectPoints[1][2]))
        lengthWidthRatio = length01/length02
    return rectDirection,lengthWidthRatio

def staticAngle(lineSegmentCoor):
    if len(lineSegmentCoor)<3:
        return 0
    # angleCount = np.zeros(18)
    eachAngleLengthCount = np.zeros(18)
    for i in range(len(lineSegmentCoor)-1):
        firstPoint = Point(lineSegmentCoor[i][0], lineSegmentCoor[i][1])
        secondPoint = Point(lineSegmentCoor[i+1][0], lineSegmentCoor[i+1][1])
        direction = angle(firstPoint,secondPoint,mode=2)
        angleGroupIndex = int((direction-0.00001)//10)
        # angleCount[angleGroupIndex]+=1
        eachAngleLengthCount[angleGroupIndex]+=distance(firstPoint, secondPoint)
    # angleCount = angleCount*eachAngleLengthCount#这一步乘以数量是不合理的，本来就用长度衡量了，怎么还乘以数量，王米琪是因为剖分了所以用数量，此处无剖分所以用长度
    maxAngle1Index=eachAngleLengthCount.argsort()[::-1][0]
    maxAngle2Index=eachAngleLengthCount.argsort()[::-1][1]
    maxAngle1 = maxAngle1Index*10+5
    maxAngle2 = maxAngle2Index*10+5
    maxAngle1Length = eachAngleLengthCount[maxAngle1Index]
    maxAngle2Length = eachAngleLengthCount[maxAngle2Index]
    angleRatio = eachAngleLengthCount/np.sum(eachAngleLengthCount)
    weightCount=np.zeros(18)
    for i in range(18):
        # weightCount[i]=max((np.cos(np.abs(i*10+5-maxAngle1)/180*np.pi)+1)/2,
        #                     (np.cos(np.abs(i*10+5-maxAngle2)/180*np.pi)+1)/2)
        weightCount[i]=max(np.abs(np.cos((i*10+5-maxAngle1)/180*np.pi)),
                            np.abs(np.cos((i*10+5-maxAngle2)/180*np.pi)))
    if maxAngle1Length == 0 or maxAngle2Length == 0:
        return 0
    lengthRatio = maxAngle1Length/maxAngle2Length
    if lengthRatio>1:
        lengthRatio=1/lengthRatio
        #方向分布指数是为了证明曲线的所有分段是否完全集中与两个主要方向中，
        # 然后结合正交指数表达这两个方向的正交性，
        # 因为可以将方向指数*正交指数*两个方向的长度比例，越接近1者正交性越强
    return np.dot(angleRatio,weightCount)*np.sin(np.abs(maxAngle1-maxAngle2)/180*np.pi)*lengthRatio

def calRiverTopology(shapefilePath):
    riverNetwork = gpd.read_file(shapefilePath)
    outletIndex = int(riverNetwork[riverNetwork['outlet']==1].index.values)
    riverList = list(range(riverNetwork.shape[0]))
    point1 = np.array(riverNetwork.loc[outletIndex,'geometry'])[0]
    point2 = np.array(riverNetwork.loc[outletIndex,'geometry'])[-1]
    select_riverList = [outletIndex]
    riverTopology = dict()
    currentIndex=outletIndex
    groupedIndexList=[currentIndex]
    for i in riverList:
        riverTopology[i]=[[],[]]
    while len(select_riverList)>0:
        riverList.remove(currentIndex)
        for i in riverList:
            if i in select_riverList or i in groupedIndexList:
                continue
            if(np.around(point1,decimals=8).tolist()
               in np.around(np.array(riverNetwork.loc[i,'geometry']),decimals=8).tolist()) \
                    or (np.around(point2,decimals=8).tolist()
                        in np.around(np.array(riverNetwork.loc[i,'geometry']),decimals=8).tolist()):
                riverTopology[currentIndex][0].append(i)
                riverTopology[i][1].append(currentIndex)
        select_riverList.remove(currentIndex)
        select_riverList+=riverTopology[currentIndex][0]
        if len(select_riverList)>0:
            currentIndex = select_riverList[0]
            groupedIndexList.append(currentIndex)
            point1 = np.array(riverNetwork.loc[currentIndex,'geometry'])[0]
            point2 = np.array(riverNetwork.loc[currentIndex,'geometry'])[-1]
    for i in riverTopology[outletIndex][0]:
        if np.around(np.array(riverNetwork.loc[outletIndex, 'geometry'])[0],decimals=8).tolist() \
                in np.around(np.array(riverNetwork.loc[i, 'geometry']),decimals=8).tolist():
            outletXY = np.array(riverNetwork.loc[outletIndex, 'geometry'])[-1]
            break
        else:
            outletXY = np.array(riverNetwork.loc[outletIndex, 'geometry'])[0]
            break
    outletPoint = Point(outletXY[0], outletXY[1])
    return riverTopology,riverNetwork,outletPoint

def calHortonCode(riverNetwork,riverTopology):
    hortonCode = np.zeros(riverNetwork.shape[0])
    unCodeList = list(range(riverNetwork.shape[0]))
    codeList = []
    downStreamList=[]
    for i in range(riverNetwork.shape[0]):
        if riverTopology[i][0]==[]:
            hortonCode[i]=1
            unCodeList.remove(i)
            codeList.append(i)
            downStreamList+=riverTopology[i][1]
    #如果所有上游都已经进行编码，则按照编码规则进行编码，相同编码则+1，不同编码则选择最大编码
    downStreamList = list(set(downStreamList))
    while len(downStreamList)>0:
        downStreamListCopy = downStreamList.copy()
        for i in downStreamListCopy:
            upHortonCode=[]
            if not set(riverTopology[i][0])<=set(codeList):
                continue
            for j in riverTopology[i][0]:
                upHortonCode.append(hortonCode[j])
            maxHortonCode = max(upHortonCode)
            if len(upHortonCode)>1 and sum(upHortonCode==maxHortonCode)==len(upHortonCode):
                hortonCode[i] = maxHortonCode+1
            else:
                hortonCode[i]=maxHortonCode
            codeList.append(i)
            downStreamList.remove(i)
            downStreamList+=riverTopology[i][1]
            downStreamList = list(set(downStreamList))
    return hortonCode

def input_data(num, path):
    all_data = []
    all_adj = []
    max_nums = 0
    for i in range(num):
        print(path + str(i) + '.shp')
        # if not os.path.isfile(path + str(i+300) + '.shp'):
        #     break
        lineCoorList,lineLengthList,firstLastPoints,centerPoint,lineSegmentCenterPoint = get_linePointList(path + str(i) + '.shp')
        # df_shape = gpd.read_file(path + str(i) + '.shp', encode='utf-8')
        # lst = df_shape['geometry']
        riverTopology, riverNetwork, outletPoint = calRiverTopology(path + str(i) + '.shp')
        hortonCode = calHortonCode(riverNetwork, riverTopology)
        center = outletPoint
        point = firstLastPoints
        curveLength=[]
        sinuousParam=[]
        rectDirection=[]
        lengthWidthRatio=[]
        centerDistanceSub=[]
        centerDistanceAdd= []
        centerDirection=[]
        centerAngle = []
        directionDistributionIndex=[]
        orthogonalityIndex=[]
        meanderIndex = []
        if(len(lineCoorList)>max_nums):
            max_nums = len(lineCoorList)
        shape = len(lineCoorList)
        adj = np.zeros(shape=(shape, shape))
        edge = []
        for j in range(len(lineCoorList)):
            _curveLength,_sinuousParam = sinuousIndex(lineCoorList[j])
            curveLength.append(_curveLength)
            sinuousParam.append(_sinuousParam)
            _rectDirection,_lengthWidthRatio = calMinRectParam(lineCoorList[j])
            rectDirection.append(_rectDirection)
            lengthWidthRatio.append(_lengthWidthRatio)
            # _direction.append(angle(point[j][0], point[j][1]))
            # centerDistanceSub.append(0.5 * (distance(point[j][0], center) + distance(point[j][1], center)))
            centerDistanceSub.append(abs(distance(point[j][0], center) - distance(point[j][1], center)))
            centerDistanceAdd.append(distance(point[j][0], center) + distance(point[j][1], center))
            centerDirection.append(angle(center,lineSegmentCenterPoint[j],mode=3))
            # centerAngle.append(angle3(center,point[j][0],point[j][1]))
            # _directionDistributionIndex,_orthogonalityIndex = staticAngle(lineCoorList[j])
            # directionDistributionIndex.append(_directionDistributionIndex)
            _orthogonalityIndex = staticAngle(lineCoorList[j])
            orthogonalityIndex.append(_orthogonalityIndex)
            meanderIndex.append(calMeander(lineCoorList[j]))
            for k in range(j+1, len(lineCoorList)): #这里应该使用首尾点
                if ((point[k][0].X == point[j][0].X and point[k][0].Y == point[j][0].Y)
                        or (point[k][0].X == point[j][1].X and point[k][0].Y == point[j][1].Y)
                        or (point[k][1].X == point[j][0].X and point[k][1].Y== point[j][0].Y)
                        or (point[k][1].X == point[j][1].X and point[k][1].Y == point[j][1].Y)):
                    adj[k][j] += 1
                    adj[j][k] += 1
                    edge.append((j, k))
        nodeDegree = np.sum(adj, axis=0)
    # <editor-fold desc="特征指标归一化">
        scaler = preprocessing.StandardScaler()
        nodeDegree = scaler.fit_transform(nodeDegree.reshape(-1, 1))
        hortonCode = scaler.fit_transform(np.array(hortonCode).reshape(-1, 1))
        centerDistanceSub = scaler.fit_transform(np.array(centerDistanceSub).reshape(-1, 1))
        centerDistanceAdd = scaler.fit_transform(np.array(centerDistanceAdd).reshape(-1, 1))
        # centerDirection = scaler.fit_transform(np.array(centerDirection).reshape(-1, 1))
        centerDirection = (np.array(centerDirection).reshape(-1, 1)/180).astype(np.float64)
        # centerAngle = scaler.fit_transform(np.array(centerAngle).reshape(-1, 1))
        curveLength = scaler.fit_transform(np.array(curveLength).reshape(-1, 1))
        sinuousParam = scaler.fit_transform(np.array(sinuousParam).reshape(-1, 1))
        lengthWidthRatio = scaler.fit_transform(np.array(lengthWidthRatio).reshape(-1, 1))
        meanderIndex = scaler.fit_transform(np.array(meanderIndex).reshape(-1, 1))
        rectDirection = scaler.fit_transform(np.array(rectDirection).reshape(-1, 1))
        # rectDirection2 = (np.array(rectDirection).reshape(-1, 1) / 180).astype(np.float64)
        # directionDistributionIndex = scaler.fit_transform(np.array(directionDistributionIndex).reshape(-1, 1))
        orthogonalityIndex = scaler.fit_transform(np.array(orthogonalityIndex).reshape(-1, 1))
    # </editor-fold>
        data = np.array([nodeDegree, hortonCode, centerDistanceSub,
                         centerDirection,curveLength,sinuousParam,
                         lengthWidthRatio,meanderIndex,rectDirection,
                         orthogonalityIndex]).T
        # data = np.array([nodeDegree, hortonCode, centerDistanceSub,centerDistanceAdd,
        #                  centerDirection,curveLength,sinuousParam,lengthWidthRatio,
        #                  meanderIndex,rectDirection, orthogonalityIndex]).T
        data = data.reshape(data.shape[1], data.shape[2])
        all_data.append(data)
        all_adj.append(adj)
    return all_data, all_adj, max_nums

def input_all_data(num,chebyshev_p, max_nums=None,readShapeFile=False,featureGroup=None,loadTestData = False,_test_size=0.3, dataName ='all_data.npy'):
    if readShapeFile:
        all_dendritic, dendritic_adj, dendritic_nums = input_data(num, './repaired_data/dendritic/')
        all_distributary, distributary_adj, distributary_nums = input_data(num, './repaired_data/distributary/')
        all_parallel, parallel_adj, parallel_nums = input_data(num, './repaired_data/parallel/')
        all_trellis, trellis_adj, trellis_nums = input_data(num, './repaired_data/trellis/')
        all_rectangle, rectangle_adj, rectangle_nums = input_data(num, './repaired_data/rectangle/')
        if max_nums==None:
            max_nums = max(parallel_nums, dendritic_nums, distributary_nums, trellis_nums,rectangle_nums)
        print('最大节点数为：',str(max_nums))
        all_data = all_dendritic + all_distributary + all_parallel + all_rectangle + all_trellis
        all_adj = dendritic_adj + distributary_adj + parallel_adj + rectangle_adj + trellis_adj
        for i in range(len(all_data)):
            shape = all_data[i].shape
            all_data[i] = np.pad(all_data[i], pad_width=((0, max_nums-shape[0]), (0, 0)), mode='constant')
            if(i<num):
                all_data[i] = [all_data[i], np.array([1, 0, 0, 0, 0])]
            elif (i < 2 * num):
                all_data[i] = [all_data[i], np.array([0, 1, 0, 0, 0])]
            elif (i < 3 * num):
                all_data[i] = [all_data[i], np.array([0, 0, 1, 0, 0])]
            elif(i < 4 * num):
                all_data[i] = [all_data[i], np.array([0, 0, 0, 1, 0])]
            else:
                all_data[i] = [all_data[i], np.array([0, 0, 0, 0, 1])]
            all_adj[i] = np.pad(all_adj[i], pad_width=((0, max_nums-shape[0]), (0, max_nums-shape[0])))
            all_data[i].append(chebyshev_polynomials(all_adj[i], chebyshev_p))
            all_data[i].append(i)
        # <editor-fold desc="统计测试用的数据特征">
        # for i in range(len(all_data)):
        #     shape = all_data[i].shape
        #     all_data[i] = np.pad(all_data[i], pad_width=((0, max_nums-shape[0]), (0, 0)), mode='constant')
        #     if(i<18):
        #         all_data[i] = [all_data[i], np.array([1,0, 0, 0, 0])]
        #     elif (i < 18+218):
        #         all_data[i] = [all_data[i], np.array([0, 1, 0, 0, 0])]
        #     elif (i < 18+218+27):
        #         all_data[i] = [all_data[i], np.array([0, 0, 1, 0, 0])]
        #     elif(i < 18+218+27+124):
        #         all_data[i] = [all_data[i], np.array([0, 0, 0, 1, 0])]
        #     else:
        #         all_data[i] = [all_data[i], np.array([0, 0, 0, 0, 1])]
        #     all_adj[i] = np.pad(all_adj[i], pad_width=((0, max_nums-shape[0]), (0, max_nums-shape[0])))
        #     all_data[i].append(chebyshev_polynomials(all_adj[i], chebyshev_p))
        #     all_data[i].append(i)
            # </editor-fold desc="特征指标归一化">
        np.save(dataName,all_data)
        # np.save('test_data.npy', all_data)
    else:
        if loadTestData:
            all_data = np.load('valid_data.npy',allow_pickle=True)
        else:
            all_data = np.load(dataName,allow_pickle=True)
        # featureDict = dict(nodeDegree=0, hortonCode=1, centerDistanceSub=2,centerDistanceAdd=3,centerDirection=4,
        #                curveLength=5,sinuousParam=6,lengthWidthRatio=7,meanderIndex=8, rectDirection=9,
        #                orthogonalityIndex=10)
        featureDict = dict(nodeDegree=0, hortonCode=1, centerDistance=2, centerDirection=3,
                           curveLength=4, sinuousParam=5, lengthWidthRatio=6, meanderIndex=7,
                           rectDirection=8, orthogonalityIndex=9)
        featureIndex=[]
        for featureName in featureGroup:
            featureIndex.append(int(featureDict[featureName]))
        for dataIndex in range(len(all_data)):
            all_data[dataIndex][0]=all_data[dataIndex][0][:,featureIndex]
    # train_X, test_X = train_test_split(all_data, test_size=0.3)
    #为了将每种类型的河网模式都是均匀划分7：3的训练和测试集
    if loadTestData:
        train_X, test_X = train_test_split(all_data, test_size=_test_size)
    else:
        firstNum=0
        for i in range(5):
            lastNum = (i+1)*num
            # np.random.shuffle(all_data[firstNum:lastNum])
            _train_X, _test_X = train_test_split(all_data[firstNum:lastNum], test_size=_test_size)
            if i==0:
                train_X = _train_X
                test_X = _test_X
            else:
                train_X = np.vstack((train_X,_train_X))
                test_X = np.vstack((test_X, _test_X))
            firstNum=lastNum
    np.random.shuffle(train_X)
    np.random.shuffle(test_X)
    test_index = []
    feature = []
    train_y = []
    test_feature = []
    test_y = []
    train_support = []
    test_support = []
    for i in range(len(train_X)):
        feature.append(train_X[i][0])
        train_y.append(train_X[i][1])
        train_support.append(train_X[i][2])
    for i in range(len(test_X)):
        test_feature.append(test_X[i][0])
        test_y.append(test_X[i][1])
        test_support.append(test_X[i][2])
        test_index.append(test_X[i][3])
    pd.DataFrame(test_index).to_csv('test.csv')
    return feature, np.array(train_y), train_support, test_feature, np.array(test_y), test_support, test_index

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        cords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return cords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1), dtype='float64')
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    a = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return a

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        # s_lap = scaled_lap
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def saveTrainData(num=[300,300,300,300,300],chebyshev_p=1,max_nums=None,saveDataName=None):
    all_dendritic, dendritic_adj, dendritic_nums = input_data(num[0], './repaired_data/dendritic/')
    all_distributary, distributary_adj, distributary_nums = input_data(num[1], './repaired_data/distributary/')
    all_parallel, parallel_adj, parallel_nums = input_data(num[2], './repaired_data/parallel/')
    all_rectangle, rectangle_adj, rectangle_nums = input_data(num[3], './repaired_data/rectangle/')
    all_trellis, trellis_adj, trellis_nums = input_data(num[4], './repaired_data/trellis/')
    if max_nums==None:
        max_nums = max(parallel_nums, dendritic_nums, distributary_nums, trellis_nums,rectangle_nums)
    print('最大节点数为：', str(max_nums))
    all_data = all_dendritic + all_distributary + all_parallel + all_rectangle + all_trellis
    all_adj = dendritic_adj + distributary_adj + parallel_adj + rectangle_adj + trellis_adj
    for i in range(len(all_data)):
        shape = all_data[i].shape
        all_data[i] = np.pad(all_data[i], pad_width=((0, max_nums - shape[0]), (0, 0)), mode='constant')
        if (i < num[0]):
            all_data[i] = [all_data[i], np.array([1, 0, 0, 0, 0])]
        elif (i < num[0]+num[1]):
            all_data[i] = [all_data[i], np.array([0, 1, 0, 0, 0])]
        elif (i < num[0]+num[1]+num[2]):
            all_data[i] = [all_data[i], np.array([0, 0, 1, 0, 0])]
        elif (i < num[0]+num[1]+num[2]+num[3]):
            all_data[i] = [all_data[i], np.array([0, 0, 0, 1, 0])]
        else:
            all_data[i] = [all_data[i], np.array([0, 0, 0, 0, 1])]
        all_adj[i] = np.pad(all_adj[i], pad_width=((0, max_nums - shape[0]), (0, max_nums - shape[0])))
        all_data[i].append(chebyshev_polynomials(all_adj[i], chebyshev_p))
        all_data[i].append(i)
    np.save(saveDataName,all_data)
    return all_data