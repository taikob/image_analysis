import os, cv2, itertools, csv
import numpy as np

colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
            'yellow': [0, 255, 255], 'white': [255, 255, 255]}

def actsum(data):
    sum=0.0
    for d in data:
        if d > 0: sum += d
    return sum

def supsum(data):
    sum = 0.0
    for d in data:
        if d < 0: sum += d
    return sum

def unit_vec(unitvec,vec,per=0):

    norm = unitvec[0] * vec[0] + unitvec[1] * vec[1]
    vecnorm = np.sqrt(np.sum(np.abs(vec**2)))
    if norm/vecnorm<per:
        norm=0

    return norm

def detect_rotmo(o,op,vec):#dimension of o and op must be 2
    #fix Inverted y axis for cv2 to correct axis
    o   = [  o[0], -   o[1]]
    op  = [ op[0], -  op[1]]
    vec = [vec[0], - vec[1]]

    opo=np.zeros(len(op))

    r=0
    norm=0
    for i in range(len(op)):
        opo[i]=op[i]-o[i]
        r    +=opo[i]**2
        norm +=vec[i]**2

    #The direction of rotation is CW (not CCW)
    r   =np.sqrt(r)
    norm=np.sqrt(norm)
    if r !=0 and norm < r:
        rotnorm=(vec[0]*opo[1] - vec[1]*opo[0]) /r
        if abs(rotnorm)/norm>=np.sqrt(2)/2:
            return rotnorm
        else:
            return 0
    else:
        return 0

def save_flow_image(root, img, data, vs, met, cc = 'yellow', lc = 'red', s = 1, l = 2):
    mask = np.zeros_like(img)
    for para in data:
        c = para[0]
        d = para[1]
        dx = para[2] * vs
        dy = para[3] * vs
        norm = para[4]
        if norm == 0:
            cv2.line(mask, (c, d), (int(c + dx), int(d + dy)), colormap['green'], l)
            cv2.line(img , (c, d), (int(c + dx), int(d + dy)), colormap['green'], l)
        else:
            cv2.line(mask, (c, d), (int(c + dx), int(d + dy)), colormap[lc], l)
            cv2.line(img , (c, d), (int(c + dx), int(d + dy)), colormap[lc], l)
        cv2.circle(mask, (c, d), s, colormap[cc], -1)
        cv2.circle(img , (c, d), s, colormap[cc], -1)

    cv2.imwrite(os.path.join(root, 'vectors_' + met + '.jpg'), mask)
    cv2.imwrite(os.path.join(root, 'result_' + met + '.jpg'), img)

def save_flow_data(data, ofabs, root, met):
    if np.count_nonzero(ofabs > 0) != 0:
        aveact = actsum(ofabs) / np.count_nonzero(ofabs > 0)
    else:
        aveact = 0
    if np.count_nonzero(ofabs < 0) != 0:
        avesup = supsum(ofabs) / np.count_nonzero(ofabs < 0)
    else:
        avesup = 0
    stdata = np.array([actsum(ofabs), aveact, np.count_nonzero(ofabs > 0),
                       supsum(ofabs), avesup, np.count_nonzero(ofabs < 0),
                       ofabs.sum(), ofabs.mean(), len(ofabs)])

    np.savetxt(os.path.join(root, 'statdata_'+ met+'.csv'), stdata)
    with open(os.path.join(root, 'data_' + met + '.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)

def lucas_kanade(img1, img2, dtct_rm=None, unitvec=None,per=None):
    window_size= 50
    quality_level= 0.3

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = quality_level,
                          minDistance = 7,
                          blockSize = 7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize = (window_size,
                                window_size),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]

        data = []
        ofabs=np.ndarray(0)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            dx = a - c
            dy = b - d
            if dtct_rm:
                norm=detect_rotmo(dtct_rm,[c,d],[dx,dy])
            elif unitvec is not None:
                norm=unit_vec(unitvec, [dx,dy], per=per)
            else:
                norm=np.sqrt(dx**2+dy**2)
            data.append([c, d, dx, dy,norm])
            if norm==0: continue
            else: ofabs = np.append(ofabs, norm)

        return data,ofabs
    except:
        return None,None

def farneback(img1, img2, dtct_rm=None, unitvec=None, per=None):
    window_size= 10
    stride= 5
    min_vec= 0.01
    try:
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3,
                                            window_size,
                                            3, 5, 1.2, 0)
        height, width = img1.shape

        data = []
        ofabs=np.ndarray(0)
        for x, y in itertools.product(range(0, width, stride),
                                      range(0, height, stride)):
            if np.linalg.norm(flow[y, x]) >= min_vec:
                dx, dy = flow[y, x].astype(float)
                if dtct_rm is not None:
                    norm=detect_rotmo(dtct_rm,[x,y],[dx,dy])
                elif unitvec is not None:
                    norm=unit_vec(unitvec, [dx,dy], per=per)
                else:
                    norm=np.sqrt(dx**2+dy**2)
                data.append([x, y, dx, dy, norm])
                ofabs = np.append(ofabs, norm)
        return data,ofabs
    except:
        return None,None

def get_optical_flow(root,file1, file2, met='lk',cc='yellow',lc='red',vs=None,s=1,l=2, dtct_rm=False, unitvec=False, per=False):
    img1 = cv2.imread(os.path.join(root,file1))
    img2 = cv2.imread(os.path.join(root,file2))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if met=='lk':
        data,ofabs=lucas_kanade(img1_gray, img2_gray, dtct_rm, unitvec,per)
        if vs is None: vs=50
    elif met=='fb':
        data,ofabs=farneback(img1_gray, img2_gray, dtct_rm, unitvec,per)
        if vs is None: vs=4
    if data is None or ofabs is None:
        stdata=np.array([0,0,0,0,0,0,0,0])
        np.savetxt(os.path.join(root,'statdata_'+met+'.csv'),stdata)
    else:
        save_flow_image(root, img1, data, vs, met, cc, lc, s, l)
        save_flow_data(data, ofabs, root, met)