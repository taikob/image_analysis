import os, cv2, itertools, csv
import numpy as np

colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
            'yellow': [0, 255, 255], 'white': [255, 255, 255]}

def act(data):
    sum=0.0
    for d in data:
        if d > 0: sum += d

    num = np.count_nonzero(data > 0)
    if num != 0: ave = sum/num
    else: ave = 0

    return sum, ave, num

def sup(data):
    sum = 0.0
    for d in data:
        if d < 0: sum += d

    num = np.count_nonzero(data < 0)
    if num != 0: ave = sum/num
    else: ave = 0

    return sum, ave, num

def unit_vec(unitvec,vec,irt=0):
    val = unitvec[0] * vec[0] + unitvec[1] * vec[1]
    vec=np.array(vec)
    vecnorm = np.sqrt(np.sum(np.abs(vec**2))) * np.sqrt(np.sum(np.abs(unitvec**2)))

    if np.abs(val)/vecnorm < irt: val=0

    return val

def detect_rotmo(o,op,vec,ord=None,ird=None,ig=False):#dimension of o and op must be 2,o: origin, op: position from origin, ord: outer radius of circle, ird: inner radius of circle, ig: ignore excluding
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

    if ord is not None and r>ord: return 0
    if ird is not None and r<ird: return 0
    if r !=0 and norm < r:
        rotval=(vec[0]*opo[1] - vec[1]*opo[0]) /r
        if abs(rotval)/norm>=np.sqrt(2)/2 or ig is not False:
            return rotval/r
        else: return 0
    else: return 0

def save_flow_image(root, img, data, vs, met, cc = 'yellow', lc = 'red', s = 1, l = 2):
    mask = np.zeros_like(img)
    for para in data:
        c = para[0]
        d = para[1]
        dx = para[2] * vs
        dy = para[3] * vs
        norm = para[4]
        if norm == 0:
            cv2.line(mask, (int(c), int(d)), (int(c + dx), int(d + dy)), colormap['green'], l)
            cv2.line(img , (int(c), int(d)), (int(c + dx), int(d + dy)), colormap['green'], l)
        else:
            cv2.line(mask, (int(c), int(d)), (int(c + dx), int(d + dy)), colormap[lc], l)
            cv2.line(img , (int(c), int(d)), (int(c + dx), int(d + dy)), colormap[lc], l)
        cv2.circle(mask, (int(c), int(d)), s, colormap[cc], -1)
        cv2.circle(img , (int(c), int(d)), s, colormap[cc], -1)

    cv2.imwrite(os.path.join(root, 'vectors_' + met + '.png'), mask)
    cv2.imwrite(os.path.join(root, 'result_' + met + '.png'), img)

def save_flow_data(data, ofabs, root, met):
    actsum, actave, actnum = act(ofabs)
    supsum, supave, supnum = sup(ofabs)

    stdata = np.array([[actsum, actave, actnum, supsum, supave, supnum,
                       ofabs.sum(), ofabs.mean(), len(ofabs)]])

    np.savetxt(os.path.join(root, 'statdata_'+ met+'.csv'), stdata, delimiter=',')
    with open(os.path.join(root, 'data_' + met + '.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)

def lucas_kanade(img1, img2, cnt=None, nvec=None, irt=None, ord=None, ird=None, ig=False):
    window_size= 50
    quality_level= 0.3

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100, qualityLevel = quality_level,
                          minDistance = 7, blockSize = 7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize = (window_size,window_size),
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
            dx = a - c; dy = b - d
            if    cnt: val = detect_rotmo(cnt,[c,d],[dx,dy],ord=ord,ird=ird,ig=ig)
            elif nvec: val = unit_vec(nvec, [dx,dy], irt=irt)
            else:      val = np.sqrt(dx**2+dy**2)

            data.append([c, d, dx, dy,val])
            if val==0: continue
            else: ofabs = np.append(ofabs, val)

        return data,ofabs
    except:
        return None,None

def farneback(img1, img2, cnt=None, nvec=None, irt=None, ord=None, ird=None, ig=False):
    window_size= 10
    stride= 5
    min_vec= 0.01
    try:
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3,
                                            window_size, 3, 5, 1.2, 0)
        height, width = img1.shape

        data = []
        ofabs=np.ndarray(0)
        for x, y in itertools.product(range(0, width, stride),
                                      range(0, height, stride)):
            if np.linalg.norm(flow[y, x]) >= min_vec:
                dx, dy = flow[y, x].astype(float)
                if    cnt: val = detect_rotmo(cnt,[x,y],[dx,dy],ord=ord,ird=ird,ig=ig)
                elif nvec: val = unit_vec(nvec, [dx,dy], irt=irt)
                else:      val = np.sqrt(dx**2+dy**2)
                data.append([x, y, dx, dy, val])
                ofabs = np.append(ofabs, val)
        return data,ofabs
    except:
        return None,None

def get_optical_flow(root,file1, file2, met='lk',cc='yellow',lc='red',vs=None,s=1,l=2, cnt=False, nvec=False, irt=False, ord=None, ird=None, ig=False):
    #cnt: center of circle, nvec: unit vector, irt: ignore rate, rd: radius
    img1 = cv2.imread(os.path.join(root,file1))
    img2 = cv2.imread(os.path.join(root,file2))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if met=='lk':
        data, ofabs = lucas_kanade(img1_gray, img2_gray, cnt, nvec,irt,ord,ird,ig)
        if vs is None: vs=50
    elif met=='fb':
        data, ofabs = farneback(img1_gray, img2_gray, cnt, nvec,irt,ord,ird,ig)
        if vs is None: vs=4
    if data is None or ofabs is None:
        stdata=np.array([[0,0,0,0,0,0,0,0,0]])
        np.savetxt(os.path.join(root,'statdata_'+met+'.csv'),stdata, delimiter=',')
    else:
        save_flow_image(root, img1, data, vs, met, cc, lc, s, l)
        save_flow_data(data, ofabs, root, met)