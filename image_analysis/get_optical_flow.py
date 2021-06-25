import optical_flow as op
import os
from natsort import natsorted

#rootpathlist=['images/Illusions','images/Painting_Modern','images/Painting_Realistic','images/Photo_Movable','images/Photo_Still']
#rootpathlist=['images/one_color','images/two_color']
rootpathlist=['images']
size=[160,120]
center = None#[(size[0] - 1) / 2, (size[1] - 1) / 2]
met='fb'

for rootpath in rootpathlist:
    for savedir in natsorted(os.listdir(rootpath)):
        savedir = rootpath + '/' + savedir
        if os.path.exists(savedir + '/runtime.txt'):
            print(savedir)
            op.get_optical_flow(savedir+'/images', 'test_20y_0.jpg', 'test_20y_1.jpg', met, dtct_rm=center)