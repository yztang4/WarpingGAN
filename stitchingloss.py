import torch

def pc_distmat(x):
    batchsize = x.size(0)
    m = x.size(1)
    xx = torch.pow(x,2).sum(2, keepdim=True).expand(batchsize,m,m)
    yy = xx.transpose(2,1)
    inner = torch.matmul(x,x.transpose(2,1))
    distance = xx+yy-2*inner
    # print(distance)
    return distance

def stitchloss(point, indices):
    batchsize = point.size(0)
    dismat = pc_distmat(point)
    keypointvariance = 0
    for i in range(batchsize):
        tmpindexes = torch.unique(indices[i].reshape(512))
        tmp = torch.index_select(dismat[i], 0, tmpindexes)
        tmp = -tmp 
        min20distances,_ = tmp.topk(40,dim=1)
        var11 = torch.var(-min20distances,1)
        keypointvariance += var11.mean()

    return keypointvariance



