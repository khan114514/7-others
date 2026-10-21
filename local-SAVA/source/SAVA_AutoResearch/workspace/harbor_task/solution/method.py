"""Canonical initial numerical method. All geometry is computed from inputs."""
import numpy as np
import torch
import ot
from geomloss import SamplesLoss
from scipy.spatial.distance import cdist

def _label_cost(x,y,r,ry):
    xx=torch.tensor(np.concatenate([x,r]),dtype=torch.float32)
    yy=np.concatenate([y,ry+10])
    W=np.zeros((20,20),dtype=np.float32)
    loss=SamplesLoss('sinkhorn',p=2,blur=np.sqrt(.1),debias=False,backend='tensorized')
    with torch.inference_mode():
        for i in range(20):
            for j in range(i+1,20):
                if not (yy==i).any() or not (yy==j).any():
                    raise ValueError('Initial batch lacks a label support')
                W[i,j]=W[j,i]=float(loss(xx[yy==i],xx[yy==j]))
    return W

def value(x,y,r,ry,seed=17,batch_size=1024):
    torch.set_num_threads(2)
    x=np.asarray(x,np.float32); r=np.asarray(r,np.float32)
    y=np.asarray(y,np.int64); ry=np.asarray(ry,np.int64)
    tx=[np.arange(i,min(i+batch_size,len(x))) for i in range(0,len(x),batch_size)]
    tr=[np.arange(i,min(i+batch_size,len(r))) for i in range(0,len(r),batch_size)]
    W=_label_cost(x[tx[0]],y[tx[0]],r[tr[0]],ry[tr[0]])
    def cost(z1,z2):
        d=torch.cdist(z1[:,:,:-1],z2[:,:,:-1]).square()/2
        return d+torch.as_tensor(W)[z1[:,:,-1].long().unsqueeze(2),z2[:,:,-1].long().unsqueeze(1)]/2
    potential=SamplesLoss('sinkhorn',p=2,blur=np.sqrt(.1),debias=False,
                         potentials=True,cost=cost,backend='tensorized')
    F={}; batchcost=np.zeros((len(tx),len(tr)))
    for i,idx in enumerate(tx):
        for j,jdx in enumerate(tr):
            a=x[idx];b=r[jdx]
            C=cdist(a,b,'sqeuclidean')/2+W[y[idx,None],ry[jdx][None,:]+10]/2
            P=ot.sinkhorn(ot.unif(len(idx)),ot.unif(len(jdx)),C/max(float(C.max()),1e-12),.1)
            batchcost[i,j]=float(np.sum(P*C))/(len(idx)+len(jdx))
            z1=torch.tensor(np.column_stack([a,y[idx]]),dtype=torch.float32)[None]
            z2=torch.tensor(np.column_stack([b,ry[jdx]+10]),dtype=torch.float32)[None]
            with torch.inference_mode(): f,_=potential(z1,z2)
            f=f.numpy().ravel().astype(float)
            F[i,j]=(f-f.mean())*len(f)/(len(f)-1)
    Q=ot.sinkhorn(ot.unif(len(tx)),ot.unif(len(tr)),
                  batchcost/max(float(batchcost.max()),1e-12),.01,
                  numItermax=10000,stopThr=1e-10)
    residual=max(np.max(np.abs(Q.sum(1)-ot.unif(len(tx)))),
                 np.max(np.abs(Q.sum(0)-ot.unif(len(tr)))))
    if residual>1e-6: raise ValueError(f'Unconverged batch transport: {residual}')
    values=np.zeros(len(x))
    for i,idx in enumerate(tx): values[idx]=sum(Q[i,j]*F[i,j] for j in range(len(tr)))
    return values
