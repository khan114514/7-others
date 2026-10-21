"""Expert Reference: weighted conditional-support compression."""
import numpy as np
import torch
import ot
from geomloss import SamplesLoss
from scipy.spatial.distance import cdist

def _coreset(x,k):
    if len(x)<=k: return x,ot.unif(len(x))
    # Deterministic farthest-first supports followed by Voronoi weighted means.
    chosen=[int(np.argmin(np.square(x-x.mean(0)).sum(1)))]
    nearest=np.square(x-x[chosen[0]]).sum(1)
    for _ in range(1,k):
        j=int(np.argmax(nearest));chosen.append(j)
        nearest=np.minimum(nearest,np.square(x-x[j]).sum(1))
    centers=x[chosen].copy()
    for _ in range(3):
        labels=cdist(x,centers,'sqeuclidean').argmin(1)
        counts=np.bincount(labels,minlength=k)
        for j in range(k):
            if counts[j]: centers[j]=x[labels==j].mean(0)
    keep=counts>0
    return centers[keep],counts[keep]/counts.sum()

def _label_cost(x,y,r,ry,coreset_size=None):
    # Official cached label-to-label distances from the first pair of batches.
    # Official exact means empirical Sinkhorn, not the Gaussian approximation.
    xx=torch.tensor(np.concatenate([x,r]),dtype=torch.float32)
    yy=np.concatenate([y,ry+10])
    W=np.zeros((20,20),dtype=np.float32)
    loss=SamplesLoss('sinkhorn',p=2,blur=np.sqrt(.1),debias=False,backend='tensorized')
    supports=[]
    for i in range(20):
        raw=xx[yy==i]
        if len(raw)==0:raise ValueError('Pilot cached first batch lacks class support')
        if coreset_size is None:supports.append((torch.ones(len(raw))/len(raw),raw))
        else:
            z,w=_coreset(raw.numpy(),coreset_size)
            supports.append((torch.tensor(w,dtype=torch.float32),torch.tensor(z,dtype=torch.float32)))
    with torch.inference_mode():
        for i in range(10):
            for j in range(10,20):
                a,z=supports[i];b,zz=supports[j]
                W[i,j]=W[j,i]=float(loss(a,z,b,zz))
    return W

def value(x,y,r,ry,seed=17,batch_size=1024,coreset_size=48):
    torch.set_num_threads(2)
    x=np.asarray(x,np.float32); r=np.asarray(r,np.float32)
    y=np.asarray(y,np.int64); ry=np.asarray(ry,np.int64)
    tx=[np.arange(i,min(i+batch_size,len(x))) for i in range(0,len(x),batch_size)]
    tr=[np.arange(i,min(i+batch_size,len(r))) for i in range(0,len(r),batch_size)]
    W=_label_cost(x[tx[0]],y[tx[0]],r[tr[0]],ry[tr[0]],coreset_size=coreset_size)
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
            # Evaluate the two coupled stages from one pairwise ground geometry.
            # The point-potential stage retains its original absolute epsilon.
            frozen_cost=torch.as_tensor(C,dtype=torch.float32)[None]
            def pair_cost(left,right):
                return frozen_cost.transpose(1,2) if left[0,0,-1]>=10 else frozen_cost
            pair_potential=SamplesLoss('sinkhorn',p=2,blur=np.sqrt(.1),debias=False,
                                      potentials=True,cost=pair_cost,backend='tensorized')
            with torch.inference_mode(): f,_=pair_potential(z1,z2)
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
