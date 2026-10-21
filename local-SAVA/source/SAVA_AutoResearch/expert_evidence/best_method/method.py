"""Hierarchical transport with reusable costs and explicit annealed dual solves."""
import numpy as np
import torch
import ot


def _schedule(a, b):
    lo = torch.minimum(a.amin(0), b.amin(0))
    hi = torch.maximum(a.amax(0), b.amax(0))
    diameter = float(torch.linalg.vector_norm(hi - lo))
    return [diameter**2] + [float(np.exp(e)) for e in np.arange(2*np.log(diameter), np.log(.1), 2*np.log(.5))] + [.1]


def _dual(C, schedule, initial):
    n,m=C.shape
    la=float(np.log(np.float32(1./n)));lb=float(np.log(np.float32(1./m)))
    shift=0.
    def direct(f,g,eps):
        R=C-f[:,None];R-=g[None,:]
        if eps<=2.:
            return f+R.amin(1)-eps*lb,g+R.amin(0)-eps*la,None,0.,None,None
        if step==0 and (float(f.max())+float(g.max()))/eps<60.:
            rm=0.
            R.mul_(-1.4426950408889634/eps).exp2_()
        else:
            rm=float(R.min());R-=rm;R.mul_(-1.4426950408889634/eps).exp2_()
        rs=R.sum(1);cs=(R.T @ torch.ones((n,1)))[:,0]
        if min(float(rs.min()),float(cs.min()))<=2e-22:
            R=C-f[:,None];R-=g[None,:];R*=(-1./eps)
            rowmax=R.amax(1);colmax=R.amax(0)
            rr=torch.exp(R-rowmax[:,None]).sum(1);cc=torch.exp(R-colmax[None,:]).sum(0)
            return f-eps*(rowmax+rr.log()+lb),g-eps*(colmax+cc.log()+la),None,0.,None,None
        return f+rm-eps*(rs.log()+lb),g+rm-eps*(cs.log()+la),R,rm,rs,cs
    f,g=initial
    
    K=None;rs=None;cs=None
    restore=None
    pe=schedule[0]
    point_steps=schedule[3:-3]
    for step,eps in enumerate(point_steps):
        last=step==len(point_steps)-1
        if last and (K is None or eps<=2.):
            if eps<=2.:
                C.sub_(f[:,None]);restore=f
                g=(g+C.amin(0)-eps*la)*.5
            else:g=(g-eps*(((f[:,None]-C)/eps).logsumexp(0)+la))*.5
            break
        if K is None or eps<=2.:
            ft,gt,K,shift,rs,cs=direct(f,g,eps)
        else:
            if last:
                q=pe/eps
                logfactor=.5*q*(float(np.log(n))-cs.log())
                K.mul_((m/rs).sqrt()[:,None])
                if abs(q-4.)<1e-5:K.square_().square_()
                else:K.pow_(q)
                sums=(K.T @ torch.ones((n,1)))[:,0]
                if float(sums.min())>0.:
                    logmass=sums.log()+logfactor
                    g.add_(logmass+la,alpha=-.5*eps)
                else:g=(g-eps*(((f[:,None]-C)/eps).logsumexp(0)+la))*.5
                break
            K.mul_((m/rs).sqrt()[:,None]).mul_((n/cs).sqrt()[None,:])
            if abs(pe/eps-4.)<1e-5:
                K.square_().square_()
            else:K.pow_(pe/eps)
            shift=0.
            if last:
                cs=(K.T @ torch.ones((n,1)))[:,0]
                if float(cs.min())>1e-34:g=(g+g+shift-eps*(cs.log()+la))*.5
                else:g=(g-eps*(((f[:,None]-C)/eps).logsumexp(0)+la))*.5
                break
            rs=K.sum(1);cs=(K.T @ torch.ones((n,1)))[:,0]
            if min(float(rs.min()),float(cs.min()))>1e-34:
                f.add_(rs.log()+lb,alpha=-.5*eps)
                g.add_(cs.log()+la,alpha=-.5*eps)
                if shift:
                    f.add_(.5*shift);g.add_(.5*shift)
                pe=eps
                continue
            else:ft,gt,K,shift,rs,cs=direct(f,g,eps)
        f=(f+ft)*.5;g=(g+gt)*.5;pe=eps
    result=C.sub_(g[None,:]).amin(1)
    if restore is not None:result.add_(restore)
    return result-.1*lb,None


def _label_cost(x, y, r, ry):
    aa = [torch.from_numpy(x[y == i]) for i in range(10)]
    bb = [torch.from_numpy(r[ry == j]) for j in range(10)]
    pairs = [(a, b) for a in aa for b in bb]
    nmax = max(len(a) for a in aa)
    mmax = max(len(b) for b in bb)
    A = torch.zeros((100, nmax, x.shape[1]))
    B = torch.zeros((100, mmax, r.shape[1]))
    an = torch.tensor([len(a) for a, b in pairs])
    bn = torch.tensor([len(b) for a, b in pairs])
    al=torch.stack([a.amin(0) for a in aa]);ah=torch.stack([a.amax(0) for a in aa])
    bl=torch.stack([b.amin(0) for b in bb]);bh=torch.stack([b.amax(0) for b in bb])
    diam=torch.linalg.vector_norm(torch.maximum(ah[:,None,:],bh[None,:,:])-torch.minimum(al[:,None,:],bl[None,:,:]),dim=2).flatten().tolist()
    ds=np.asarray(diam,np.float64)
    logs=2*np.log(ds);delta=2*np.log(.5)
    count=np.ceil((np.log(.1)-logs)/delta).astype(np.int64)
    length=count+2
    steps=np.arange(int(length.max()))[:,None]
    effective=(logs+delta)-logs
    temperatures=np.exp(logs[None,:]+(steps-1)*effective[None,:])
    temperatures=np.where((steps>=1)&(steps<=count[None,:]),temperatures,.1)
    temperatures[0]=ds*ds
    etable=torch.from_numpy(temperatures.astype(np.float32))
    atable=torch.from_numpy(steps<length[None,:])
    hard=(temperatures.astype(np.float32).max(1)<=256.).tolist()
    end=int(length.max())-3
    for i,a in enumerate(aa):A[i*10:(i+1)*10,:len(a)]=a
    for j,b in enumerate(bb):B[j::10,:len(b)]=b
    C=torch.bmm(B,A.transpose(1,2)).transpose(1,2)
    C.mul_(-1.).add_(A.square().sum(2)[:,:,None]/2).add_(B.square().sum(2)[:,None,:]/2)
    amask = torch.arange(nmax)[None,:] < an[:,None]
    bmask = torch.arange(mmax)[None,:] < bn[:,None]
    C.masked_fill_(~(amask[:,:,None] & bmask[:,None,:]), float('inf'))
    la=(1/an.float()).log()[:,None];lb=(1/bn.float()).log()[:,None]
    def direct(f,g,eps):
        R=(C-f[:,:,None]).sub_(g[:,None,:])
        if hard[t]:
            ft=f+R.amin(2)-eps[:,None]*lb
            gt=g+R.amin(1)-eps[:,None]*la
            return torch.where(amask,ft,0.),torch.where(bmask,gt,0.),None,None,None,None
        if t==2 and float(((f.amax(1)+g.amax(1))/eps).max())<60.:
            rm=torch.zeros(100)
            K=R.mul_((-1.4426950408889634/eps)[:,None,None]).exp2_()
        else:
            rm=R.amin((1,2))
            K=R.sub_(rm[:,None,None]).mul_((-1.4426950408889634/eps)[:,None,None]).exp2_()
        rs=K.sum(2);cs=K.sum(1)
        check=torch.minimum(torch.where(amask,rs,float('inf')).amin(1),torch.where(bmask,cs,float('inf')).amin(1))
        if float(check.min())>1e-34:
            ft=f+rm[:,None]-eps[:,None]*(rs.log()+lb)
            gt=g+rm[:,None]-eps[:,None]*(cs.log()+la)
            return torch.where(amask,ft,0.),torch.where(bmask,gt,0.),K,rm,rs,cs
        v=(g[:,None,:]-C)/eps[:,None,None]
        ft=-eps[:,None]*(v.logsumexp(2)+lb)
        v=(f[:,:,None]-C)/eps[:,None,None]
        gt=-eps[:,None]*(v.logsumexp(1)+la)
        return torch.where(amask,ft,0.),torch.where(bmask,gt,0.),None,None,None,None
    amean=A.sum(1)/an[:,None];bmean=B.sum(1)/bn[:,None]
    asq=A.square().sum(2)/2;bsq=B.square().sum(2)/2
    f=torch.where(amask,asq+bsq.sum(1)[:,None]/bn[:,None]-(A*bmean[:,None,:]).sum(2),0.)
    g=torch.where(bmask,bsq+asq.sum(1)[:,None]/an[:,None]-(B*amean[:,None,:]).sum(2),0.)
    K=None;shift=None;rs=None;cs=None
    pe=None;pa=torch.ones((100,1),dtype=torch.bool)
    hard_start=None
    for t in range(2,end):
        eps=etable[t]
        active=atable[t,:,None]
        if hard[t]:
            hard_start=t
            break
        if K is None or hard[t]:
            ft,gt,K,shift,rs,cs=direct(f,g,eps)
        else:
            ar=torch.where(amask & pa,(rs/bn[:,None]).sqrt(),1.)
            bc=torch.where(bmask & pa,(cs/an[:,None]).sqrt(),1.)
            K.mul_(ar.reciprocal()[:,:,None]).mul_(bc.reciprocal()[:,None,:])
            q=pe/eps
            if bool(torch.all(torch.abs(q-4.)<1e-4)):K.square_().square_()
            else:K.pow_(q[:,None,None])
            shift=torch.where(pa[:,0],0.,shift)
            rs=K.sum(2);cs=K.sum(1)
            check=torch.minimum(torch.where(amask,rs,float('inf')).amin(1),torch.where(bmask,cs,float('inf')).amin(1))
            if float(check.min())>1e-34:
                ft=f+shift[:,None]-eps[:,None]*(rs.log()+lb)
                gt=g+shift[:,None]-eps[:,None]*(cs.log()+la)
                ft=torch.where(amask,ft,0.);gt=torch.where(bmask,gt,0.)
            else:ft,gt,K,shift,rs,cs=direct(f,g,eps)
        df=torch.where(active,ft-f,0.)
        dg=torch.where(active,gt-g,0.)
        f.add_(df,alpha=.5)
        g.add_(dg,alpha=.5)
        pe=eps;pa=active
    if hard_start is not None:
        C.sub_(f[:,:,None]).sub_(g[:,None,:])
        for t in range(hard_start,end):
            eps=etable[t,:,None];active=atable[t,:,None]
            df=torch.where(amask & active,C.amin(2)-eps*lb,0.)
            dg=torch.where(bmask & active,C.amin(1)-eps*la,0.)
            f.add_(df,alpha=.5);g.add_(dg,alpha=.5)
            if t<end-1:
                C.sub_(df[:,:,None],alpha=.5).sub_(dg[:,None,:],alpha=.5)
        C.sub_(df[:,:,None],alpha=.5).sub_(dg[:,None,:],alpha=.75)
        g.add_(dg,alpha=.25)
    else:
        g.add_(dg,alpha=.25)
        C.sub_(f[:,:,None]).sub_(g[:,None,:])
    ft=torch.where(amask,f+C.amin(2)-.1*lb,0.)
    gt=torch.where(bmask,g+C.amin(1)-.1*la,0.)
    costs=(ft/an[:,None]).sum(1)+(gt/bn[:,None]).sum(1)
    W=np.zeros((20,20),np.float32)
    W[:10,10:]=costs.numpy().reshape(10,10);W[10:,:10]=W[:10,10:].T
    return W


def _expectation(C):
    n,m=C.shape
    K=(C*(-14.426950408889634/float(C.max()))).exp2_()
    v=(1./m)/K.sum(0)
    u=(1./n)/(K @ v[:,None])[:,0]
    col=v*(K.T @ u[:,None])[:,0]
    w=v*(2.-m*col).clamp_(min=.01)
    rowmass=(K @ w[:,None])[:,0]
    energy=(K.mul_(C) @ w[:,None])[:,0]
    return float((energy/rowmass).mean())


def _outer(batchcost):
    n,m=batchcost.shape
    cost=batchcost/max(float(batchcost.max()),1e-12)
    if m!=2:
        return ot.sinkhorn(ot.unif(n),ot.unif(m),cost,.01,numItermax=10000,stopThr=1e-10)
    d=(cost[:,1]-cost[:,0])/.01
    lo=-float(d.max())-40.;hi=-float(d.min())+40.
    z=-float(d.mean())
    for it in range(50):
        s=1./(1.+np.exp(-np.clip(d+z,-700.,700.)))
        err=float(s.mean())-.5
        if abs(err)<1e-13:break
        if err>0.:hi=z
        else:lo=z
        deriv=float((s*(1.-s)).mean())
        next_z=z-err/max(deriv,1e-30)
        z=next_z if lo<next_z<hi else (lo+hi)/2.
    return np.column_stack((s,1.-s))/n


def value(x, y, r, ry, seed=17, batch_size=1024):
    torch.set_flush_denormal(True)
    torch.set_num_threads(2)
    x=np.asarray(x,np.float32); r=np.asarray(r,np.float32)
    y=np.asarray(y,np.int64); ry=np.asarray(ry,np.int64)
    tx=[np.arange(i,min(i+batch_size,len(x))) for i in range(0,len(x),batch_size)]
    tr=[np.arange(i,min(i+batch_size,len(r))) for i in range(0,len(r),batch_size)]
    with torch.inference_mode():
        W=_label_cost(x[tx[0]],y[tx[0]],r[tr[0]],ry[tr[0]])
        WL=W[:10,10:].copy()/2
        X=torch.from_numpy(x);R=torch.from_numpy(r)
        xn=(X*X).sum(1)/2;rn=(R*R).sum(1)/2
        am=torch.stack([X[idx].mean(0) for idx in tx]);bm=torch.stack([R[idx].mean(0) for idx in tr])
        amn=torch.stack([xn[idx].mean() for idx in tx]);bmn=torch.stack([rn[idx].mean() for idx in tr])
        aw=np.stack([np.bincount(y[idx],minlength=10)/len(idx) for idx in tx]).astype(np.float32)
        bw=np.stack([np.bincount(ry[idx],minlength=10)/len(idx) for idx in tr]).astype(np.float32)
        fm=xn[:,None]+bmn[None,:]-X @ bm.T+torch.from_numpy((WL @ bw.T)[y])
        gm=rn[:,None]+amn[None,:]-R @ am.T+torch.from_numpy((aw @ WL).T[ry])
        Ax=torch.zeros((len(x),22));Br=torch.zeros((len(r),22))
        Ax[:,:10]=-X;Ax[:,10]=xn;Ax[:,11]=1.
        Ax[:,12:]=torch.from_numpy(WL[y])
        Br[:,:10]=R;Br[:,10]=1.;Br[:,11]=rn
        Br[torch.arange(len(r)),12+torch.from_numpy(ry)]=1.
        allcost=Ax @ Br.T
        F={}; batchcost=np.zeros((len(tx),len(tr)))
        for i,idx in enumerate(tx):
            for j,jdx in enumerate(tr):
                a=x[idx]; b=r[jdx]
                aa=X[i*batch_size:i*batch_size+len(idx)];bb=R[j*batch_size:j*batch_size+len(jdx)]
                T=allcost[i*batch_size:i*batch_size+len(idx),j*batch_size:j*batch_size+len(jdx)].contiguous()
                batchcost[i,j]=_expectation(T)/(len(idx)+len(jdx))
                # Reuse the same empirical geometry for coupling and point potentials.
                lo=np.minimum(a.min(0),b.min(0));hi=np.maximum(a.max(0),b.max(0))
                yl=int(y[idx].min());yh=int(y[idx].max());rl=int(ry[jdx].min())+10;rh=int(ry[jdx].max())+10
                d2=float(np.sum((hi-lo)**2)) + float((max(yh,rh)-min(yl,rl))**2)
                diameter=np.sqrt(d2)
                schedule=[diameter**2]+[float(np.exp(e)) for e in np.arange(2*np.log(diameter),np.log(.1),2*np.log(.5))]+[.1]
                f,_=_dual(T,schedule,(fm[idx,j],gm[jdx,i]))
                f=f.numpy().ravel().astype(float)
                F[i,j]=(f-f.mean())*len(f)/(len(f)-1)
    Q=_outer(batchcost)
    values=np.zeros(len(x))
    for i,idx in enumerate(tx): values[idx]=sum(Q[i,j]*F[i,j] for j in range(len(tr)))
    return values
