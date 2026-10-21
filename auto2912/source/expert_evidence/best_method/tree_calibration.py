"""Pooled nonlinear correction from current, anchor and historical context."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import calibration


def _features(f, context=False):
    b = np.asarray(f['base'], dtype=np.float32)
    a = np.asarray(f['anchor'], dtype=np.float32)
    z = np.asarray(f['history'], dtype=np.float32)
    clock = np.asarray(f['clock'], dtype=np.float32)
    def broadcast(v):
        return np.broadcast_to(v[:, None, :], b.shape)
    last = broadcast(z[:, -1, :])
    avg3 = broadcast(np.mean(z[:, -3:, :], axis=1))
    avg6 = broadcast(np.mean(z[:, -6:, :], axis=1))
    avg = broadcast(np.mean(z, axis=1))
    std = broadcast(np.std(z, axis=1))
    delta = broadcast(np.mean(z[:, -3:, :],axis=1)-np.mean(z[:, :3, :],axis=1))
    spread = broadcast(np.max(z,axis=1)-np.min(z,axis=1))
    horizon = np.broadcast_to(np.arange(b.shape[1])[None, :, None],b.shape)
    bd = np.gradient(b, axis=1)
    ad = np.gradient(a, axis=1)
    node = np.broadcast_to(np.arange(b.shape[2])[None,None,:],b.shape)
    features = [b, a-b, last-b, avg3-b, avg6-b, avg-b, std, delta,
                spread, clock, horizon, bd, ad, a, node]
    if context:
        def network(v):
            return np.broadcast_to(np.mean(v,axis=2,keepdims=True),b.shape)
        def block(v):
            return np.broadcast_to(np.mean(v,axis=1,keepdims=True),b.shape)
        features.extend([network(b),network(a-b),network(avg3-b),
                         network(delta),network(std),
                         (a-b)/np.maximum(a,30),std/np.maximum(avg,30),
                         block(a-b),block(b)-b,
                         b[:,0,None,:]+np.zeros_like(b)-b,
                         np.sin(2*np.pi*clock),np.cos(2*np.pi*clock)])
    return np.stack(features,axis=-1).reshape(-1,len(features))


def fit(features, targets, seed, mode='raw', max_iter=90, context=False):
    with threadpool_limits(limits=2):
        y = np.asarray(targets['target'] if isinstance(targets,dict) else targets)
        valid = np.isfinite(y)&(y>0)
        x = _features(features,context)
        residual = y-np.asarray(features['base'])
        model = HistGradientBoostingRegressor(loss='absolute_error',
                    max_iter=max_iter,learning_rate=0.06,max_leaf_nodes=15,
                    min_samples_leaf=250,l2_regularization=20,
                    categorical_features=[14],early_stopping=False,
                    random_state=seed)
        state = {'mode':mode, 'context':context}
        if mode=='residual':
            state['linear']=calibration.fit(features,targets,seed)
            residual=y-calibration.predict(state['linear'],features)
        model.fit(x[valid.ravel()],residual[valid])
        state['model']=model
        return state


def predict(state,features):
    with threadpool_limits(limits=2):
        residual=state['model'].predict(_features(features,state.get('context',False))).reshape(features['base'].shape)
        base=np.asarray(features['base'])
        if state['mode']=='residual':
            base=calibration.predict(state['linear'],features)
        return np.maximum(base+residual,0).astype(np.float32)
