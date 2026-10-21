"""Expert-only learned conditional residual correction, fixed equal mixture."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def design(features):
    base = np.asarray(features['base'],dtype=np.float32)
    history = np.asarray(features['history'],dtype=np.float32)
    last = np.broadcast_to(history[:,-1,None,:],base.shape)
    recent = np.broadcast_to(history[:,-3:,:].mean(axis=1)[:,None,:],base.shape)
    slope = np.broadcast_to((history[:,-1,:]-history[:,-4,:])[:,None,:],base.shape)
    horizon = np.broadcast_to(np.arange(base.shape[1],dtype=np.float32)[None,:,None]/12,base.shape)
    node = np.broadcast_to(np.arange(base.shape[2],dtype=np.float32)[None,None,:]/base.shape[2],base.shape)
    clock = features['clock']
    return np.stack([base,last,recent,slope,features['anchor'],np.sin(clock*2*np.pi),np.cos(clock*2*np.pi),horizon,node],axis=-1).reshape(-1,9)


def fit(features,targets,seed):
    x = design(features)
    residual = (targets-features['base']).reshape(-1)
    keep = (targets.reshape(-1)>0) & np.isfinite(residual)
    model = HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,
                 max_leaf_nodes=15,l2_regularization=10,learning_rate=.08,random_state=seed)
    model.fit(x[keep],residual[keep])
    return model


def predict(state,features):
    correction = state.predict(design(features)).reshape(features['base'].shape)
    return (features['base']+.5*correction).astype(np.float32)
