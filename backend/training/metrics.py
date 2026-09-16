import numpy as np


def retrieval_metrics(queries, gallery, labels, threshold=.88, margin=.04):
    queries=np.asarray(queries,dtype=np.float32)
    gallery=np.asarray(gallery,dtype=np.float32)
    scores=queries @ gallery.T
    order=np.argsort(-scores,axis=1,kind='stable')
    labels=np.asarray(labels,dtype=np.int64)
    correct=order[:,0]==labels
    best=np.take_along_axis(scores,order[:,:2],axis=1)
    accepted=(best[:,0]>=threshold)&((best[:,0]-best[:,1])>=margin)
    ranks=np.argmax(order==labels[:,None],axis=1)+1
    return {
        'queries':len(labels), 'gallery_size':len(gallery),
        'top1':float(correct.mean()),
        'top3':float((ranks<=3).mean()), 'top5':float((ranks<=5).mean()),
        'mrr':float((1/ranks).mean()), 'accepted':int(accepted.sum()),
        'coverage':float(accepted.mean()),
        'accepted_precision':float(correct[accepted].mean()) if accepted.any() else None,
        'false_accepts':int((accepted & ~correct).sum()),
        'threshold':threshold, 'margin':margin,
    }
