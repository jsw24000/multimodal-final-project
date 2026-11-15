import numpy as np
from experiments.pa_mechanism.phase2.summarize import miou

def test():
    a=np.array([[8,2],[1,9]],dtype=np.int64)
    b=np.array([[4,1],[3,2]],dtype=np.int64)
    pooled=a+b
    expected=((12/(12+3+4))+(11/(11+4+3)))/2*100
    assert abs(miou(pooled)-expected)<1e-10
    # Image IoU averages are not the pooled dataset IoU in general.
    assert abs(miou(pooled)-(miou(a)+miou(b))/2)>1
    rng=np.random.default_rng(24000)
    ix=rng.integers(0,2,2)
    assert np.array_equal(np.stack([a,b])[ix].sum(0),sum([a,b][i] for i in ix))

if __name__=='__main__':test();print('statistics tests: PASS')
