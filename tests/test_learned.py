import pytest
import torch
from torch.nn import functional as F
from interfacemark.learned import features, multi_loss, train, score


@pytest.mark.parametrize('channels',[4,16])
def test_features(channels):
    x=torch.randn(2,channels,64,64); k=torch.randn(1,channels,64,64)
    f=features(x,k)
    assert f.shape==(2,1,17*channels)
    assert torch.allclose(f[:,0,:channels],(x*k).mean((-2,-1)),atol=1e-6)
    assert torch.isfinite(f).all()


def test_loss():
    sc=torch.tensor([-.3,.2],requires_grad=True)
    sw=torch.tensor([.4,.8],requires_grad=True); wk=torch.tensor([.1,-.2],requires_grad=True)
    tc=torch.tensor([-1.,0.]); tw=torch.tensor([1.,2.]); weights=torch.tensor([3.,1.])
    cls=.5*(F.binary_cross_entropy_with_logits(sc,torch.zeros_like(sc),reduction='none')+
            F.binary_cross_entropy_with_logits(sw,torch.ones_like(sw),reduction='none'))
    teacher=.5*(F.smooth_l1_loss(sc,tc,reduction='none')+F.smooth_l1_loss(sw,tw,reduction='none'))
    expected=((cls+.25*F.softplus(.5-sw+sc)+.20*teacher+
               .15*F.binary_cross_entropy_with_logits(wk,torch.zeros_like(wk),reduction='none'))*weights).sum()/4
    actual=multi_loss(sc,sw,wk,tc,tw,weights)
    assert torch.allclose(actual,expected)
    actual.backward(); assert all(torch.isfinite(t.grad).all() for t in (sc,sw,wk))


def cache(offset):
    return dict(clean=torch.randn(4,2,68),marked=torch.randn(4,2,68),wrong_marked=torch.randn(4,2,68),
                projection_clean=torch.randn(4),projection_marked=torch.randn(4),
                views=['identity','blur_2'],ids=[str(offset+i) for i in range(4)])


def test_train_roundtrip(tmp_path):
    torch.set_num_threads(2)
    tr,va=cache(0),cache(10)
    result=train(tr,va,tmp_path/'model.pt',epochs=2)
    loaded=torch.load(tmp_path/'model.pt',weights_only=True)
    assert result['protocol']=='multi_loss_identity_teacher_v1'
    assert torch.equal(score(va['clean'],loaded),score(va['clean'],loaded))
    assert torch.isfinite(score(va['clean'],loaded)).all()
    with pytest.raises(ValueError,match='overlap'):
        train(tr,tr,tmp_path/'bad.pt',epochs=1)
