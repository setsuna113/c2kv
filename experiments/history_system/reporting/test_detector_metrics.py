from experiments.history_system.reporting.detector_metrics import binary_metrics


def test_unknown_is_excluded_from_confusion_but_kept_in_coverage():
    x=binary_metrics([{'label':1,'triggered':True},{'label':0,'triggered':True},{'label':1,'triggered':False},{'label':None,'triggered':False}])
    assert (x['tp'],x['fp'],x['fn'],x['tn'])==(1,1,1,0)
    assert x['precision']==x['recall']==x['f1']==0.5
    assert x['fpr']==1 and x['label_coverage']==0.75


def test_undefined_denominator_stays_missing():
    x=binary_metrics([{'label':None,'triggered':False}])
    assert all(x[k] is None for k in ['precision','recall','f1','fpr'])
