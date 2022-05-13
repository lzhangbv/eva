from kfac.eva import KFAC as EVA
from kfac.kfac import KFAC as KFAC 
from kfac.kfac import KFACParamScheduler

kfac_mappers = {
    'eva': EVA,
    'kfac': KFAC,
    }

def get_kfac_module(kfac='eva'):
    return kfac_mappers[kfac]
