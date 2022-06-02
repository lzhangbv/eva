from kfac.eva import KFAC as EVA
from kfac.kfac import KFAC as KFAC 
from kfac.sam import KFAC as SAM
from kfac.kfac import KFACParamScheduler

kfac_mappers = {
    'eva': EVA,
    'kfac': KFAC,
    'sam': SAM
    }

def get_kfac_module(kfac='eva'):
    return kfac_mappers[kfac]
