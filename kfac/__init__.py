from kfac.eva import KFAC as EVA
from kfac.semi_eva import KFAC as SEMI_EVA
from kfac.kfac import KFAC as KFAC 
from kfac.sam import KFAC as SAM
from kfac.adasgd import KFAC as ADASGD
from kfac.adasgd import KFAC2 as ADASGD2
from kfac.kfac import KFACParamScheduler

kfac_mappers = {
    'eva': EVA,
    'semi-eva': SEMI_EVA,
    'kfac': KFAC,
    'adasgd': ADASGD,
    'adasgd2': ADASGD2,
    'sam': SAM, 
    }

def get_kfac_module(kfac='eva'):
    return kfac_mappers[kfac]
