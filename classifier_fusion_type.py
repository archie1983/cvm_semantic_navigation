from enum import Enum
##
# When we want to fuse LLM classified results with CVM ones, we then want to use one or another
# strategy. E.g., do we want to base it on LLM results and fill in CVM results where LLM
# failed, or do we want to use CVM results as a base and fill in LLM results where CVM
# couldn't classify.
##
class ClassifierFusionType(Enum):
    LLM_THEN_CVM = 1 # Use LLM result as a base and where we don't have result, take CVM result
    CVM_THEN_LLM = 2 # Use CVM result as a base and where we don't have result, take LLM result
    NO_FUSION = 3 # Don't do fusion
