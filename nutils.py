# Author: Randa Yoga Saputra
# Version: 1
# Purpose: Utils

def findTrueKey(dic: dict) -> str: 
    """Given a dictionary as input, return the first key
    where a value of True is set as the value. An empty string is
    returned if no truth value was found. 
    """
    kv = [(key, dic[key]) for key in dic if dic[key]]
    return kv[0][0] if len(kv) > 0 else ""