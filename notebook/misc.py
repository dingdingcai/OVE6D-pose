import numpy as np
def str2dict(ss):
    obj_score = dict()
    for obj_str in ss.split(','):
        obj_s = obj_str.strip()
        if len(obj_s) > 0:
            obj_id = obj_s.split(':')[0].strip()
            obj_s = obj_s.split(':')[1].strip()
            if len(obj_s) > 0:
                obj_score[int(obj_id)] = float(obj_s)
    return obj_score

def cal_score(adi_str, add_str):
    adi_score = str2dict(adi_str)
    add_score = str2dict(add_str)
    add_score[10] = adi_score[10]
    add_score[11] = adi_score[11]
    if 3 in add_score:
        add_score.pop(3)
    if 7 in add_score:
        add_score.pop(7)
    return np.mean(list(add_score.values()))

def printAD(add, adi, name='RAW'):
    print("{}: ADD:{:.5f}, ADI:{:.5f}, ADD(-S):{:.5f}".format(
        name,
    np.sum(list(str2dict(add).values()))/len(str2dict(add)),
    np.sum(list(str2dict(adi).values()))/len(str2dict(adi)),
    cal_score(adi_str=adi, add_str=add)))
    