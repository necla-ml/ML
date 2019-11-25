from collections import OrderedDict

def trim(stat, prefix='module'):
    r"""Remove prefix of state_dict keys.
    """

    stat_new = OrderedDict()
    for k, v in stat.items():
        if k.startswith(prefix):
            stat_new[k[len(prefix)+1:]] = v

    return stat_new if stat_new else stat