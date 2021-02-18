#
# Custom filters for Jinja.
#
# @see https://jinja.palletsprojects.com/en/2.11.x/api/#writing-filters
#

from datetime import datetime


def dateTimeFormat(value, format='%H:%M:%S / %d-%m-%Y'):
    return value.strftime(format)


def tsFormat(ts, format='%Y-%m-%d %H:%M'):
    return datetime.utcfromtimestamp(ts).strftime(format)


def durationFormat(seconds):
    if not seconds:
        return 0

    h = seconds // 3600
    s = seconds % 3600
    m = s // 60
    s = s % 60

    if s >= 30:
        if m == 59:
            h += 1
            m = 0
        else:
            m += 1

    if h > 0:
        dur = f"{h}\N{DEGREE SIGN} {m}'"
    else:
        dur = f"{m}'"

    return dur
