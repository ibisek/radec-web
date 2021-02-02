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
