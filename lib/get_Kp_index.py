#!/usr/bin/env python3

"""
getKpindex.py
===================================
GFZ German Research Centre for Geosciences (CC BY 4.0)
Author I. Wehner
created with Python 3.8.10
last modified on 25 May 2022
to run getKpindex function run:  from getKpindex import getKpindex
Downloaded from: https://kp.gfz-potsdam.de/en/data#c42 and https://kp.gfz-potsdam.de/app/webservice/python
-----------------------------------
"""

import json
import logging
import urllib.request
from datetime import datetime

logger = logging.getLogger(__name__)


def __checkdate__(starttime, endtime):
    if starttime > endtime:
        raise NameError("Error! Start time must be before or equal to end time")
    return True


def __checkIndex__(index):
    if index not in ['Kp', 'ap', 'Ap', 'Cp', 'C9', 'Hp30', 'Hp60', 'ap30', 'ap60', 'SN', 'Fobs', 'Fadj']:
        raise IndexError(
            "Error! Wrong index parameter! \nAllowed are only the string parameter: 'Kp', 'ap', 'Ap', 'Cp', 'C9', 'Hp30', 'Hp60', 'ap30', 'ap60', 'SN', 'Fobs', 'Fadj'")
    return True


def __checkstatus__(status):
    if status not in ['all', 'def']:
        raise IndexError("Error! Wrong option parameter! \nAllowed are only the string parameter: 'def'")
    return True


def __addstatus__(url, status):
    if status == 'def':
        url = url + '&status=def'
    return url


def getKpindex(starttime, endtime, index, status='all'):
    """
    ---------------------------------------------------------------------------------
    download 'Kp', 'ap', 'Ap', 'Cp', 'C9', 'Hp30', 'Hp60', 'ap30', 'ap60', 'SN', 'Fobs' or 'Fadj' index data from kp.gfz-potsdam.de
    date format for starttime and endtime is 'yyyy-mm-dd' or 'yyyy-mm-ddTHH:MM:SSZ'
    optional 'def' parameter to get only definitve values (only available for 'Kp', 'ap', 'Ap', 'Cp', 'C9', 'SN')
    Hpo index and Fobs/Fadj does not have the status info
    example: (time, index, status) = getKpindex('2021-09-29', '2021-10-01','Ap','def')
    example: (time, index, status) = getKpindex('2021-09-29T12:00:00Z', '2021-10-01T12:00:00Z','Kp')
    ---------------------------------------------------------------------------------
    """
    result_t = 0
    result_index = 0
    result_s = 0

    if len(starttime) == 10 and len(endtime) == 10:
        starttime = starttime + 'T00:00:00Z'
        endtime = endtime + 'T23:59:00Z'

    try:
        d1 = datetime.strptime(starttime, '%Y-%m-%dT%H:%M:%SZ')
        d2 = datetime.strptime(endtime, '%Y-%m-%dT%H:%M:%SZ')

        __checkdate__(d1, d2)
        __checkIndex__(index)
        __checkstatus__(status)

        time_string = "start=" + d1.strftime('%Y-%m-%dT%H:%M:%SZ') + "&end=" + d2.strftime('%Y-%m-%dT%H:%M:%SZ')
        url = 'https://kp.gfz-potsdam.de/app/json/?' + time_string + "&index=" + index
        if index not in ['Hp30', 'Hp60', 'ap30', 'ap60', 'Fobs', 'Fadj']:
            url = __addstatus__(url, status)

        webURL = urllib.request.urlopen(url)
        binary = webURL.read()
        text = binary.decode('utf-8')

        try:
            data = json.loads(text)
            result_t = data["datetime"]
            result_index = data[index]
            if index not in ['Hp30', 'Hp60', 'ap30', 'ap60', 'Fobs', 'Fadj']:
                result_s = tuple(data["status"])
        except:
            logger.warning(f"json.loads error: {text}")

    except NameError as er:
        logger.error(f"NameError: {er}")
    except IndexError as er:
        logger.error(f"IndexError: {er}")
    except ValueError as er:
        logger.error("Error! Wrong datetime string")
        logger.error("Both dates must be the same format.")
        logger.error("Datetime strings must be in format yyyy-mm-dd or yyyy-mm-ddTHH:MM:SSZ")
        logger.error(f"ValueError: {er}")
    except urllib.error.URLError as er:
        logger.error(f"Connection Error \nCan not reach {url}")
        logger.error(f"URLError: {er}")

    finally:
        return result_t, result_index, result_s
