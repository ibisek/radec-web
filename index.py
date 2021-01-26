"""
Created on 22. 01. 2021

@author: ibisek
"""

from flask import Flask
from flask import render_template, redirect
from datetime import datetime
from pandas import DataFrame
from collections import namedtuple

from data.structures import FileFormat
from db.dao import flightRecordingDao
from db.dao.airplanesDao import AirplanesDao
from db.dao.equipmentDao import EquipmentDao
from db.dao.enginesDao import EnginesDao
from db.dao.flightsDao import FlightsDao
from db.dao.notificationsDao import NotificationsDao
from db.dao.filesDao import FilesDao
from db.dao.cyclesDao import CyclesDao
from db.dao.componentsDao import ComponentsDao
from db.dao.flightRecordingDao import FlightRecordingDao

Dataset = namedtuple('Dataset', ['label', 'data', 'color'])

app = Flask(__name__)

airplanesDao = AirplanesDao()
componentsDao = ComponentsDao()
enginesDao = EnginesDao()
equipmentDao = EquipmentDao()
filesDao = FilesDao()
flightsDao = FlightsDao()
notificationsDao = NotificationsDao()
flightRecordingDao = FlightRecordingDao()


def _listNotifications(limit: int = 10):
    notifications = notificationsDao.listMostRecent(limit=limit)

    cyclesDao = CyclesDao()
    for n in notifications:
        setattr(n, 'formattedDt', None)
        if n.ts > 0:
            n.formattedDt = datetime.utcfromtimestamp(n.ts).strftime('%Y-%m-%d %H:%M')

        setattr(n, 'airplane', None)
        setattr(n, 'engine', None)
        setattr(n, 'cycle', None)

        if n.airplane_id:
            n.airplane = airplanesDao.getOne(id=n.airplane_id)
        if n.engine_id:
            n.engine = enginesDao.getOne(id=n.engine_id)
            if not n.airplane:
                n.airplane = airplanesDao.getOne(id=n.engine.airplane_id)
        if n.cycle_id:
            n.cycle = cyclesDao.getOne(id=n.cycle_id)

    return notifications


def _listFiles(limit: int = 10):
    # files = [f for f in filesDao.get()]
    files = filesDao.list(limit=limit)
    totNumFiles = len(files)
    # files = files[:limit]

    for file in files:
        file.formatName = FileFormat(file.format).name

    return totNumFiles, files


def _listAirplanes(airplaneId: int = None):
    if airplaneId:
        airplanes = [a for a in airplanesDao.get(id=airplaneId)]
    else:
        airplanes = [a for a in airplanesDao.get()]

    for airplane in airplanes:
        eq = equipmentDao.getOne(id=airplane.equipment_id)
        if eq:
            setattr(airplane, 'type', eq.label)
        else:
            setattr(airplane, 'type', None)

    return airplanes


def _listEngines(airplaneId=None):
    if airplaneId:
        engines = [a for a in enginesDao.get(airplane_id=airplaneId)]
    else:
        engines = [a for a in enginesDao.get()]

    for engine in engines:
        eq = equipmentDao.getOne(id=engine.equipment_id)
        if eq:
            setattr(engine, 'type', eq.label)
        else:
            setattr(engine, 'type', None)

        airplane = AirplanesDao().getOne(id=engine.airplane_id)
        setattr(engine, 'airplane', airplane)

        notifications = [n for n in notificationsDao.get(engine_id=engine.id)]
        if notifications:
            numInfo = 0
            numWarning = 0
            numUrgent = 0
            for n in notifications:
                if n.type >= 255:
                    numUrgent += 1
                elif n.type <= 1:
                    numInfo += 1
                else:
                    numWarning += 1
            numNotifications = {'len': len(notifications), 'info': numInfo, 'warning': numWarning, 'urgent': numUrgent}
            setattr(engine, 'numNotifications', numNotifications)

        else:
            setattr(engine, 'notifications', [])

    return engines


def _listComponents(engineId):
    components = [c for c in componentsDao.get(engine_id=engineId)]

    for component in components:
        eq = equipmentDao.getOne(id=component.equipment_id)
        setattr(component, 'type', eq.label)

    return components


def _listFlights(airplaneId: int):
    return []


@app.route('/')
def index():
    airplanes = _listAirplanes()
    engines = _listEngines()

    totNumFiles, files = _listFiles(limit=10)
    notifications = _listNotifications()

    return render_template('index.html', airplanes=airplanes, engines=engines,
                           files=files, totNumFiles=totNumFiles,
                           notifications=notifications)


@app.route('/airplane/<airplaneId>')
def indexAirplane(airplaneId:int):
    try:
        airplaneId = int(airplaneId) if airplaneId else None
    except ValueError:
        return render_template('errorMsg.html', message="No such data!")

    airplane = airplanesDao.getOne(id=airplaneId)
    if not airplane:
        return render_template('errorMsg.html', message="No such data!")

    engines = _listEngines(airplaneId=airplaneId)

    files = filesDao.listRawFilesForAirplane(airplaneId=airplaneId)
    totNumFiles = len(files)

    # notifications = _listNotifications(airplaneIds=[airplaneId], engineIds=[e.id for e in engines])
    notifications = None

    return render_template('index.html', airplanes=[airplane], engines=engines,
                           files=files, totNumFiles=totNumFiles,
                           notifications=notifications)


@app.route('/engine/<engineId>')
def indexEngine(engineId: int):
    try:
        airplaneId = int(engineId) if engineId else None
    except ValueError:
        return render_template('errorMsg.html', message="No such data!")

    engine = enginesDao.getOne(id=engineId)
    if not engine:
        return render_template('errorMsg.html', message="No such data!")

    airplane = airplanesDao.getOne(id=engine.airplane_id)

    # enhances with some metadata (this is so lame but it works):
    engine = [e for e in _listEngines(airplaneId=airplane.id) if e.id == engine.id][0]

    components = None
    if engineId:
        components = _listComponents(engineId=engineId)

    # notifications = _listNotifications(engineId=engineId)
    notifications = None

    return render_template('index.html', airplanes=[airplane], engines=[engine], components=components, notifications=notifications)


@app.route('/chart/<engineId>/<flightId>/<flightIdx>/<cycleId>/<cycleIdx>')
def showChart(engineId: int, flightId: int, flightIdx: int, cycleId: int, cycleIdx: int):
    try:
        engineId = int(engineId)
        flightId = int(flightId)
        flightIdx = int(flightIdx)
        cycleId = int(cycleId)
        cycleIdx = int(cycleIdx)
    except Exception as ex:
        print(ex)
        return render_template('errorMsg.html', message="No such data!")

    df: DataFrame = flightRecordingDao.loadDf(engineId=engineId, flightId=flightId, flightIdx=flightIdx, cycleId=cycleId, cycleIdx=cycleIdx)
    if df.empty:
        return render_template('errorMsg.html', message="No such data!")

    title = f"Flight recording for engineId = {engineId}, flightId = {flightId}, idx = {flightIdx}, cycleId = {cycleId}, idx = {cycleIdx}"

    labels = ','.join([datetime.utcfromtimestamp(dt.astype(datetime) / 1e9).strftime('"%Y-%m-%d %H:%M"') for dt in df.index.values])

    iasKey = 'IAS' if 'IAS' in df.keys() else 'TAS'
    keys = ('ALT', iasKey, 'ITT', 'NG', 'NP')
    colors = ('rgba(0, 0, 255, 1)', 'rgba(0, 255, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 0, 255, 1)', 'rgba(0, 255, 255, 1)')
    datasets = []
    for color, key in zip(colors, keys):
        data = ','.join([f'{float(a):.0f}' for a in df[key].values])
        ds = Dataset(label=key, data=data, color=color)
        datasets.append(ds)

    return render_template('chart.html', aspectRatio=16/9, chartId=1, title=title, labels=labels, datasets=datasets)


@app.route('/pokus')
def pokus():
    engineId = 3
    flightId = 1800
    flightIdx = 2
    cycleId = 4235
    cycleIdx = 0

    df: DataFrame = flightRecordingDao.loadDf(engineId=engineId, flightId=flightId, flightIdx=flightIdx, cycleId=cycleId, cycleIdx=cycleIdx)

    title = f"Flight recording for engineId = {engineId}, flightId = {flightId}, idx = {flightIdx}, cycleId = {cycleId}, idx = {cycleId}"

    labels = ','.join([datetime.utcfromtimestamp(dt.astype(datetime)/1e9).strftime('"%Y-%m-%d %H:%M"') for dt in df.index.values])

    keys = ('ALT', 'IAS', 'ITT', 'NG', 'NP')
    colors = ('rgba(0, 0, 255, 1)', 'rgba(0, 255, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 0, 255, 1)', 'rgba(0, 255, 255, 1)')
    datasets = []
    for color, key in zip(colors, keys):
        data = ','.join([f'{float(a):.0f}' for a in df[key].values])
        ds = Dataset(label=key, data=data, color=color)
        datasets.append(ds)

    return render_template('pokus.html', id=1, title=title, labels=labels, datasets=datasets)


# @app.route('/cam')
# def cam():
#     return render_template('cam.html')


# @app.route('/test', methods=['GET'])
# def test():
#     return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=True)
