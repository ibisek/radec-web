"""
Created on 22. 01. 2021

@author: ibisek
"""

from flask import Flask
from flask import render_template
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
        file.formatLabel = FileFormat(file.format).name

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
@app.route('/airplane/<airplaneId>')
@app.route('/engine/<engineId>')
def index(airplaneId=None, engineId=None):
    try:
        airplaneId = int(airplaneId) if airplaneId else None
        engineId = int(engineId) if engineId else None
    except ValueError:
        airplaneId = None
        engineId = None

    airplanes = engines = flights = cycles = None

    if not engineId:
        airplanes = _listAirplanes(airplaneId=airplaneId)
        engines = _listEngines(airplaneId=airplaneId)
    else:
        engine = enginesDao.getOne(id=engineId)
        airplane = airplanesDao.getOne(id=engine.airplane_id)
        engines = _listEngines(airplaneId=airplane.id)
        engines = [e for e in engines if e.id == engineId]

    components = None
    if engineId:
        components = _listComponents(engineId=engineId)
    elif len(engines) == 1:
        components = _listComponents(engineId=engines[0].id)

    notifications = files = totNumFiles = None
    if not airplaneId and not engineId:
        totNumFiles, files = _listFiles(limit=10)
        notifications = _listNotifications()

    # if airplaneId or engineId:
    #     if not airplaneId:
    #         airplaneId = engines[0].airplane.id
    #     flights = _listFlights(airplaneId=airplaneId, engineId=engineId)
    #     # cycles = listCycles(flights)

    return render_template('index.html', airplanes=airplanes, engines=engines, components=components, files=files, totNumFiles=totNumFiles,
                           notifications=notifications, flights=flights, cycles=cycles)


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

    return render_template('pokus.html', title=title, labels=labels, datasets=datasets)


# @app.route('/cam')
# def cam():
#     return render_template('cam.html')


# @app.route('/test', methods=['GET'])
# def test():
#     return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=True)
