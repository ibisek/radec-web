"""
Created on 22. 01. 2021

@author: ibisek
"""

import numpy as np
from flask import Flask, render_template, redirect, make_response
from datetime import datetime
from pandas import DataFrame
from collections import namedtuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from configuration import DEBUG
from customFilters import tsFormat, durationFormat
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
from db.dao.regressionResultsDao import RegressionResultsDao

Dataset = namedtuple('Dataset', ['label', 'unit', 'data', 'color'])

app = Flask(__name__)
# register custom filters:
app.jinja_env.filters['tsFormat'] = tsFormat
app.jinja_env.filters['durationFormat'] = durationFormat

airplanesDao = AirplanesDao()
componentsDao = ComponentsDao()
enginesDao = EnginesDao()
equipmentDao = EquipmentDao()
filesDao = FilesDao()
cyclesDao = CyclesDao()
flightsDao = FlightsDao()
notificationsDao = NotificationsDao()
flightRecordingDao = FlightRecordingDao()
regressionResultsDao = RegressionResultsDao()


def _listNotifications(limit: int = 10):
    notifications = notificationsDao.listMostRecent(limit=limit)

    cyclesDao = CyclesDao()
    for n in notifications:
        setattr(n, 'formattedDt', None)
        if n.start_ts > 0:
            n.formattedDt = datetime.utcfromtimestamp(n.start_ts).strftime('%Y-%m-%d %H:%M')

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
        airplanes = [airplanesDao.getOne(id=airplaneId)]
    else:
        airplanes = [a for a in airplanesDao.get()]

    for airplane in airplanes:
        eq = equipmentDao.getOne(id=airplane.equipment_id)
        if eq:
            setattr(airplane, 'type', eq.label)
        else:
            setattr(airplane, 'type', None)

        numNotifications = notificationsDao.countNotificationsFor(airplaneId=airplane.id)
        setattr(airplane, 'numNotifications', numNotifications)

    return airplanes


def _listEngines(airplaneId=None):
    if airplaneId:
        engines = [e for e in enginesDao.get(airplane_id=airplaneId)]
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

        numNotifications = notificationsDao.countNotificationsFor(engineId=engine.id)
        setattr(engine, 'numNotifications', numNotifications)

    return engines


def _listComponents(engineId):
    components = [c for c in componentsDao.get(engine_id=engineId)]

    for component in components:
        eq = equipmentDao.getOne(id=component.equipment_id)
        setattr(component, 'type', eq.label)

    return components


def enhanceAirplane(airplane):
    # enhance the airplane by its operation indicators:
    numFlights, flightTime, operationTime = flightsDao.getAirplaneStats(airplaneId=airplane.id)
    setattr(airplane, 'numFlights', numFlights)
    setattr(airplane, 'flightTime', flightTime)
    setattr(airplane, 'operationTime', operationTime)


@app.route('/')
def index():
    airplanes = _listAirplanes()
    for airplane in airplanes:
        enhanceAirplane(airplane)

    engines = _listEngines()

    totNumFiles, files = _listFiles(limit=10)
    notifications = _listNotifications()

    return render_template('index.html', airplanes=airplanes, engines=engines,
                           files=files, totNumFiles=totNumFiles,
                           notifications=notifications)


@app.route('/airplane/<airplaneId>')
def indexAirplane(airplaneId: int):
    try:
        airplaneId = int(airplaneId)
    except ValueError:
        return render_template('errorMsg.html', message="No such data!")

    airplanes = _listAirplanes(airplaneId=airplaneId)
    if len(airplanes) == 0:
        return render_template('errorMsg.html', message="No such data!")

    airplane = airplanes[0]
    enhanceAirplane(airplane)

    engines = _listEngines(airplaneId=airplaneId)

    files = filesDao.listRawFilesForAirplane(airplaneId=airplaneId, limit=10)
    totNumFiles = len(files)

    flights, _ = flightsDao.listForView(airplaneId=airplaneId)

    # notifications = _listNotifications(airplaneIds=[airplaneId], engineIds=[e.id for e in engines])
    notifications = None

    return render_template('index.html', airplanes=[airplane], engines=engines,
                           files=files, totNumFiles=totNumFiles,
                           flights=flights, airplaneId=airplaneId,
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
    enhanceAirplane(airplane)

    # enhances with some metadata (this is so lame but it just works):
    engine = [e for e in _listEngines(airplaneId=airplane.id) if e.id == engine.id][0]

    components = None
    if engineId:
        components = _listComponents(engineId=engineId)

    cycles, _ = cyclesDao.listForView(engineId=engineId)

    # notifications = _listNotifications(engineId=engineId)
    notifications = None

    menuItems = [{'text': 'Trend monitoring', 'link': f'/trends/{engineId}'}]

    return render_template('index.html', menuItems=menuItems,
                           airplanes=[airplane], engines=[engine], components=components,
                           cycles=cycles,
                           notifications=notifications)


@app.route('/csv/<type>/<id>', methods=['GET'])
def csv(type: str, id: int):
    """
    :param type: c-ycle, f-light
    :param id: cycles for engineId or flights for airplaneId
    :return:
    """
    if not type or not id or type not in ('c', 'f'):
        return redirect('/')

    if type == 'c':
        records, colNames = cyclesDao.listForView(engineId=id)
    else:
        records, colNames = flightsDao.listForView(airplaneId=id)

    csvText = ';'.join([colName for colName in colNames]) + '\n'
    for record in records:
        row = ";".join([str(getattr(record, colName, '')) for colName in colNames])
        csvText += row + "\n"

    typeStr = 'cycles' if type == 'c' else 'flights'

    output = make_response(csvText)
    output.headers["Content-Disposition"] = f"attachment; filename={typeStr}_{datetime.now().strftime('%Y-%m-%d')}.csv"
    output.headers["Content-type"] = "text/csv"

    return output


@app.route('/chart/<engineId>/<what>/<whatId>')
def showChart(engineId: int, what: str, whatId: int):
    """
    :param engineId:
    :param what:  c/f - cycle/flight
    :param whatId: cycle of flight id
    """
    try:
        engineId = int(engineId)
        whatId = int(whatId)
        assert what in ['c', 'f']
    except Exception as ex:
        print(ex)
        return render_template('errorMsg.html', message="No such data!")

    that = flightsDao.getOne(id=whatId) if what == 'f' else cyclesDao.getOne(id=whatId)

    df: DataFrame = flightRecordingDao.loadDf(engineId=engineId, startTs=that.rec_start_ts, endTs=that.rec_end_ts)
    if df.empty:
        return render_template('errorMsg.html', message="No such data!")

    thatTitle = 'cycle' if what == 'c' else 'flight'
    title = f"Flight recording for engineId = {engineId}, {thatTitle}Id = {whatId}"

    labels = ','.join([datetime.utcfromtimestamp(dt.astype(datetime) / 1e9).strftime('"%Y-%m-%d %H:%M"') for dt in df.index.values])

    iasKey = 'IAS' if 'IAS' in df.keys() else 'TAS'
    keys = ('ALT', iasKey, 'ITT', 'T0', 'NG', 'NP')
    colors = ('rgba(0, 0, 255, 1)', 'rgba(0, 255, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(252, 160, 3, 1)', 'rgba(0, 255, 255, 1)', 'rgba(255, 0, 255, 1)')
    units = ('m', 'km/h', '°C', '°C', '%', '1/min')
    datasets = []
    for color, key, unit in zip(colors, keys, units):
        data = ','.join([f'{float(a):.0f}' for a in df[key].values])
        ds = Dataset(label=key, unit=unit, data=data, color=color)
        datasets.append(ds)

    return render_template('chart.html', aspectRatio=16/9, chartId=1, title=title, labels=labels, units=units, datasets=datasets)


def renderNotEnufDataErr():
    return render_template('errorMsg.html',
                           message="Not enuf data!<br><br>You need to gather at least 50 data points before this analysis starts making any sense.")


@app.route('/trends/<engineId>')
def showTrends(engineId: int):
    functions = regressionResultsDao.listFunctions(engineId=engineId)
    if len(functions) == 0:
        return renderNotEnufDataErr()

    allTitles = []
    allLabels = []
    allDatasets = []

    keys = ['delta', 'mean', 'y_linreg', 'y_rolling',
            'trend',
            'rangeMax', 'rangeMin']  # , 'y_polyreg'
    colors = ('rgba(0, 0, 255, 1)', 'rgba(0, 0, 0, 1)', 'rgba(127, 23, 231, 1)', 'rgba(255, 0, 255, 1)',
              'rgba(0, 80, 0, 1)',
              'rgba(255, 0, 0, 1)', 'rgba(255, 0, 0, 1)')
    for fn in functions:
        df: DataFrame = regressionResultsDao.loadRegressionResultsData(engineId=engineId, function=fn)
        if len(df) < 50:
            return renderNotEnufDataErr()

        # TODO -- MAGIC start --
        df['mean'] = df['delta'].mean()

        x = np.arange(len(df)).reshape(-1, 1)
        y = df['delta'].values.reshape(-1, 1)
        linReg = LinearRegression()
        linReg.fit(x, y)
        df['y_linreg'] = linReg.predict(x)

        # degree = 3
        # polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        # polyreg.fit(x, y)
        # df['y_polyreg'] = polyreg.predict(x)

        df['y_rolling'] = df['delta'].rolling(10, center=True).mean()
        df = df.fillna(df['y_rolling'].mean())

        # "marek/ge" smoothing algorithm:
        smoothingCoeff = 0.1
        df['trend'] = 0
        for i in range(1, len(df)):
            df['trend'].iloc[i] = df['trend'].iloc[i - 1] + smoothingCoeff * (df['delta'].iloc[i] - df['trend'].iloc[i - 1])
        # TODO -- MAGIC END --

        # Permitted ranges for series:
        yKey = fn.split('-')[0]
        if yKey == 'NGR':
            val = 2
        elif yKey == 'ITTR':
            val = 60
        elif yKey == 'FCR':
            val = 20
        else:   # 'OILT' & 'SPR' are currently not defined
            val = None
        if val is not None:
            df['rangeMax'] = val
            df['rangeMin'] = -val
            keys = ['delta', 'mean', 'y_linreg', 'y_rolling', 'trend', 'rangeMax', 'rangeMin']  # , 'y_polyreg'
        else:
            keys = ['delta', 'mean', 'y_linreg', 'y_rolling', 'trend']

        allTitles.append(f"{fn} for engine id {engineId}")
        allLabels.append(','.join([datetime.utcfromtimestamp(dt.astype(datetime)/1e9).strftime('"%Y-%m-%d %H:%M"') for dt in df.index.values]))

        datasets = []
        for color, key in zip(colors, keys):
            data = ','.join([f'{float(a):.2f}' for a in df[key].values])
            unit = ''
            ds = Dataset(label=key, unit=unit, data=data, color=color)
            datasets.append(ds)
        allDatasets.append(datasets)

    return render_template('charts.html', aspectRatio=3, titles=allTitles, labels=allLabels, datasets=allDatasets)

# -----------------------------------------------------------------------------

# @app.route('/pokus')
# def pokus():
#     engineId = 3
#     flightId = 1800
#     flightIdx = 2
#     cycleId = 4235
#     cycleIdx = 0
#
#     df: DataFrame = flightRecordingDao.loadDf(engineId=engineId, flightId=flightId, flightIdx=flightIdx, cycleId=cycleId, cycleIdx=cycleIdx)
#
#     title = f"Flight recording for engineId = {engineId}, flightId = {flightId}, idx = {flightIdx}, cycleId = {cycleId}, idx = {cycleId}"
#
#     labels = ','.join([datetime.utcfromtimestamp(dt.astype(datetime)/1e9).strftime('"%Y-%m-%d %H:%M"') for dt in df.index.values])
#
#     keys = ('ALT', 'IAS', 'ITT', 'NG', 'NP')
#     colors = ('rgba(0, 0, 255, 1)', 'rgba(0, 255, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(255, 0, 255, 1)', 'rgba(0, 255, 255, 1)')
#     datasets = []
#     for color, key in zip(colors, keys):
#         data = ','.join([f'{float(a):.0f}' for a in df[key].values])
#         ds = Dataset(label=key, data=data, color=color)
#         datasets.append(ds)
#
#     return render_template('pokus.html', id=1, title=title, labels=labels, datasets=datasets)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = DEBUG
    app.run(debug=DEBUG)
