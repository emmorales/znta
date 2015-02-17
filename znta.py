# ========================================================
# Title : ZNTA Solution
# Author: Eric Morfa Morales
# Email : eric.m.morales@gmail.com
# ========================================================

import numpy as np
import matplotlib.pyplot as plt

def createGrid(xMin, xMax, xNum, yMin, yMax, yNum):
    xPoints = np.linspace(xMin, xMax, num=xNum)
    yPoints = np.linspace(yMin, yMax, num=yNum)
    return np.meshgrid(xPoints, yPoints, indexing='xy')

def getRiverPoints():
    fileRiver = open('spree.csv', 'r')
    coordRiver = []
    for l in fileRiver:
        lat, lon = l.split(',')
        coordRiver.append([float(lat),float(lon)])
    fileRiver.close()
    return np.apply_along_axis(convertGPStoXY, axis=1, arr=coordRiver)

def convertGPStoXY(p):
    SW_lat = 52.464011
    SW_lon = 13.274099
    Px = -(p[1] - SW_lon) * np.cos(SW_lat) * 111.323
    Py = (p[0] - SW_lat) * 111.323
    return [Px, Py]

def convertXYtoGPS(p):
    SW_lat = 52.464011
    SW_lon = 13.274099    
    P_Lat = p[1]/111.323 + SW_lat
    P_Lon = SW_lon - p[0]/(111.323*np.cos(SW_lat))
    return [P_Lat, P_Lon]

def createLinePointsRiver(p):
    p1 = np.delete(p, -1,axis=0)
    p2 = np.delete(p,  0,axis=0)
    return [[p1[i],p2[i]] for i in range(0,len(p1))]

def distToSingleLine(x, y, lpt):
    ax, ay = lpt[0][0], lpt[0][1]
    bx, by = lpt[1][0], lpt[1][1]
    u = ((x-bx)*(ax-bx)+(y-by)*(ay-by))/((ax-bx)**2+(ay-by)**2)          # parameter of projected point along line
    ua, ub, uc = np.array(u>1), np.array(u<0), np.array((u>=0) & (u<=1)) # select parameter values for projected point          
    A = np.sqrt((x-ax)**2+(y-ay)**2) * ua                                # project computed values
    B = np.sqrt((x-bx)**2+(y-by)**2) * ub                                # project computed values
    C = np.sqrt((x-bx-u*(ax-bx))**2+(y-by-u*(ay-by))**2) * uc            # project computed values
    return A+B+C                                                         
                     
def distToMultipleLines(x, y, lpt):
    dist = np.array([distToSingleLine(x,y,l) for l in lpt])              # distance of each line to every point
    nLines, nycoord, nxcoord = dist.shape[0], dist.shape[1], dist.shape[2]
    D = []                                                                   
    for j in range(0,nycoord):
        for k in range(0,nxcoord):
            d = []
            for n in range(0,nLines):
                d.append(dist[n][j][k])
            D.append(min(d))                                             
    return np.array(D).reshape(nycoord,nxcoord)                          # distance of each point to closest line

def distToGate(x, y, coordBBG):
    xBBG, yBBG = coordBBG[0], coordBBG[1]
    return np.sqrt((xBBG-x)**2+(yBBG-y)**2)

def applyGaussian(mu, sigma, d):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((d-mu)/sigma)**2)

def applyLogNormal(mu, mode, d):
    sigma = np.sqrt(mu-np.log(mode))
    return 1/(d*sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((np.log(d)-mu)/sigma)**2)

def createRiverXY(rpt):
    rptx = [p[0] for p in rpt]
    rpty = [p[1] for p in rpt]
    return [rptx, rpty]

def plotHeatMap(x, y, data, plotbounds, plottitle):
    plt.pcolor(x, y, data, cmap='GnBu')
    plt.title(plottitle)
    plt.xlabel('West-East Direction [km]')
    plt.ylabel('South-North Direction [km]')
    plt.axis(plotbounds)    
    plt.colorbar()
    
def main():
    # create grid of map points (make gxNum, gyNum larger to make grid finer)
    gxMin, gxMax, gxNum =  -2, 20, 200
    gyMin, gyMax, gyNum =  -5, 16, 200
    grid = createGrid(gxMin, gxMax, gxNum, gyMin, gyMax, gyNum)
    x,y = grid
    
    # River Spree
    muRiver    = 0
    sigmaRiver = 2.730/2
    ptRiver    = getRiverPoints()
    lptRiver   = createLinePointsRiver(ptRiver)
    distRiver  = distToMultipleLines(x, y, lptRiver)
    probRiver  = applyGaussian(muRiver, sigmaRiver, distRiver)           # probability distribution around River Spree
    RiverXY    = createRiverXY(ptRiver)

    # Satellite (approximating the path with a straight line is sufficient in a 20km x 20 km area)
    muSat       = 0
    sigmaSat    = 2.4/2
    SatStartGPS = [52.590117,13.39915]
    SatEndGPS   = [52.437385,13.553989]
    SatStartXY  = convertGPStoXY(SatStartGPS)
    SatEndXY    = convertGPStoXY(SatEndGPS)
    SatX, SatY  = [SatStartXY[0], SatEndXY[0]], [SatStartXY[1], SatEndXY[1]]
    lptSat      = [SatStartXY, SatEndXY]
    distSat     = distToSingleLine(x, y, lptSat)
    probSat     = applyGaussian(muSat, sigmaSat, distSat)                # probability distribution around Satellite
    
    # Brandenburg Gate
    muGate    = 4.7
    modeGate  = 3.877
    GPSGate   = [52.516288,13.377689]
    coordGate = convertGPStoXY(GPSGate)
    distGate  = distToGate(x, y, coordGate)
    probGate  = applyLogNormal(muGate, modeGate, distGate)               # probability distribution around Brandenburg Gate
    
    # joint distribution
    prob = probRiver * probSat * probGate
    
    # find position of analyst
    maxindex = np.unravel_index(prob.argmax(), prob.shape)               # find the index of the largest value
    posXY    = [x[maxindex], y[maxindex]]
    posGPS   = convertXYtoGPS(posXY)
    textLat  = 'Latitude:' + str(posGPS[0])
    textLon  = 'Longitude:' + str(posGPS[1])
    print 'x-coordinate of analyst:', posXY[0]
    print 'y-coordinate of analyst:', posXY[1]
    print 'GPS-coordinate of analyst (Latitude) :', posGPS[0]
    print 'GPS-coordinate of analyst (Longitute):', posGPS[1]
    
    # plotting
    plotbounds = [gxMin, gxMax, gyMin, gyMax]
    plt.figure(num=None, figsize=(8,8), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(221)
    plotHeatMap(x, y, probRiver, plotbounds, 'Gaussian Distribution around\n Spree River')
    plt.plot(RiverXY[0], RiverXY[1], 'o-', color='orange')
    
    plt.subplot(222)
    plotHeatMap(x, y, probSat, plotbounds, 'Gaussian Distribution around\n Satellite Path')
    plt.plot(SatX,SatY, 'o-', color='orange')
    
    plt.subplot(223)
    plotHeatMap(x, y, probGate, plotbounds, 'Log-Normal Distribution around\n Brandenburg Gate')
    plt.plot([coordGate[0]],[coordGate[1]], 'o', color='orange')
    
    plt.subplot(224)
    plotHeatMap(x, y, prob, plotbounds, 'Joint Probability Distribution and\n Position of Analyst')   
    plt.plot([posXY[0]],[posXY[1]], 'o', color='orange')
    plt.annotate(textLat,xy=(-1,-1.7))
    plt.annotate(textLon,xy=(-1,-3))
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
