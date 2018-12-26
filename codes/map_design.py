#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:14:36 2018

@author: louisgiron
"""
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


class displayMap:

    def __init__(self, data, values):
        self.data = data
        self.long = data['longitude']
        self.lat = data['latitude']
        self.val = data[values]

    def display_map(self):
        plt.figure(figsize=(12, 15))

        map1 = Basemap(projection='merc', resolution='h', llcrnrlat=-13,
                       urcrnrlat=0, llcrnrlon=28, urcrnrlon=41)

        # Create the map 1
        map1.drawcoastlines()
        map1.fillcontinents(color='tan', lake_color='lightblue')
        # draw parallels and meridians.
        map1.drawparallels(np.arange(-90., 91., 30.),
                           labels=[True, True, False, False], dashes=[2, 2])
        map1.drawmeridians(np.arange(-180., 181., 60.),
                           labels=[False, False, False, True], dashes=[2, 2])
        map1.drawmapboundary(fill_color='lightblue')
        map1.drawstates(linewidth=0.5, linestyle='solid', color='k')
        map1.drawrivers(linewidth=0.5, linestyle='solid', color='blue')

        # Plot labels on map 1
        # fit position data as regards map object
        ind_class_0 = np.argwhere(self.val.values == 'functional')
        ind_class_1 = np.argwhere(self.val.values == 'functional needs repair')
        ind_class_2 = np.argwhere(self.val.values == 'non functional')

        perc_class_0 = np.round(len(ind_class_0)/len(self.val.values)*100,
                                decimals=1)
        perc_class_1 = np.round(len(ind_class_1)/len(self.val.values)*100,
                                decimals=1)
        perc_class_2 = np.round(len(ind_class_2)/len(self.val.values)*100,
                                decimals=1)

        long_2, lat_2 = (self.long.values[ind_class_2],
                         self.lat.values[ind_class_2])

        x2, y2 = map1(long_2, lat_2)

        map1.scatter(x2, y2, marker='o', color='red', zorder=10, s=10,
                     label='non functional - ' + str(perc_class_2) + '%')

        long_0, lat_0 = (self.long.values[ind_class_0],
                         self.lat.values[ind_class_0])

        x0, y0 = map1(long_0, lat_0)

        map1.scatter(x0, y0, marker='o', color='green', zorder=10, s=10,
                     label='functional - ' + str(perc_class_0) + '%')

        long_1, lat_1 = (self.long.values[ind_class_1],
                         self.lat.values[ind_class_1])

        x1, y1 = map1(long_1, lat_1)

        map1.scatter(x1, y1, marker='o', color='orange', zorder=10, s=10,
                     label='functional needs repair - ' +
                     str(perc_class_1) + '%')

        plt.legend()
        plt.title('Geographical distribution of waterpoint by ' +
                  'functionality in Tanzania')
