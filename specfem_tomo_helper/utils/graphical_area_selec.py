#!/usr/bin/env python

import cartopy
import netCDF4
import numpy as np
from pyproj import Proj
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, RadioButtons
import sys


class param_container:
    def __init__(self, model, projection):
        self.model = model
        self.projection = projection
        self.hemisphere = "+north"
        self.x0 = None
        self.y0 = None
        self.easting = None
        self.northing = None


def maptool(model, myProj):

    holder = param_container(model, myProj)

    def new_projection(x0, y0, easting, northing, projection):
        points = []
        ux0, uy0 = projection(x0, y0)
        # print(ux0, uy0, ux0+easting*1000, uy0+northing*1000)
        e, n = easting*1000, northing*1000
        points.append(projection(ux0, uy0, 'inverse'))
        points.append(projection(ux0+e, uy0, 'inverse'))
        points.append(projection(ux0+e, uy0+n, 'inverse'))
        points.append(projection(ux0, uy0+n, 'inverse'))
        points.append(projection(ux0, uy0, 'inverse'))
        return np.asarray(points).T[0, :], np.asarray(points).T[1, :]

    def print_result(x0, y0, easting, northing, projection):
        ux0, uy0 = projection(x0, y0)
        print('LATITUDE_MIN                    = '+f'{uy0:.1f}'+'\n'
              + 'LATITUDE_MAX                    = '+f'{uy0+northing*1e3:.1f}'+'\n'
              + 'LONGITUDE_MIN                   = '+f'{ux0:.1f}'+'\n'
              + 'LONGITUDE_MAX                   = '+f'{ux0+easting*1e3:.1f}')
        return np.asarray([ux0, uy0, ux0+easting*1e3, uy0+northing*1e3])

    # Extract lat, lon info from model
    lat = holder.model.lat
    lon = holder.model.lon
    # Extracting coordinates and creating model polygon
    model_lat = lat.min(), lat.max()
    model_lon = lon.min(), lon.max()
    model_x_coordinates, model_y_coordinates = np.meshgrid(model_lon, model_lat, indexing="ij")
    poly_corners = np.vstack((model_x_coordinates.flatten(),
                              np.roll(model_y_coordinates.T.flatten(), 1))).T

    # Defining figure
    fig, ax = plt.subplots(figsize=(10, 7.5))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    # ---------------------------------------------
    # Axes generation (plot + widget boxes)
    axcolor = 'lightgoldenrodyellow'
    ax_lat = plt.axes([0.25, 0.1, 0.60, 0.03], facecolor=axcolor)
    ax_lon = plt.axes([0.25, 0.15, 0.60, 0.03], facecolor=axcolor)
    east_box = plt.axes([0.1, 0.25, 0.1, 0.05])
    north_box = plt.axes([0.1, 0.30, 0.1, 0.05])
    utm_box = plt.axes([0.1, 0.37, 0.1, 0.05])
    export_val_box = plt.axes([0.77, 0.03, 0.1, 0.04])
    rax = plt.axes([0.1, 0.45, 0.1, 0.08])
    # --------------------------------------------
    # Widgets initialisation
    box_lat = Slider(ax_lat, 'Latitude', model_lat[0], model_lat[1], valinit=lat.min())
    box_lon = Slider(ax_lon, 'Longitude', model_lon[0], model_lon[1], valinit=lon.min())
    holder.northing = 400
    holder.easting = 500
    initial_UTM_zone = myProj.crs.utm_zone
    button_easting = TextBox(east_box, 'Easting', initial=holder.easting)
    button_northing = TextBox(north_box, 'Northing', initial=holder.northing)
    button_utm = TextBox(utm_box, 'UTM zone', initial=initial_UTM_zone)
    button_save = Button(export_val_box, 'Export values')
    radio = RadioButtons(rax, ('North', 'South'))
    # --------------------------------------------
    # Def updating figure functions

    def update(val):
        holder.y0 = box_lat.val
        holder.x0 = box_lon.val
        projected = new_projection(holder.x0, holder.y0, holder.easting,
                                   holder.northing, holder.projection)
        l.set_ydata(projected[1])
        l.set_xdata(projected[0])
        fig.canvas.draw_idle()

    def submit_utm(text):
        holder.projection = Proj("+proj=utm +zone="+str(text) +
                                 " "+holder.hemisphere+" +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        coords = np.asarray(l.get_data()).T[0]
        projected = new_projection(coords[0], coords[1], holder.easting,
                                   holder.northing, holder.projection)
        l.set_ydata(projected[1])
        l.set_xdata(projected[0])
        fig.canvas.draw_idle()

    def submit_zone(label):
        hzdict = {'North': '+north', 'South': '+south'}
        holder.hemisphere = hzdict[label]
        holder.projection = Proj("+proj=utm +zone="+str(button_utm.text) +
                                 " "+holder.hemisphere+" +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        coords = np.asarray(l.get_data()).T[0]
        projected = new_projection(coords[0], coords[1], holder.easting,
                                   holder.northing, holder.projection)
        l.set_ydata(projected[1])
        l.set_xdata(projected[0])
        fig.canvas.draw_idle()

    def submit_easting(text):
        holder.easting = np.float(text)
        # print(easting)
        coords = np.asarray(l.get_data()).T[0]
        projected = new_projection(coords[0], coords[1], holder.easting,
                                   holder.northing, holder.projection)
        # print(projected)
        l.set_ydata(projected[1])
        l.set_xdata(projected[0])
        fig.canvas.draw_idle()

    #
    def submit_northing(text):
        holder.northing = np.float(text)
        coords = np.asarray(l.get_data()).T[0]
        projected = new_projection(coords[0], coords[1], holder.easting,
                                   holder.northing, holder.projection)
        l.set_ydata(projected[1])
        l.set_xdata(projected[0])
        fig.canvas.draw_idle()

    # function that print the utm coordinates and close the figure.
    def submit_save(function):
        coords = np.asarray(l.get_data()).T[0]
        projected = new_projection(coords[0], coords[1], holder.easting,
                                   holder.northing, holder.projection)
        print_result(coords[0], coords[1], holder.easting, holder.northing, holder.projection)

        holder.ux0, holder.uy0 = holder.projection(coords[0], coords[1])
        holder.extent = np.asarray([holder.ux0, holder.ux0+holder.easting*1e3,
                                    holder.uy0, holder.uy0+holder.northing*1e3])

        plt.close('all')
        print(holder.projection)

    # --------------------------------------------
    # Def action deifinition
    box_lat.on_changed(update)  # run update() on slider action
    box_lon.on_changed(update)  # run update() on slider action
    button_utm.on_submit(submit_utm)  # run submit_utm() on textbox modification
    button_easting.on_submit(submit_easting)  # run submit_easting() on textbox modification
    button_northing.on_submit(submit_northing)  # run submit_northing() on textbox modification
    button_save.on_clicked(submit_save)  # run submit_save() and close fig on button press
    radio.on_clicked(submit_zone)

    # --------------------------------------------
    # Initial map plotting and plot parameters
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.stock_img()
    ax.coastlines()
    ax.set_extent([model_lon[0]-1, model_lon[1]+1, model_lat[0]-1, model_lat[1]+1])
    # Plot model bounds (red)
    poly = mpatches.Polygon(poly_corners, closed=True, ec='r', fill=False,
                            lw=1, fc=None, transform=ccrs.PlateCarree())
    plotted_poly = ax.add_patch(poly)
    # Plot interpolation zone (blue)
    l, = ax.plot(new_projection(lon.min(), lat.min(),
                                holder.easting, holder.northing, holder.projection)[0],
                 new_projection(lon.min(), lat.min(),
                                holder.easting, holder.northing, holder.projection)[1])

    plt.show()
    return(holder)
