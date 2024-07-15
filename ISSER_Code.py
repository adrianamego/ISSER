# -*- coding: utf-8 -*-
"""
Created on Sun May  8 05:50:21 2022
Codigo para realizar el modelado espacial de los sucesos finales
Requiere disponer de la imagen georeferencida del sitio del modelado
Requiere el Shapefile de los elementos de la zona del modelado con la estructura 
de columnas para que puedan ser procesadas por el codigo.
@author: Adriana Mesa
Doctora en Ingeniería de procesos químicos
Se dispone de esta publicacion con fines académicos y su uso para otros fines
debe solicitarse por escrito.
"""
############################################################################################

from tkinter import messagebox
import random
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox
from pyproj import Proj, transform
from shapely.geometry import Point
from tqdm import tqdm
from shapely.geometry import Polygon
from glob import glob
from scipy.io import loadmat
from osgeo import gdal as GD  
import matplotlib.pyplot as mplot
from pyproj import Proj
import matplotlib.pyplot as plot  
import seaborn as sns
sns.set(style="darkgrid")
from time import time
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import ndimage as ndi
import scipy.misc
from random import randint
from itertools import zip_longest
import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage
from tkinter import DoubleVar
from skimage import feature
from tkinter import Canvas
from tkinter import IntVar
from tkinter import Label
from tkinter import Entry
from tkinter import Menu
from tkinter import Tk
from tkinter import NW
#import pandas_bokeh
import csv
import os.path ##Cargue
from datetime import datetime, date, time, timedelta
from datetime import *
import time
import calendar
import re
from math import pi
import os
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange, linspace, pi, sin
import statistics
import scipy
from pandastable import Table, TableModel
from pandas.core.computation.ops import isnumeric
import webbrowser
import sys
import os  
import joblib 
from numpy import mean
from numpy import std
from matplotlib import pyplot
import xarray as xr
from osgeo import gdal, ogr, osr
import geopandas as gpd
from shapely.geometry import shape,Polygon,MultiPolygon,mapping
from mpl_toolkits.axes_grid1 import make_axes_locatable
import folium
import branca.colormap as cm
import requests
import zipfile
from scipy.interpolate import griddata
from functools import partial
import fiona
from shapely.geometry import shape, Point
import mplcursors
from tkinter import messagebox
from shapely.geometry import MultiPoint
import os




############################################################################################
df00 = pd.DataFrame(columns=['Class', 'Impact Radius', 'Probability'])  #Flash fire
df11 = pd.DataFrame(columns=['Class', 'Impact Radius', 'Probability'])  #Jet fire
df22 = pd.DataFrame(columns=['Class', 'Impact Radius', 'Probability'])  #Pool fire
df33 = pd.DataFrame(columns=['Class', 'Impact Radius', 'Probability'])  #Over pressure


##############################################################################################

#Read file

class DataFrameTable(tk.Frame):
    def __init__(self, parent=None, df=pd.DataFrame()):
        super().__init__()
        self.parent = parent
        self.pack(fill=tk.BOTH, expand=True)
        self.table = Table(
            self, dataframe=df,
            showtoolbar=False,
            showstatusbar=True,
            editable=False)
        self.table.show()
##############################################################################################
def add_shapefile():
    ventana.filename = filedialog.askopenfilename(initialdir = ruta,title = "Select buildings shapefile:", filetypes = (("Shapefiles", "*.shp"), ("all files", "*.*")))    
    global ruta130    
    ruta130 = ventana.filename
    # Load the new shapefile
    # new_gdf = gpd.read_file(ruta130)
    # shapefiles.append(new_gdf)
    
    try:
        # Load the TIFF file
        from osgeo import gdal
        ds = gdal.Open(ruta120)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]
        # Load the shapefile
        centroids = [] #Empy
               
        with fiona.open(ruta) as f:
            # Iterates over all the entries in the shapefile
            for feature in f:
                # Gets the geometry of the polygon
                polygon = shape(feature['geometry'])
                # Calculate the centroid of the polygon
                centroid = polygon.centroid
                # Stores the coordinates of the centroid in the list
                centroids.append((centroid.x, centroid.y))
                
    
        # DataFrame centroids
        df5 = pd.DataFrame(centroids, columns=['xc', 'yc'])
        gdf1 = gpd.read_file(ruta)    
        gdf = pd.concat([gdf1, df5], axis=1)
        gdf = gdf.reset_index(inplace=False, drop=True)
        gdf['IDTK'] = gdf.index + 1

        root = tk.Tk()

        # Create figure and axis for plotting
        fig, ax = plt.subplots()

        # Plot the TIFF data on the axis with its original coordinate reference system
        show(data, ax=ax, transform=src.transform)

        # Overlay the shapefile on top of the TIFF data
        gdf.plot(ax=ax, facecolor='none', edgecolor='red')

        # Create a Tk canvas for the plot
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        # Add the toolbar to the canvas
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        def add_shapefile():
            # Load the new shapefile
            new_gdf = gpd.read_file(ruta130)
            # Overlay the new shapefile on top of the existing data
            new_gdf.plot(ax=ax, facecolor='none', edgecolor='blue')
            # Redraw the canvas
            canvas.draw()
        # Add a button to the canvas that will call the `add_shapefile` function
        button = tk.Button(root, text="Add Shapefile", command=add_shapefile)
        button.pack(side='bottom')
        for index, row in gdf.iterrows():
            ax.annotate(row['IDTK'], (row.xc, row.yc), color='red', )

        root.mainloop()
    except Exception as e:
        # Load the shapefile
        centroids = [] #Empy
               
        with fiona.open(ruta) as f:
            # Iterates over all the entries in the shapefile
            for feature in f:
                # Gets the geometry of the polygon
                polygon = shape(feature['geometry'])
                # Calculate the centroid of the polygon
                centroid = polygon.centroid
                # Stores the coordinates of the centroid in the list
                centroids.append((centroid.x, centroid.y))
                
    
        # DataFrame centroids
        df5 = pd.DataFrame(centroids, columns=['xc', 'yc'])
        gdf1 = gpd.read_file(ruta)    
        gdf = pd.concat([gdf1, df5], axis=1)
        gdf = gdf.reset_index(inplace=False, drop=True)
        gdf['IDTK'] = gdf.index + 1

        root = tk.Tk()

        # Create figure and axis for plotting
        fig, ax = plt.subplots()

        # Overlay the shapefile on top of the TIFF data
        gdf.plot(ax=ax, facecolor='none', edgecolor='red')

        # Create a Tk canvas for the plot
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        # Add the toolbar to the canvas
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        def add_shapefile():
            # Load the new shapefile
            new_gdf = gpd.read_file(ruta130)
            # Overlay the new shapefile on top of the existing data
            new_gdf.plot(ax=ax, facecolor='none', edgecolor='blue')
            # Redraw the canvas
            canvas.draw()
        # Add a button to the canvas that will call the `add_shapefile` function
        button = tk.Button(root, text="Add Shapefile", command=add_shapefile)
        button.pack(side='bottom')
        for index, row in gdf.iterrows():
            ax.annotate(row['IDTK'], (row.xc, row.yc), color='red', )
        
        root.mainloop()
    
############################################################################################
# Definiendo los dataframe por defecto
columns = ['Class', 'Impact Radius', 'Probability', 'Probit People', 'Probit House', 'Probit IAp', 'Probit INps', 'Probit LNps', 'Probit INss', 'Probit LNss',  'Probit Ewater', 'Probit Enlc', 'Probit Eforest']

# Definiendo los datos
data = [
    ("LFL", 0, 1, 1, 1, 1, 0.25, 0, 0.25, 0, 0, 1, 1),
    ("1/2 LFL", 0, 1, 1, 1, 1, 0.5, 0.1, 0.5, 0.1, 0, 1, 1),
    ]

# Creando el DataFrame
df00 = pd.DataFrame(data, columns=columns)



#Table input Flash fire RIP
def table():
    from  tkinter import ttk
    ws  = Tk()
    ws.title('Potential Impact Radius Flash Fire')
    ws.geometry('1320x440')
    ws['bg'] = '#c2c4c3'
    game_frame = Frame(ws)
    game_frame.pack()
    
    
    #scrollbar
    game_scroll = Scrollbar(game_frame)
    game_scroll.pack(side=RIGHT, fill=Y)

    game_scroll = Scrollbar(game_frame,orient='horizontal')
    game_scroll.pack(side= BOTTOM,fill=X)

    my_game = ttk.Treeview(game_frame,yscrollcommand=game_scroll.set, xscrollcommand =game_scroll.set)


    my_game.pack()

    game_scroll.config(command=my_game.yview)
    game_scroll.config(command=my_game.xview)

    #define our column
     
    my_game['columns'] = ('Thermal Radiation [Kw/m2]', 'Impact Radius Length [m]', 'Probability of Death', 'Probit People','Probit House','Probit IAp', 'Probit INps', 'Probit LNps', 'Probit INss', 'Probit LNss', 'Probit Ewater', 'Probit Enlc', 'Probit Eforest')

    # format our column
    my_game.column("#0", width=0,  stretch=NO)
    my_game.column("Thermal Radiation [Kw/m2]",anchor=CENTER, width=100)
    my_game.column("Impact Radius Length [m]",anchor=CENTER,width=80)
    my_game.column("Probability of Death",anchor=CENTER,width=100)
    my_game.column("Probit People",anchor=CENTER,width=100)
    my_game.column("Probit House",anchor=CENTER,width=100)
    my_game.column("Probit IAp",anchor=CENTER,width=100)
    my_game.column("Probit INps",anchor=CENTER,width=100)
    my_game.column("Probit LNps",anchor=CENTER,width=100)
    my_game.column("Probit INss",anchor=CENTER,width=100)
    my_game.column("Probit LNss",anchor=CENTER,width=100)
    my_game.column("Probit Ewater",anchor=CENTER,width=100)
    my_game.column("Probit Enlc",anchor=CENTER,width=100)
    my_game.column("Probit Eforest",anchor=CENTER,width=100)
    
    


    #Create Headings 
    my_game.heading("#0",text="",anchor=CENTER)
    my_game.heading("Thermal Radiation [Kw/m2]",text="Thermal Radiation [Kw/m2]",anchor=CENTER)
    my_game.heading("Impact Radius Length [m]",text="Impact Radius Length [m]",anchor=CENTER)
    my_game.heading("Probability of Death",text="Probability of Death",anchor=CENTER)
    my_game.heading("Probit People",text="Probit People",anchor=CENTER)
    my_game.heading("Probit House",text="Probit House",anchor=CENTER)
    my_game.heading("Probit IAp",text="Probit IAp",anchor=CENTER)
    my_game.heading("Probit INps",text="Probit INps",anchor=CENTER)
    my_game.heading("Probit LNps",text="Probit LNps",anchor=CENTER)
    my_game.heading("Probit INss",text="Probit INss",anchor=CENTER)
    my_game.heading("Probit LNss",text="Probit LNss",anchor=CENTER)
    my_game.heading("Probit Ewater",text="Probit Ewater",anchor=CENTER)
    my_game.heading("Probit Enlc",text="Probit Enlc",anchor=CENTER)
    my_game.heading("Probit Eforest",text="Probit Eforest",anchor=CENTER)
    

    #add data tank
    data = [
        ("LFL", "0", "1", "1", "1", "1", "0.25", "0", "0.25", "0", "0", "1", "1"),
        ("1/2 LFL", "0", "1", "1", "1", "1", "0.5", "0.1", "0.5", "0.1", "0", "1", "1"),
    ]
    
    for i, row in enumerate(data):
        my_game.insert("", "end", iid=i, values=row)
    

    my_game.pack()

    frame = Frame(ws)
    frame.pack(pady=10)
    
       
    #labels
    
    playerid= Label(frame,text = "Thermal Radiation [Kw/m2]")
    playerid.grid(row=0,column=0 )

    playername = Label(frame,text="Impact Radius Length [m]")
    playername.grid(row=0,column=1)

    playerrank = Label(frame,text="Probability of Death")
    playerrank.grid(row=0,column=2)
    
    playerrank1 = Label(frame,text="Probit People")
    playerrank1.grid(row=0,column=3)
    
    playerrank2 = Label(frame,text="Probit House")
    playerrank2.grid(row=0,column=4)
    
    playerrank3 = Label(frame,text="Probit IAp")
    playerrank3.grid(row=0,column=5)
    
    playerrank4 = Label(frame,text="Probit INps")
    playerrank4.grid(row=0,column=6)
    
    playerrank5 = Label(frame,text="Probit LNps")
    playerrank5.grid(row=0,column=7)
    
    playerrank6 = Label(frame,text="Probit INss")
    playerrank6.grid(row=0,column=8)
    
    playerrank7 = Label(frame,text="Probit LNss")
    playerrank7.grid(row=0,column=9)
    
    playerrank8 = Label(frame,text="Probit Ewater")
    playerrank8.grid(row=0,column=10)
    
    playerrank9 = Label(frame,text="Probit Enlc")
    playerrank9.grid(row=0,column=11)
    
    playerrank10 = Label(frame,text="Probit Eforest")
    playerrank10.grid(row=0,column=12)
    
       
    #Entry boxes
    playerid_entry = StringVar()
    playername_entry = StringVar()
    playerrank_entry = StringVar()
    playerrank1_entry = StringVar()
    playerrank2_entry = StringVar()
    playerrank3_entry = StringVar()
    playerrank4_entry = StringVar()
    playerrank5_entry = StringVar()
    playerrank6_entry = StringVar()
    playerrank7_entry = StringVar()
    playerrank8_entry = StringVar()
    playerrank9_entry = StringVar()
    playerrank10_entry = StringVar()
    
    # Definir el ancho de las cajas de entrada
    entry_width = 14  # Este es un ejemplo, puedes ajustar el tamaño como necesites
    
        
    playerid_entry = Entry(frame, textvariable=playerid_entry, width=entry_width)
    playerid_entry.grid(row=1, column=0)
    
    playername_entry = Entry(frame, textvariable=playername_entry, width=entry_width)
    playername_entry.grid(row=1, column=1)
    
    playerrank_entry = Entry(frame, textvariable=playerrank_entry, width=entry_width)
    playerrank_entry.grid(row=1, column=2)
    
    playerrank1_entry = Entry(frame, textvariable=playerrank1_entry, width=entry_width)
    playerrank1_entry.grid(row=1, column=3)
    
    playerrank2_entry = Entry(frame, textvariable=playerrank2_entry, width=entry_width)
    playerrank2_entry.grid(row=1, column=4)
    
    playerrank3_entry = Entry(frame, textvariable=playerrank3_entry, width=entry_width)
    playerrank3_entry.grid(row=1, column=5)
    
    playerrank4_entry = Entry(frame, textvariable=playerrank4_entry, width=entry_width)
    playerrank4_entry.grid(row=1, column=6)
    
    playerrank5_entry = Entry(frame, textvariable=playerrank5_entry, width=entry_width)
    playerrank5_entry.grid(row=1, column=7)
    
    playerrank6_entry = Entry(frame, textvariable=playerrank6_entry, width=entry_width)
    playerrank6_entry.grid(row=1, column=8)
    
    playerrank7_entry = Entry(frame, textvariable=playerrank7_entry, width=entry_width)
    playerrank7_entry.grid(row=1, column=9)
    
    playerrank8_entry = Entry(frame, textvariable=playerrank8_entry, width=entry_width)
    playerrank8_entry.grid(row=1, column=10)
    
    playerrank9_entry = Entry(frame, textvariable=playerrank9_entry, width=entry_width)
    playerrank9_entry.grid(row=1, column=11)
    
    playerrank10_entry = Entry(frame, textvariable=playerrank10_entry, width=entry_width)
    playerrank10_entry.grid(row=1, column=12)

    
    #Select Record
    def select_record():
        #clear entry boxes
        playerid_entry.delete(0,END)
        playername_entry.delete(0,END)
        playerrank_entry.delete(0,END)
        playerrank1_entry.delete(0,END)
        playerrank2_entry.delete(0,END)
        playerrank3_entry.delete(0,END)
        playerrank4_entry.delete(0,END)
        playerrank5_entry.delete(0,END)
        playerrank6_entry.delete(0,END)
        playerrank7_entry.delete(0,END)
        playerrank8_entry.delete(0,END)
        playerrank9_entry.delete(0,END)
        playerrank10_entry.delete(0,END)
                
        
        #grab record
        selected=my_game.focus()
        #grab record values
        values = my_game.item(selected,'values')
        #temp_label.config(text=selected)

        #output to entry boxes
        playerid_entry.insert(0,values[0])
        playername_entry.insert(0,values[1])
        playerrank_entry.insert(0,values[2])
        playerrank1_entry.insert(0,values[3])
        playerrank2_entry.insert(0,values[4])
        playerrank3_entry.insert(0,values[5])
        playerrank4_entry.insert(0,values[6])
        playerrank5_entry.insert(0,values[7])
        playerrank6_entry.insert(0,values[8])
        playerrank7_entry.insert(0,values[9])
        playerrank8_entry.insert(0,values[10])
        playerrank9_entry.insert(0,values[11])
        playerrank10_entry.insert(0,values[12])
        

    #save Record    
    def funcion_prueba ():
        playerid_entry = StringVar()
        playername_entry = StringVar()
        playerrank_entry = StringVar()
        playerrank1_entry = StringVar()
        playerrank2_entry = StringVar()
        playerrank3_entry = StringVar()
        playerrank4_entry = StringVar()
        playerrank5_entry = StringVar()
        playerrank6_entry = StringVar()
        playerrank7_entry = StringVar()
        playerrank8_entry = StringVar()
        playerrank9_entry = StringVar()
        playerrank10_entry = StringVar()
        
      
    diccionario = {}
    def inicio():
        diccionario[str(playerid_entry.get())] = [str(playerid_entry.get()),float(playername_entry.get()),float(playerrank_entry.get()),float(playerrank1_entry.get()),float(playerrank2_entry.get()),float(playerrank3_entry.get()),float(playerrank4_entry.get())
                                                  ,float(playerrank5_entry.get()),float(playerrank6_entry.get()),float(playerrank7_entry.get()),float(playerrank8_entry.get()),float(playerrank9_entry.get()),float(playerrank10_entry.get())]
        selected=my_game.focus()
        #save new data 
        my_game.item(selected,text="",values=(playerid_entry.get(),playername_entry.get(),playerrank_entry.get(),playerrank1_entry.get(),playerrank2_entry.get(),playerrank3_entry.get(),playerrank4_entry.get(),playerrank5_entry.get()
                                              ,playerrank6_entry.get(),playerrank7_entry.get(),playerrank8_entry.get(),playerrank9_entry.get(),playerrank10_entry.get()))
    
    def update_record():
        selected=my_game.focus()
        global df00
        df00 = df00 * 0
        df00.drop(df00.index, inplace=True)
        df1 = pd.DataFrame(diccionario.values())
        df1.rename(columns={0: 'Class', 1: 'Impact Radius', 2: 'Probability', 3: 'Probit People', 4: 'Probit House', 5: 'Probit IAp', 6: 'Probit INps', 7: 'Probit LNps', 8: 'Probit INss', 9: 'Probit LNss', 10: 'Probit Ewater', 11: 'Probit Enlc', 12: 'Probit Eforest'}, inplace=True)
        df00 = df00.append(df1)
        df00 = df00[df00.iloc[:, 1] != 0]
        ws.quit()
        ws.destroy()
    #Buttons
    select_button = Button(ws,text="Select radiation level", command=select_record)
    select_button.pack(pady =8)

    edit_button = Button(ws,text="Update",command=inicio)
    edit_button.pack(pady = 8)
    #Storage table for process
    edit_button = Button(ws,text="Store and close",command=update_record)
    edit_button.pack(pady = 8)
    
    temp_label =Label(ws,text="")
    temp_label.pack()
    

    ws.mainloop()
    
###############################################################################################
#Table input jet fire RIP
columns = ['Class', 'Impact Radius', 'Probability', 'Probit People', 'Probit House', 'Probit IAp', 'Probit INps', 'Probit LNps', 'Probit INss', 'Probit LNss',  'Probit Ewater', 'Probit Enlc', 'Probit Eforest']

# Definiendo los nuevos datos para df11
data = [
    (">37.5", "0", "1", "1", "1", "1", "1", "1", "1", "1", "0", "1", "1"),
    (">20.9", "0", "0.83", "1", "1", "1", "0.5", "1", "0.5", "1", "0", "1", "1"),
    (">14.5", "0", "0.39", "1", "1", "1", "0.25", "0", "0.25", "0", "0", "1", "1"),
    (">9.5", "0", "0.04", "0.86", "0", "0", "0", "0", "0", "0", "0", "0.86", "0.66"),
    (">7.27", "0", "0", "0.5", "0", "0", "0", "0", "0", "0", "0", "0.5", "0"),
    (">5.0", "0", "0", "0.07", "0", "0", "0", "0", "0", "0", "0", "0.07", "0"),
    (">1.6", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"),
]

# Creando el DataFrame df11
df11 = pd.DataFrame(data, columns=columns)

def table11():
    from  tkinter import ttk
    ws  = Tk()
    ws.title('Potential Impact Radius Jet Fire')
    ws.geometry('1320x440')
    ws['bg'] = '#c2c4c3'
    game_frame = Frame(ws)
    game_frame.pack()
    
    
    #scrollbar
    game_scroll = Scrollbar(game_frame)
    game_scroll.pack(side=RIGHT, fill=Y)

    game_scroll = Scrollbar(game_frame,orient='horizontal')
    game_scroll.pack(side= BOTTOM,fill=X)

    my_game = ttk.Treeview(game_frame,yscrollcommand=game_scroll.set, xscrollcommand =game_scroll.set)


    my_game.pack()

    game_scroll.config(command=my_game.yview)
    game_scroll.config(command=my_game.xview)

    #define our column
     
    my_game['columns'] = ('Thermal Radiation [Kw/m2]', 'Impact Radius Length [m]', 'Probability of Death', 'Probit People','Probit House','Probit IAp', 'Probit INps', 'Probit LNps', 'Probit INss', 'Probit LNss', 'Probit Ewater', 'Probit Enlc', 'Probit Eforest')

    # format our column
    my_game.column("#0", width=0,  stretch=NO)
    my_game.column("Thermal Radiation [Kw/m2]",anchor=CENTER, width=100)
    my_game.column("Impact Radius Length [m]",anchor=CENTER,width=80)
    my_game.column("Probability of Death",anchor=CENTER,width=100)
    my_game.column("Probit People",anchor=CENTER,width=100)
    my_game.column("Probit House",anchor=CENTER,width=100)
    my_game.column("Probit IAp",anchor=CENTER,width=100)
    my_game.column("Probit INps",anchor=CENTER,width=100)
    my_game.column("Probit LNps",anchor=CENTER,width=100)
    my_game.column("Probit INss",anchor=CENTER,width=100)
    my_game.column("Probit LNss",anchor=CENTER,width=100)
    my_game.column("Probit Ewater",anchor=CENTER,width=100)
    my_game.column("Probit Enlc",anchor=CENTER,width=100)
    my_game.column("Probit Eforest",anchor=CENTER,width=100)
  
    #Create Headings 
    my_game.heading("#0",text="",anchor=CENTER)
    my_game.heading("Thermal Radiation [Kw/m2]",text="Thermal Radiation [Kw/m2]",anchor=CENTER)
    my_game.heading("Impact Radius Length [m]",text="Impact Radius Length [m]",anchor=CENTER)
    my_game.heading("Probability of Death",text="Probability of Death",anchor=CENTER)
    my_game.heading("Probit People",text="Probit People",anchor=CENTER)
    my_game.heading("Probit House",text="Probit House",anchor=CENTER)
    my_game.heading("Probit IAp",text="Probit IAp",anchor=CENTER)
    my_game.heading("Probit INps",text="Probit INps",anchor=CENTER)
    my_game.heading("Probit LNps",text="Probit LNps",anchor=CENTER)
    my_game.heading("Probit INss",text="Probit INss",anchor=CENTER)
    my_game.heading("Probit LNss",text="Probit LNss",anchor=CENTER)
    my_game.heading("Probit Ewater",text="Probit Ewater",anchor=CENTER)
    my_game.heading("Probit Enlc",text="Probit Enlc",anchor=CENTER)
    my_game.heading("Probit Eforest",text="Probit Eforest",anchor=CENTER)


    #add data tank
    data = [
    (">37.5", "0", "1", "1", "1", "1", "1", "1", "1", "1", "0", "1", "1"),
    (">20.9", "0", "0.83", "1", "1", "1", "0.5", "1", "0.5", "1", "0", "1", "1"),
    (">14.5", "0", "0.39", "1", "1", "1", "0.25", "0", "0.25", "0", "0", "1", "1"),
    (">9.5", "0", "0.04", "0.86", "0", "0", "0", "0", "0", "0", "0", "0.86", "0.66"),
    (">7.27", "0", "0", "0.5", "0", "0", "0", "0", "0", "0", "0", "0.5", "0"),
    (">5.0", "0", "0", "0.07", "0", "0", "0", "0", "0", "0", "0", "0.07", "0"),
    (">1.6", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"),
    ]
    
    for i, row in enumerate(data):
        my_game.insert("", "end", iid=i, values=row)
    

    my_game.pack()

    frame = Frame(ws)
    frame.pack(pady=12)

    #labels
    
    playerid= Label(frame,text = "Thermal Radiation [Kw/m2]")
    playerid.grid(row=0,column=0 )

    playername = Label(frame,text="Impact Radius Length [m]")
    playername.grid(row=0,column=1)

    playerrank = Label(frame,text="Probability of Death")
    playerrank.grid(row=0,column=2)
    
    playerrank1 = Label(frame,text="Probit People")
    playerrank1.grid(row=0,column=3)
    
    playerrank2 = Label(frame,text="Probit House")
    playerrank2.grid(row=0,column=4)
    
    playerrank3 = Label(frame,text="Probit IAp")
    playerrank3.grid(row=0,column=5)
    
    playerrank4 = Label(frame,text="Probit INps")
    playerrank4.grid(row=0,column=6)
    
    playerrank5 = Label(frame,text="Probit LNps")
    playerrank5.grid(row=0,column=7)
    
    playerrank6 = Label(frame,text="Probit INss")
    playerrank6.grid(row=0,column=8)
    
    playerrank7 = Label(frame,text="Probit LNss")
    playerrank7.grid(row=0,column=9)
    
    playerrank8 = Label(frame,text="Probit Ewater")
    playerrank8.grid(row=0,column=10)
    
    playerrank9 = Label(frame,text="Probit Enlc")
    playerrank9.grid(row=0,column=11)
    
    playerrank10 = Label(frame,text="Probit Eforest")
    playerrank10.grid(row=0,column=12)
    
       
    #Entry boxes
    playerid_entry = StringVar()
    playername_entry = StringVar()
    playerrank_entry = StringVar()
    playerrank1_entry = StringVar()
    playerrank2_entry = StringVar()
    playerrank3_entry = StringVar()
    playerrank4_entry = StringVar()
    playerrank5_entry = StringVar()
    playerrank6_entry = StringVar()
    playerrank7_entry = StringVar()
    playerrank8_entry = StringVar()
    playerrank9_entry = StringVar()
    playerrank10_entry = StringVar()
    
    # Definir el ancho de las cajas de entrada
    entry_width = 14  # Este es un ejemplo, puedes ajustar el tamaño como necesites
    
        
    playerid_entry = Entry(frame, textvariable=playerid_entry, width=entry_width)
    playerid_entry.grid(row=1, column=0)
    
    playername_entry = Entry(frame, textvariable=playername_entry, width=entry_width)
    playername_entry.grid(row=1, column=1)
    
    playerrank_entry = Entry(frame, textvariable=playerrank_entry, width=entry_width)
    playerrank_entry.grid(row=1, column=2)
    
    playerrank1_entry = Entry(frame, textvariable=playerrank1_entry, width=entry_width)
    playerrank1_entry.grid(row=1, column=3)
    
    playerrank2_entry = Entry(frame, textvariable=playerrank2_entry, width=entry_width)
    playerrank2_entry.grid(row=1, column=4)
    
    playerrank3_entry = Entry(frame, textvariable=playerrank3_entry, width=entry_width)
    playerrank3_entry.grid(row=1, column=5)
    
    playerrank4_entry = Entry(frame, textvariable=playerrank4_entry, width=entry_width)
    playerrank4_entry.grid(row=1, column=6)
    
    playerrank5_entry = Entry(frame, textvariable=playerrank5_entry, width=entry_width)
    playerrank5_entry.grid(row=1, column=7)
    
    playerrank6_entry = Entry(frame, textvariable=playerrank6_entry, width=entry_width)
    playerrank6_entry.grid(row=1, column=8)
    
    playerrank7_entry = Entry(frame, textvariable=playerrank7_entry, width=entry_width)
    playerrank7_entry.grid(row=1, column=9)
    
    playerrank8_entry = Entry(frame, textvariable=playerrank8_entry, width=entry_width)
    playerrank8_entry.grid(row=1, column=10)
    
    playerrank9_entry = Entry(frame, textvariable=playerrank9_entry, width=entry_width)
    playerrank9_entry.grid(row=1, column=11)
    
    playerrank10_entry = Entry(frame, textvariable=playerrank10_entry, width=entry_width)
    playerrank10_entry.grid(row=1, column=12)

    
    #Select Record
    def select_record():
        #clear entry boxes
        playerid_entry.delete(0,END)
        playername_entry.delete(0,END)
        playerrank_entry.delete(0,END)
        playerrank1_entry.delete(0,END)
        playerrank2_entry.delete(0,END)
        playerrank3_entry.delete(0,END)
        playerrank4_entry.delete(0,END)
        playerrank5_entry.delete(0,END)
        playerrank6_entry.delete(0,END)
        playerrank7_entry.delete(0,END)
        playerrank8_entry.delete(0,END)
        playerrank9_entry.delete(0,END)
        playerrank10_entry.delete(0,END)
                
        
        #grab record
        selected=my_game.focus()
        #grab record values
        values = my_game.item(selected,'values')
        #temp_label.config(text=selected)

        #output to entry boxes
        playerid_entry.insert(0,values[0])
        playername_entry.insert(0,values[1])
        playerrank_entry.insert(0,values[2])
        playerrank1_entry.insert(0,values[3])
        playerrank2_entry.insert(0,values[4])
        playerrank3_entry.insert(0,values[5])
        playerrank4_entry.insert(0,values[6])
        playerrank5_entry.insert(0,values[7])
        playerrank6_entry.insert(0,values[8])
        playerrank7_entry.insert(0,values[9])
        playerrank8_entry.insert(0,values[10])
        playerrank9_entry.insert(0,values[11])
        playerrank10_entry.insert(0,values[12])
        

    #save Record    
    def funcion_prueba ():
        playerid_entry = StringVar()
        playername_entry = StringVar()
        playerrank_entry = StringVar()
        playerrank1_entry = StringVar()
        playerrank2_entry = StringVar()
        playerrank3_entry = StringVar()
        playerrank4_entry = StringVar()
        playerrank5_entry = StringVar()
        playerrank6_entry = StringVar()
        playerrank7_entry = StringVar()
        playerrank8_entry = StringVar()
        playerrank9_entry = StringVar()
        playerrank10_entry = StringVar()
        
      
    diccionario = {}
    def inicio():
        diccionario[str(playerid_entry.get())] = [str(playerid_entry.get()),float(playername_entry.get()),float(playerrank_entry.get()),float(playerrank1_entry.get()),float(playerrank2_entry.get()),float(playerrank3_entry.get()),float(playerrank4_entry.get())
                                                  ,float(playerrank5_entry.get()),float(playerrank6_entry.get()),float(playerrank7_entry.get()),float(playerrank8_entry.get()),float(playerrank9_entry.get()),float(playerrank10_entry.get())]
        selected=my_game.focus()
        #save new data 
        my_game.item(selected,text="",values=(playerid_entry.get(),playername_entry.get(),playerrank_entry.get(),playerrank1_entry.get(),playerrank2_entry.get(),playerrank3_entry.get(),playerrank4_entry.get(),playerrank5_entry.get()
                                              ,playerrank6_entry.get(),playerrank7_entry.get(),playerrank8_entry.get(),playerrank9_entry.get(),playerrank10_entry.get()))
    
    def update_record():
        selected=my_game.focus()
        global df11
        df11 = df11 * 0
        df11.drop(df11.index, inplace=True)
        df1 = pd.DataFrame(diccionario.values())
        df1.rename(columns={0: 'Class', 1: 'Impact Radius', 2: 'Probability', 3: 'Probit People', 4: 'Probit House', 5: 'Probit IAp', 6: 'Probit INps', 7: 'Probit LNps', 8: 'Probit INss', 9: 'Probit LNss', 10: 'Probit Ewater', 11: 'Probit Enlc', 12: 'Probit Eforest'}, inplace=True)
        df11 = df11.append(df1)
        df11 = df11[df11.iloc[:, 1] != 0]
        ws.quit()
        ws.destroy()
    #Buttons
    select_button = Button(ws,text="Select radiation level", command=select_record)
    select_button.pack(pady =10)

    edit_button = Button(ws,text="Update",command=inicio)
    edit_button.pack(pady = 10)
    #Storage table for process
    edit_button = Button(ws,text="Store and close",command=update_record)
    edit_button.pack(pady = 10)
    
    temp_label =Label(ws,text="")
    temp_label.pack()
    

    ws.mainloop()
    
###############################################################################################
#Table input pool fire RIP
columns = ['Class', 'Impact Radius', 'Probability', 'Probit People', 'Probit House', 'Probit IAp', 'Probit INps', 'Probit LNps', 'Probit INss', 'Probit LNss',  'Probit Ewater', 'Probit Enlc', 'Probit Eforest']

# Utilizando los mismos encabezados para df22 y los datos proporcionados
data_df22 = [
    (">37.5", "0", "1", "1", "1", "1", "1", "1", "1", "1", "0", "1", "1"),
    (">20.9", "0", "0.83", "1", "1", "1", "0.5", "1", "0.5", "1", "0", "1", "1"),
    (">14.5", "0", "0.39", "1", "1", "1", "0.25", "0", "0.25", "0", "0", "1", "1"),
    (">9.5", "0", "0.04", "0.86", "0", "0", "0", "0", "0", "0", "0", "0.86", "0.66"),
    (">7.27", "0", "0", "0.5", "0", "0", "0", "0", "0", "0", "0", "0.5", "0"),
    (">5.0", "0", "0", "0.07", "0", "0", "0", "0", "0", "0", "0", "0.07", "0"),
    (">1.6", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"),
]

# Creando el DataFrame df22
df22 = pd.DataFrame(data_df22, columns=columns)

def table22():
    from  tkinter import ttk
    ws  = Tk()
    ws.title('Potential Impact Radius Pool Fire')
    ws.geometry('1320x440')
    ws['bg'] = '#c2c4c3'
    game_frame = Frame(ws)
    game_frame.pack()
    
    
    #scrollbar
    game_scroll = Scrollbar(game_frame)
    game_scroll.pack(side=RIGHT, fill=Y)

    game_scroll = Scrollbar(game_frame,orient='horizontal')
    game_scroll.pack(side= BOTTOM,fill=X)

    my_game = ttk.Treeview(game_frame,yscrollcommand=game_scroll.set, xscrollcommand =game_scroll.set)


    my_game.pack()

    game_scroll.config(command=my_game.yview)
    game_scroll.config(command=my_game.xview)

    #define our column
     
    my_game['columns'] = ('Thermal Radiation [Kw/m2]', 'Impact Radius Length [m]', 'Probability of Death', 'Probit People','Probit House','Probit IAp', 'Probit INps', 'Probit LNps', 'Probit INss', 'Probit LNss', 'Probit Ewater', 'Probit Enlc', 'Probit Eforest')

    # format our column
    my_game.column("#0", width=0,  stretch=NO)
    my_game.column("Thermal Radiation [Kw/m2]",anchor=CENTER, width=100)
    my_game.column("Impact Radius Length [m]",anchor=CENTER,width=80)
    my_game.column("Probability of Death",anchor=CENTER,width=100)
    my_game.column("Probit People",anchor=CENTER,width=100)
    my_game.column("Probit House",anchor=CENTER,width=100)
    my_game.column("Probit IAp",anchor=CENTER,width=100)
    my_game.column("Probit INps",anchor=CENTER,width=100)
    my_game.column("Probit LNps",anchor=CENTER,width=100)
    my_game.column("Probit INss",anchor=CENTER,width=100)
    my_game.column("Probit LNss",anchor=CENTER,width=100)
    my_game.column("Probit Ewater",anchor=CENTER,width=100)
    my_game.column("Probit Enlc",anchor=CENTER,width=100)
    my_game.column("Probit Eforest",anchor=CENTER,width=100)
  
    #Create Headings 
    my_game.heading("#0",text="",anchor=CENTER)
    my_game.heading("Thermal Radiation [Kw/m2]",text="Thermal Radiation [Kw/m2]",anchor=CENTER)
    my_game.heading("Impact Radius Length [m]",text="Impact Radius Length [m]",anchor=CENTER)
    my_game.heading("Probability of Death",text="Probability of Death",anchor=CENTER)
    my_game.heading("Probit People",text="Probit People",anchor=CENTER)
    my_game.heading("Probit House",text="Probit House",anchor=CENTER)
    my_game.heading("Probit IAp",text="Probit IAp",anchor=CENTER)
    my_game.heading("Probit INps",text="Probit INps",anchor=CENTER)
    my_game.heading("Probit LNps",text="Probit LNps",anchor=CENTER)
    my_game.heading("Probit INss",text="Probit INss",anchor=CENTER)
    my_game.heading("Probit LNss",text="Probit LNss",anchor=CENTER)
    my_game.heading("Probit Ewater",text="Probit Ewater",anchor=CENTER)
    my_game.heading("Probit Enlc",text="Probit Enlc",anchor=CENTER)
    my_game.heading("Probit Eforest",text="Probit Eforest",anchor=CENTER)


    #add data tank
    data = [
    (">37.5", "0", "1", "1", "1", "1", "1", "1", "1", "1", "0", "1", "1"),
    (">20.9", "0", "0.83", "1", "1", "1", "0.5", "1", "0.5", "1", "0", "1", "1"),
    (">14.5", "0", "0.39", "1", "1", "1", "0.25", "0", "0.25", "0", "0", "1", "1"),
    (">9.5", "0", "0.04", "0.86", "0", "0", "0", "0", "0", "0", "0", "0.86", "0.66"),
    (">7.27", "0", "0", "0.5", "0", "0", "0", "0", "0", "0", "0", "0.5", "0"),
    (">5.0", "0", "0", "0.07", "0", "0", "0", "0", "0", "0", "0", "0.07", "0"),
    (">1.6", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"),
    ]
    
    for i, row in enumerate(data):
        my_game.insert("", "end", iid=i, values=row)
    

    my_game.pack()

    frame = Frame(ws)
    frame.pack(pady=12)

    #labels
    
    playerid= Label(frame,text = "Thermal Radiation [Kw/m2]")
    playerid.grid(row=0,column=0 )

    playername = Label(frame,text="Impact Radius Length [m]")
    playername.grid(row=0,column=1)

    playerrank = Label(frame,text="Probability of Death")
    playerrank.grid(row=0,column=2)
    
    playerrank1 = Label(frame,text="Probit People")
    playerrank1.grid(row=0,column=3)
    
    playerrank2 = Label(frame,text="Probit House")
    playerrank2.grid(row=0,column=4)
    
    playerrank3 = Label(frame,text="Probit IAp")
    playerrank3.grid(row=0,column=5)
    
    playerrank4 = Label(frame,text="Probit INps")
    playerrank4.grid(row=0,column=6)
    
    playerrank5 = Label(frame,text="Probit LNps")
    playerrank5.grid(row=0,column=7)
    
    playerrank6 = Label(frame,text="Probit INss")
    playerrank6.grid(row=0,column=8)
    
    playerrank7 = Label(frame,text="Probit LNss")
    playerrank7.grid(row=0,column=9)
    
    playerrank8 = Label(frame,text="Probit Ewater")
    playerrank8.grid(row=0,column=10)
    
    playerrank9 = Label(frame,text="Probit Enlc")
    playerrank9.grid(row=0,column=11)
    
    playerrank10 = Label(frame,text="Probit Eforest")
    playerrank10.grid(row=0,column=12)
    
       
    #Entry boxes
    playerid_entry = StringVar()
    playername_entry = StringVar()
    playerrank_entry = StringVar()
    playerrank1_entry = StringVar()
    playerrank2_entry = StringVar()
    playerrank3_entry = StringVar()
    playerrank4_entry = StringVar()
    playerrank5_entry = StringVar()
    playerrank6_entry = StringVar()
    playerrank7_entry = StringVar()
    playerrank8_entry = StringVar()
    playerrank9_entry = StringVar()
    playerrank10_entry = StringVar()
    
    # Definir el ancho de las cajas de entrada
    entry_width = 14  # Este es un ejemplo, puedes ajustar el tamaño como necesites
    
        
    playerid_entry = Entry(frame, textvariable=playerid_entry, width=entry_width)
    playerid_entry.grid(row=1, column=0)
    
    playername_entry = Entry(frame, textvariable=playername_entry, width=entry_width)
    playername_entry.grid(row=1, column=1)
    
    playerrank_entry = Entry(frame, textvariable=playerrank_entry, width=entry_width)
    playerrank_entry.grid(row=1, column=2)
    
    playerrank1_entry = Entry(frame, textvariable=playerrank1_entry, width=entry_width)
    playerrank1_entry.grid(row=1, column=3)
    
    playerrank2_entry = Entry(frame, textvariable=playerrank2_entry, width=entry_width)
    playerrank2_entry.grid(row=1, column=4)
    
    playerrank3_entry = Entry(frame, textvariable=playerrank3_entry, width=entry_width)
    playerrank3_entry.grid(row=1, column=5)
    
    playerrank4_entry = Entry(frame, textvariable=playerrank4_entry, width=entry_width)
    playerrank4_entry.grid(row=1, column=6)
    
    playerrank5_entry = Entry(frame, textvariable=playerrank5_entry, width=entry_width)
    playerrank5_entry.grid(row=1, column=7)
    
    playerrank6_entry = Entry(frame, textvariable=playerrank6_entry, width=entry_width)
    playerrank6_entry.grid(row=1, column=8)
    
    playerrank7_entry = Entry(frame, textvariable=playerrank7_entry, width=entry_width)
    playerrank7_entry.grid(row=1, column=9)
    
    playerrank8_entry = Entry(frame, textvariable=playerrank8_entry, width=entry_width)
    playerrank8_entry.grid(row=1, column=10)
    
    playerrank9_entry = Entry(frame, textvariable=playerrank9_entry, width=entry_width)
    playerrank9_entry.grid(row=1, column=11)
    
    playerrank10_entry = Entry(frame, textvariable=playerrank10_entry, width=entry_width)
    playerrank10_entry.grid(row=1, column=12)

    
    #Select Record
    def select_record():
        #clear entry boxes
        playerid_entry.delete(0,END)
        playername_entry.delete(0,END)
        playerrank_entry.delete(0,END)
        playerrank1_entry.delete(0,END)
        playerrank2_entry.delete(0,END)
        playerrank3_entry.delete(0,END)
        playerrank4_entry.delete(0,END)
        playerrank5_entry.delete(0,END)
        playerrank6_entry.delete(0,END)
        playerrank7_entry.delete(0,END)
        playerrank8_entry.delete(0,END)
        playerrank9_entry.delete(0,END)
        playerrank10_entry.delete(0,END)
                
        
        #grab record
        selected=my_game.focus()
        #grab record values
        values = my_game.item(selected,'values')
        #temp_label.config(text=selected)

        #output to entry boxes
        playerid_entry.insert(0,values[0])
        playername_entry.insert(0,values[1])
        playerrank_entry.insert(0,values[2])
        playerrank1_entry.insert(0,values[3])
        playerrank2_entry.insert(0,values[4])
        playerrank3_entry.insert(0,values[5])
        playerrank4_entry.insert(0,values[6])
        playerrank5_entry.insert(0,values[7])
        playerrank6_entry.insert(0,values[8])
        playerrank7_entry.insert(0,values[9])
        playerrank8_entry.insert(0,values[10])
        playerrank9_entry.insert(0,values[11])
        playerrank10_entry.insert(0,values[12])
        

    #save Record    
    def funcion_prueba ():
        playerid_entry = StringVar()
        playername_entry = StringVar()
        playerrank_entry = StringVar()
        playerrank1_entry = StringVar()
        playerrank2_entry = StringVar()
        playerrank3_entry = StringVar()
        playerrank4_entry = StringVar()
        playerrank5_entry = StringVar()
        playerrank6_entry = StringVar()
        playerrank7_entry = StringVar()
        playerrank8_entry = StringVar()
        playerrank9_entry = StringVar()
        playerrank10_entry = StringVar()
        
      
    diccionario = {}
    def inicio():
        diccionario[str(playerid_entry.get())] = [str(playerid_entry.get()),float(playername_entry.get()),float(playerrank_entry.get()),float(playerrank1_entry.get()),float(playerrank2_entry.get()),float(playerrank3_entry.get()),float(playerrank4_entry.get())
                                                  ,float(playerrank5_entry.get()),float(playerrank6_entry.get()),float(playerrank7_entry.get()),float(playerrank8_entry.get()),float(playerrank9_entry.get()),float(playerrank10_entry.get())]
        selected=my_game.focus()
        #save new data 
        my_game.item(selected,text="",values=(playerid_entry.get(),playername_entry.get(),playerrank_entry.get(),playerrank1_entry.get(),playerrank2_entry.get(),playerrank3_entry.get(),playerrank4_entry.get(),playerrank5_entry.get()
                                              ,playerrank6_entry.get(),playerrank7_entry.get(),playerrank8_entry.get(),playerrank9_entry.get(),playerrank10_entry.get()))
    
    def update_record():
        selected=my_game.focus()
        global df22
        df22 = df22 * 0
        df22.drop(df22.index, inplace=True)
        df1 = pd.DataFrame(diccionario.values())
        df1.rename(columns={0: 'Class', 1: 'Impact Radius', 2: 'Probability', 3: 'Probit People', 4: 'Probit House', 5: 'Probit IAp', 6: 'Probit INps', 7: 'Probit LNps', 8: 'Probit INss', 9: 'Probit LNss', 10: 'Probit Ewater', 11: 'Probit Enlc', 12: 'Probit Eforest'}, inplace=True)
        df22 = df22.append(df1)
        df22 = df22[df22.iloc[:, 1] != 0]
        ws.quit()
        ws.destroy()
    #Buttons
    select_button = Button(ws,text="Select radiation level", command=select_record)
    select_button.pack(pady =10)

    edit_button = Button(ws,text="Update",command=inicio)
    edit_button.pack(pady = 10)
    #Storage table for process
    edit_button = Button(ws,text="Store and close",command=update_record)
    edit_button.pack(pady = 10)
    
    temp_label =Label(ws,text="")
    temp_label.pack()
    

    ws.mainloop()
    
###############################################################################################
#Table input Over Pressure RIP
columns = ['Class', 'Impact Radius', 'Probability', 'Probit People', 'Probit House', 'Probit IAp', 'Probit INps', 'Probit LNps', 'Probit INss', 'Probit LNss',  'Probit Ewater', 'Probit Enlc', 'Probit Eforest']


data_df33 = [
    (">14", "0", "1", "1", "1", "1", "1", "1", "1", "1", "0", "1", "1"),
    (">6.4", "0", "1", "1", "0.9924", "0.9924", "0.9924", "0.9924", "0.9924", "0.9924", "0", "1", "0.9924"),
    (">4.2", "0", "1", "1", "0.8847", "0.8847", "0.8847", "0.8847", "0.8847", "0.8847", "0", "1", "0.8847"),
    (">3.25", "0", "1", "1", "0.6737", "0.6737", "0.6737", "0.6737", "0.6737", "0.6737", "0", "1", "0.6737"),
    (">3.0", "0", "1", "1", "0.5857", "0.5857", "0.5857", "0.5857", "0.5857", "0.5857", "0", "1", "0.5857"),
    (">2.0", "0", "1", "1", "0.1666", "0.1666", "0.1666", "0.1666", "0.1666", "0.1666", "0", "1", "0.1666"),
    (">0.4", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"),
]

# Creando el DataFrame df33
df33 = pd.DataFrame(data_df33, columns=columns)

def table33():
    from  tkinter import ttk
    ws  = Tk()
    ws.title('Potential Impact Radius Over Pressure')
    ws.geometry('1320x440')
    ws['bg'] = '#c2c4c3'
    game_frame = Frame(ws)
    game_frame.pack()
    
    
    #scrollbar
    game_scroll = Scrollbar(game_frame)
    game_scroll.pack(side=RIGHT, fill=Y)

    game_scroll = Scrollbar(game_frame,orient='horizontal')
    game_scroll.pack(side= BOTTOM,fill=X)

    my_game = ttk.Treeview(game_frame,yscrollcommand=game_scroll.set, xscrollcommand =game_scroll.set)


    my_game.pack()

    game_scroll.config(command=my_game.yview)
    game_scroll.config(command=my_game.xview)

    #define our column
     
    my_game['columns'] = ('Thermal Radiation [Kw/m2]', 'Impact Radius Length [m]', 'Probability of Death', 'Probit People','Probit House','Probit IAp', 'Probit INps', 'Probit LNps', 'Probit INss', 'Probit LNss', 'Probit Ewater', 'Probit Enlc', 'Probit Eforest')

    # format our column
    my_game.column("#0", width=0,  stretch=NO)
    my_game.column("Thermal Radiation [Kw/m2]",anchor=CENTER, width=100)
    my_game.column("Impact Radius Length [m]",anchor=CENTER,width=80)
    my_game.column("Probability of Death",anchor=CENTER,width=100)
    my_game.column("Probit People",anchor=CENTER,width=100)
    my_game.column("Probit House",anchor=CENTER,width=100)
    my_game.column("Probit IAp",anchor=CENTER,width=100)
    my_game.column("Probit INps",anchor=CENTER,width=100)
    my_game.column("Probit LNps",anchor=CENTER,width=100)
    my_game.column("Probit INss",anchor=CENTER,width=100)
    my_game.column("Probit LNss",anchor=CENTER,width=100)
    my_game.column("Probit Ewater",anchor=CENTER,width=100)
    my_game.column("Probit Enlc",anchor=CENTER,width=100)
    my_game.column("Probit Eforest",anchor=CENTER,width=100)
  
    #Create Headings 
    my_game.heading("#0",text="",anchor=CENTER)
    my_game.heading("Thermal Radiation [Kw/m2]",text="Thermal Radiation [Kw/m2]",anchor=CENTER)
    my_game.heading("Impact Radius Length [m]",text="Impact Radius Length [m]",anchor=CENTER)
    my_game.heading("Probability of Death",text="Probability of Death",anchor=CENTER)
    my_game.heading("Probit People",text="Probit People",anchor=CENTER)
    my_game.heading("Probit House",text="Probit House",anchor=CENTER)
    my_game.heading("Probit IAp",text="Probit IAp",anchor=CENTER)
    my_game.heading("Probit INps",text="Probit INps",anchor=CENTER)
    my_game.heading("Probit LNps",text="Probit LNps",anchor=CENTER)
    my_game.heading("Probit INss",text="Probit INss",anchor=CENTER)
    my_game.heading("Probit LNss",text="Probit LNss",anchor=CENTER)
    my_game.heading("Probit Ewater",text="Probit Ewater",anchor=CENTER)
    my_game.heading("Probit Enlc",text="Probit Enlc",anchor=CENTER)
    my_game.heading("Probit Eforest",text="Probit Eforest",anchor=CENTER)


    #add data tank
    data = [
    (">14", "0", "1", "1", "1", "1", "1", "1", "1", "1", "0", "1", "1"),
    (">6.4", "0", "1", "1", "0.9924", "0.9924", "0.9924", "0.9924", "0.9924", "0.9924", "0", "1", "0.9924"),
    (">4.2", "0", "1", "1", "0.8847", "0.8847", "0.8847", "0.8847", "0.8847", "0.8847", "0", "1", "0.8847"),
    (">3.25", "0", "1", "1", "0.6737", "0.6737", "0.6737", "0.6737", "0.6737", "0.6737", "0", "1", "0.6737"),
    (">3.0", "0", "1", "1", "0.5857", "0.5857", "0.5857", "0.5857", "0.5857", "0.5857", "0", "1", "0.5857"),
    (">2.0", "0", "1", "1", "0.1666", "0.1666", "0.1666", "0.1666", "0.1666", "0.1666", "0", "1", "0.1666"),
    (">0.4", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"),
    ]
    
    for i, row in enumerate(data):
        my_game.insert("", "end", iid=i, values=row)
    

    my_game.pack()

    frame = Frame(ws)
    frame.pack(pady=12)

    #labels
    
    playerid= Label(frame,text = "Thermal Radiation [Kw/m2]")
    playerid.grid(row=0,column=0 )

    playername = Label(frame,text="Impact Radius Length [m]")
    playername.grid(row=0,column=1)

    playerrank = Label(frame,text="Probability of Death")
    playerrank.grid(row=0,column=2)
    
    playerrank1 = Label(frame,text="Probit People")
    playerrank1.grid(row=0,column=3)
    
    playerrank2 = Label(frame,text="Probit House")
    playerrank2.grid(row=0,column=4)
    
    playerrank3 = Label(frame,text="Probit IAp")
    playerrank3.grid(row=0,column=5)
    
    playerrank4 = Label(frame,text="Probit INps")
    playerrank4.grid(row=0,column=6)
    
    playerrank5 = Label(frame,text="Probit LNps")
    playerrank5.grid(row=0,column=7)
    
    playerrank6 = Label(frame,text="Probit INss")
    playerrank6.grid(row=0,column=8)
    
    playerrank7 = Label(frame,text="Probit LNss")
    playerrank7.grid(row=0,column=9)
    
    playerrank8 = Label(frame,text="Probit Ewater")
    playerrank8.grid(row=0,column=10)
    
    playerrank9 = Label(frame,text="Probit Enlc")
    playerrank9.grid(row=0,column=11)
    
    playerrank10 = Label(frame,text="Probit Eforest")
    playerrank10.grid(row=0,column=12)
    
       
    #Entry boxes
    playerid_entry = StringVar()
    playername_entry = StringVar()
    playerrank_entry = StringVar()
    playerrank1_entry = StringVar()
    playerrank2_entry = StringVar()
    playerrank3_entry = StringVar()
    playerrank4_entry = StringVar()
    playerrank5_entry = StringVar()
    playerrank6_entry = StringVar()
    playerrank7_entry = StringVar()
    playerrank8_entry = StringVar()
    playerrank9_entry = StringVar()
    playerrank10_entry = StringVar()
    
    # Definir el ancho de las cajas de entrada
    entry_width = 14  # Este es un ejemplo, puedes ajustar el tamaño como necesites
    
        
    playerid_entry = Entry(frame, textvariable=playerid_entry, width=entry_width)
    playerid_entry.grid(row=1, column=0)
    
    playername_entry = Entry(frame, textvariable=playername_entry, width=entry_width)
    playername_entry.grid(row=1, column=1)
    
    playerrank_entry = Entry(frame, textvariable=playerrank_entry, width=entry_width)
    playerrank_entry.grid(row=1, column=2)
    
    playerrank1_entry = Entry(frame, textvariable=playerrank1_entry, width=entry_width)
    playerrank1_entry.grid(row=1, column=3)
    
    playerrank2_entry = Entry(frame, textvariable=playerrank2_entry, width=entry_width)
    playerrank2_entry.grid(row=1, column=4)
    
    playerrank3_entry = Entry(frame, textvariable=playerrank3_entry, width=entry_width)
    playerrank3_entry.grid(row=1, column=5)
    
    playerrank4_entry = Entry(frame, textvariable=playerrank4_entry, width=entry_width)
    playerrank4_entry.grid(row=1, column=6)
    
    playerrank5_entry = Entry(frame, textvariable=playerrank5_entry, width=entry_width)
    playerrank5_entry.grid(row=1, column=7)
    
    playerrank6_entry = Entry(frame, textvariable=playerrank6_entry, width=entry_width)
    playerrank6_entry.grid(row=1, column=8)
    
    playerrank7_entry = Entry(frame, textvariable=playerrank7_entry, width=entry_width)
    playerrank7_entry.grid(row=1, column=9)
    
    playerrank8_entry = Entry(frame, textvariable=playerrank8_entry, width=entry_width)
    playerrank8_entry.grid(row=1, column=10)
    
    playerrank9_entry = Entry(frame, textvariable=playerrank9_entry, width=entry_width)
    playerrank9_entry.grid(row=1, column=11)
    
    playerrank10_entry = Entry(frame, textvariable=playerrank10_entry, width=entry_width)
    playerrank10_entry.grid(row=1, column=12)

    
    #Select Record
    def select_record():
        #clear entry boxes
        playerid_entry.delete(0,END)
        playername_entry.delete(0,END)
        playerrank_entry.delete(0,END)
        playerrank1_entry.delete(0,END)
        playerrank2_entry.delete(0,END)
        playerrank3_entry.delete(0,END)
        playerrank4_entry.delete(0,END)
        playerrank5_entry.delete(0,END)
        playerrank6_entry.delete(0,END)
        playerrank7_entry.delete(0,END)
        playerrank8_entry.delete(0,END)
        playerrank9_entry.delete(0,END)
        playerrank10_entry.delete(0,END)
                
        
        #grab record
        selected=my_game.focus()
        #grab record values
        values = my_game.item(selected,'values')
        #temp_label.config(text=selected)

        #output to entry boxes
        playerid_entry.insert(0,values[0])
        playername_entry.insert(0,values[1])
        playerrank_entry.insert(0,values[2])
        playerrank1_entry.insert(0,values[3])
        playerrank2_entry.insert(0,values[4])
        playerrank3_entry.insert(0,values[5])
        playerrank4_entry.insert(0,values[6])
        playerrank5_entry.insert(0,values[7])
        playerrank6_entry.insert(0,values[8])
        playerrank7_entry.insert(0,values[9])
        playerrank8_entry.insert(0,values[10])
        playerrank9_entry.insert(0,values[11])
        playerrank10_entry.insert(0,values[12])
        

    #save Record    
    def funcion_prueba ():
        playerid_entry = StringVar()
        playername_entry = StringVar()
        playerrank_entry = StringVar()
        playerrank1_entry = StringVar()
        playerrank2_entry = StringVar()
        playerrank3_entry = StringVar()
        playerrank4_entry = StringVar()
        playerrank5_entry = StringVar()
        playerrank6_entry = StringVar()
        playerrank7_entry = StringVar()
        playerrank8_entry = StringVar()
        playerrank9_entry = StringVar()
        playerrank10_entry = StringVar()
        
      
    diccionario = {}
    def inicio():
        diccionario[str(playerid_entry.get())] = [str(playerid_entry.get()),float(playername_entry.get()),float(playerrank_entry.get()),float(playerrank1_entry.get()),float(playerrank2_entry.get()),float(playerrank3_entry.get()),float(playerrank4_entry.get())
                                                  ,float(playerrank5_entry.get()),float(playerrank6_entry.get()),float(playerrank7_entry.get()),float(playerrank8_entry.get()),float(playerrank9_entry.get()),float(playerrank10_entry.get())]
        selected=my_game.focus()
        #save new data 
        my_game.item(selected,text="",values=(playerid_entry.get(),playername_entry.get(),playerrank_entry.get(),playerrank1_entry.get(),playerrank2_entry.get(),playerrank3_entry.get(),playerrank4_entry.get(),playerrank5_entry.get()
                                              ,playerrank6_entry.get(),playerrank7_entry.get(),playerrank8_entry.get(),playerrank9_entry.get(),playerrank10_entry.get()))
    
    def update_record():
        selected=my_game.focus()
        global df33
        df33 = df33 * 0
        df33.drop(df33.index, inplace=True)
        df1 = pd.DataFrame(diccionario.values())
        df1.rename(columns={0: 'Class', 1: 'Impact Radius', 2: 'Probability', 3: 'Probit People', 4: 'Probit House', 5: 'Probit IAp', 6: 'Probit INps', 7: 'Probit LNps', 8: 'Probit INss', 9: 'Probit LNss', 10: 'Probit Ewater', 11: 'Probit Enlc', 12: 'Probit Eforest'}, inplace=True)
        df33 = df33.append(df1)
        df33 = df33[df33.iloc[:, 1] != 0]
        ws.quit()
        ws.destroy()
    #Buttons
    select_button = Button(ws,text="Select Over Pressure [PSI]", command=select_record)
    select_button.pack(pady =10)

    edit_button = Button(ws,text="Update",command=inicio)
    edit_button.pack(pady = 10)
    #Storage table for process
    edit_button = Button(ws,text="Store and close",command=update_record)
    edit_button.pack(pady = 10)
    
    temp_label =Label(ws,text="")
    temp_label.pack()
    

    ws.mainloop()
    
###############################################################################################

def upload_image():
    ventana.filename = filedialog.askopenfilename(initialdir="C:", title="Select the background image:", filetypes=(("Imagenes TIF", "*.tif"), ("Imagenes PNG", "*.png")))    
    global ruta120
    ruta120 = ventana.filename
    ruta2 = ruta120.removesuffix('.tif')
    ruta3 = ruta2 + ".png"
    
    try:
        # Abre el archivo TIF utilizando GDAL
        ds = gdal.Open(ruta120)
        if ds.RasterCount >= 3:
            data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
        else:
            data = ds.ReadAsArray()

        # Obtiene información de transformación para posicionar correctamente la imagen
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]

        # Crea la figura y el eje para la trama
        fig, ax = plt.subplots()

        # Plot the TIFF data on the axis with its original coordinate reference system
        ax.imshow(data, extent=extent, origin='upper')
        
        # Configura el lienzo en Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        ventana.mainloop()

    except Exception as e:
        print("Error:", e)
    
#####################################################################
############Funciones para carga#####################################
def calcular_centroides(ruta):
    centroids = []
    with fiona.open(ruta) as f:
        for feature in f:
            polygon = shape(feature['geometry'])
            centroid = polygon.centroid
            centroids.append((centroid.x, centroid.y))
    return centroids

def agregar_centroides_a_gdf(ruta):
    df_centroides = pd.DataFrame(calcular_centroides(ruta), columns=['Este', 'Norte'])
    gdf1 = gpd.read_file(ruta)
    gdf = pd.concat([gdf1, df_centroides], axis=1)
    gdf = gdf.reset_index(drop=True)
    gdf['IDTK'] = gdf.index + 1
    return gdf

def graficar_centroides(gdf, ax):
    for index, row in gdf.iterrows():
        ax.annotate(row['IDTK'], (row.Este, row.Norte), color='red', )

def crear_dataframe_area_radio(gdf):
    df = gdf.copy()
    df['Area[m]'] = df['geometry'].area
    df['Radius[m]'] = (df['Area[m]'] / np.pi)**0.5
    df = df.drop('geometry', axis=1)
    df = df[["IDTK", "Area[m]", "Radius[m]", "Este", "Norte"]]
    return df

def check_columns_and_geometry(gdf):
    columns = gdf.columns.str.lower()
    if 'id' not in columns or 'clase' not in columns:
        messagebox.showwarning("Advertencia", "La cobertura de tanques no reporta el Id y la Clase de elemento")
        return False
    if not all(gdf.geom_type.isin(['Polygon', 'MultiPolygon'])):
        messagebox.showwarning("Advertencia", "El shapefile debe ser de tipo polígono.")
        return False
    return True

def tanks():
    ventana.filename = filedialog.askopenfilename(
        initialdir='C:', title="select tank shapefile:",
        filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*"))
    )
    global ruta
    ruta = ventana.filename
    ruta2 = os.path.splitext(ruta)[0]
    ruta_shapefile_1 = ruta2 + "1.shp"
    
    gdf = gpd.read_file(ruta)
    if not check_columns_and_geometry(gdf):
        return
    gdf = agregar_centroides_a_gdf(ruta)
    if gdf is None:
        return
    gdf.to_file(ruta_shapefile_1)

    fig, ax = plt.subplots()
    try:
        ds = gdal.Open(ruta120)
        if ds.RasterCount >= 3:
            data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
            gt = ds.GetGeoTransform()
            extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
            ax.imshow(data, extent=extent, origin='upper')
    except Exception as e:
        print("Failed to load raster:", e)

    gdf.plot(ax=ax, facecolor='none', edgecolor='red')
    graficar_centroides(gdf, ax)

    # Configuración del canvas de Tkinter
    canvas = tk.Canvas(ventana)
    canvas.pack()
    graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
    graph_canvas.draw()
    graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
    mplcursors.cursor()
    toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
    toolbar.update()
    canvas.place(x=715, y=160, width=780, height=530)

    df_info = crear_dataframe_area_radio(gdf)
    Label(ventana, text="Information Tank", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
    frame = Frame(ventana)
    frame.pack(fill='both', expand=True)
    frame.place(x=20, y=170, width=650, height=560)
    pt = Table(frame, dataframe=df_info, showtoolbar=True, showstatusbar=True)
    pt.show()
    ventana.mainloop()

def check_columns_and_geometrybuil(gdf):
    columns = gdf.columns.str.lower()
    if 'people' not in columns or 'nucleado' not in columns:
        messagebox.showwarning("Advertencia", "El shapefile no reporta los campos People y Nucleado.")
        return False
    if not all(gdf.geom_type.isin(['Polygon', 'MultiPolygon'])):
        messagebox.showwarning("Advertencia", "El shapefile debe ser de tipo polígono.")
        return False
    return True

def agregar_centroides_a_gdf2(gdf):
    gdf['centroid_x'] = gdf.geometry.centroid.x
    gdf['centroid_y'] = gdf.geometry.centroid.y
    gdf['Id'] = gdf.index + 1  # Calculando el Id como el índice + 1
    return gdf

def graficar_centroides2(gdf, ax):
    # Grafica los centroides y anota el Id de cada polígono
    for idx, row in gdf.iterrows():
        ax.scatter(row['centroid_x'], row['centroid_y'], color='blue')
        ax.annotate(str(row['Id']), (row['centroid_x'], row['centroid_y']), color='red')

def buildings():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select buildings shapefile:",
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta10
    ruta10 = filename
    if filename:
        ruta2 = os.path.splitext(filename)[0]
        ruta_png = ruta2 + ".png"
        ruta_shapefile_1 = ruta2 + "1.shp"
        
        gdf = gpd.read_file(filename)
        if not check_columns_and_geometrybuil(gdf):
            return
        gdf = agregar_centroides_a_gdf2(gdf)
        gdf.to_file(ruta_shapefile_1)
        
        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')
        graficar_centroides2(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        ventana.mainloop()

def check_columns_and_geometrypublic(gdf):
    columns = gdf.columns.str.lower()
    if 'people' not in columns:
        messagebox.showwarning("Advertencia", "El shapefile no reporta los campos People")
        return False
    if not all(gdf.geom_type.isin(['Polygon', 'MultiPolygon'])):
        messagebox.showwarning("Advertencia", "El shapefile debe ser de tipo polígono.")
        return False
    return True

def public():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select public goods shapefile:",
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta60    
    ruta60 = filename
    if filename:
        ruta_base = os.path.splitext(filename)[0]
        gdf = gpd.read_file(filename)
        if not check_columns_and_geometrypublic(gdf):
            return
        gdf = agregar_centroides_a_gdf2(gdf)
        ruta_shapefile_modificado = ruta_base + "1.shp"
        gdf.to_file(ruta_shapefile_modificado)

        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')
        graficar_centroides2(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        ventana.mainloop()

def check_columns_and_geometryline(gdf):
    columns = gdf.columns.str.lower()
    if 'people' not in columns:
        messagebox.showwarning("Advertencia", "El shapefile no reporta el campo People.")
        return False
    if not all(gdf.geom_type.isin(['LineString', 'MultiLineString'])):
        messagebox.showwarning("Advertencia", "El shapefile debe ser de tipo línea.")
        return False
    return True

def graficar_centroidesline(gdf, ax):
    # Grafica los centroides y anota el valor de la columna 'people' de cada línea
    for idx, row in gdf.iterrows():
        centroid = row['geometry'].centroid
        ax.scatter(centroid.x, centroid.y, color='blue')
        ax.annotate(str(row['people']), (centroid.x, centroid.y), color='red')

def linearpublic():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select linear public goods shapefile:",
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta90    
    ruta90 = filename
    if filename:
        base = os.path.splitext(filename)[0]
        ruta_png = base + ".png"
        
        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf = gpd.read_file(filename)
        if not check_columns_and_geometryline(gdf):
            return
        gdf.plot(ax=ax, facecolor='none', edgecolor='red')

        graficar_centroidesline(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        ventana.mainloop()


def social():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select social shapefile:",
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta110
    ruta110 = filename
    if filename:
        base = os.path.splitext(filename)[0]
        ruta_shapefile_1 = base + "1.shp"

        gdf = gpd.read_file(filename)
        if not check_columns_and_geometrypublic(gdf):
            return
        gdf = agregar_centroides_a_gdf2(gdf)
        gdf.to_file(ruta_shapefile_1)
        
        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')
        graficar_centroides2(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        # Annotate each feature in the GeoDataFrame
        for index, row in gdf.iterrows():
            ax.annotate(row['Id'], (row['centroid_x'], row['centroid_y']), color='red')

        ventana.mainloop()

def linealsocial():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select linear social shapefile:",
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta100    
    ruta100 = filename
    if filename:
        base = os.path.splitext(filename)[0]
        ruta_shapefile_1 = base + "1.shp"

        gdf = gpd.read_file(filename)
        if not check_columns_and_geometryline(gdf):
            return
        gdf.to_file(ruta_shapefile_1)
        
        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')
        graficar_centroidesline(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        ventana.mainloop()

def check_columns_and_geometryprod(gdf):
    columns = gdf.columns.str.lower()
    if 'people' not in columns or 'level' not in columns:
        messagebox.showwarning("Advertencia", "El shapefile no reporta los campos People y Level.")
        return False
    if not all(gdf.geom_type.isin(['Polygon', 'MultiPolygon'])):
        messagebox.showwarning("Advertencia", "El shapefile debe ser de tipo polígono.")
        return False
    return True


def graficar_centroidesprod(gdf, ax):
    # Grafica los centroides y anota el valor de la columna 'level' de cada polígono
    for idx, row in gdf.iterrows():
        ax.scatter(row['centroid_x'], row['centroid_y'], color='blue')
        ax.annotate(str(row['level']), (row['centroid_x'], row['centroid_y']), color='red')


def productive():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select productive shapefile:",
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta80
    ruta80 = filename
    if filename:
        base = os.path.splitext(filename)[0]
        gdf = gpd.read_file(filename)
        if not check_columns_and_geometryprod(gdf):
            return
        gdf = agregar_centroides_a_gdf2(gdf)
        
        # Determinar las rutas basadas en el nombre del archivo
        ruta_png = f"{base}.png"
        ruta_shapefile_1 = f"{base}1.shp"
        ruta_points_shp = f"{base}points.shp"
        ruta_cruce_shp = f"{base}cruce.shp"
        ruta_100_shp = f"{base}100.shp"
        
        # Guardar el GeoDataFrame con los centroides agregados
        gdf.to_file(ruta_shapefile_1)
        
        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')
        graficar_centroidesprod(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        ventana.mainloop()

def check_columns_and_geometryenv(gdf):
    columns = gdf.columns.str.lower()
    if 'sensitive' not in columns:
        messagebox.showwarning("Advertencia", "El shapefile no reporta el campo Sensitive.")
        return False
    if not all(gdf.geom_type.isin(['Polygon', 'MultiPolygon'])):
        messagebox.showwarning("Advertencia", "El shapefile debe ser de tipo polígono.")
        return False
    return True


def graficar_centroidesenv(gdf, ax):
    # Grafica los centroides y anota el valor de la columna 'sensitive' de cada polígono
    for idx, row in gdf.iterrows():
        ax.scatter(row['centroid_x'], row['centroid_y'], color='blue')
        ax.annotate(str(row['sensitive']), (row['centroid_x'], row['centroid_y']), color='blue')

def enviromental():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select enviromental shapefile:",
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta50    
    ruta50 = filename
    if filename:
        base = os.path.splitext(filename)[0]
        gdf = gpd.read_file(filename)
        if not check_columns_and_geometryenv(gdf):
            return
        gdf = agregar_centroides_a_gdf2(gdf)
        
        # Rutas derivadas del nombre del archivo
        ruta_png = f"{base}.png"
        ruta_shapefile_1 = f"{base}1.shp"
        
        # Guardar el GeoDataFrame con los centroides agregados
        gdf.to_file(ruta_shapefile_1)
        
        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf.plot(ax=ax, facecolor='none', edgecolor='blue')
        graficar_centroidesenv(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)
        
        ventana.mainloop()

def graficar_centroideshouse(gdf, ax):
    # Grafica los centroides y anota el valor de la columna 'people' de cada polígono
    for idx, row in gdf.iterrows():
        ax.scatter(row['centroid_x'], row['centroid_y'], color='blue')
        ax.annotate(str(row['people']), (row['centroid_x'], row['centroid_y']), color='red')


def households():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select housing shapefile:", 
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta20    
    ruta20 = filename
    if filename:
        base = os.path.splitext(filename)[0]
        gdf = gpd.read_file(filename)
        if not check_columns_and_geometrybuil(gdf):
            return
        gdf = agregar_centroides_a_gdf2(gdf)
        
        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')
        graficar_centroideshouse(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        ventana.mainloop()

def check_columns_and_geometrypeople(gdf):
    columns = gdf.columns.str.lower()
    if 'people' not in columns:
        messagebox.showwarning("Advertencia", "El shapefile no reporta el campo People.")
        return False
    if not all(gdf.geom_type.isin(['Point', 'MultiPoint'])):
        messagebox.showwarning("Advertencia", "El shapefile debe ser de tipo polígono.")
        return False
    return True

def people():
    filename = filedialog.askopenfilename(initialdir=ruta, title="Select the people permanence shapefile:", 
                                          filetypes=(("Shapefiles", "*.shp"), ("all files", "*.*")))
    global ruta40    
    ruta40 = filename
    if filename:
        base = os.path.splitext(filename)[0]
        gdf = gpd.read_file(filename)
        if not check_columns_and_geometrypeople(gdf):
            return
        gdf = agregar_centroides_a_gdf2(gdf)
        
        fig, ax = plt.subplots()
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')
        graficar_centroideshouse(gdf, ax)

        # Configuración del canvas de Tkinter
        canvas = tk.Canvas(ventana)
        canvas.pack()
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        canvas.place(x=715, y=160, width=780, height=530)

        ventana.mainloop()


       
############################################################################################

for column in df00.columns:
    if column != 'Class':
        df00[column] = pd.to_numeric(df00[column], errors='coerce', downcast='float')


def graf_RIP():
    import geopandas as gpd
    ruta2 = ruta.removesuffix('.shp')
    ruta400 = ruta2 + "1.shp"
    ruta500 = ruta2 + "100.shp"
    ruta600 = ruta2 + "200.shp"
    
    from shapely.geometry import shape, Point        
    gdf0 = gpd.read_file(ruta400)
    gdf = gdf0[gdf0.IDTK != 255]
    gdf = gdf[gdf.IDTK != 0]
    gdf.to_file(ruta500)
    #Solo para graficar y visualizar
    df = df44
    df498 = df44.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    
               
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    
    centroids = [] #Empy
       
    with fiona.open(ruta500) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este1', 'Norte1'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    shapefile03['Id'] = shapefile03['IDTK']
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    df4 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    # Obtener solo los nombres de archivo sin la ruta
    #shp_filenames = [os.path.basename(f) for f in shp_files]
    #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    from osgeo import gdal, ogr


    gdf = gpd.GeoDataFrame()

    for shape in work[0]:
        gdf_temp = gpd.read_file(shape)
        gdf = gdf.append(gdf_temp, ignore_index=True)
    
    for shape in work1[0]:
        gdf_temp = gpd.read_file(shape)
        gdf = gdf.append(gdf_temp, ignore_index=True)
        
    gdf00 = gdf.to_crs(3116)#3116/3857
    
        
    polygons = gdf00.geometry
           
    polygons.to_file(ruta600)
    
    try:
        # Cargar la imagen TIFF usando GDAL
        ds = gdal.Open(ruta120)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]
        
        # Crear figura y ejes para la trama
        fig, ax = plt.subplots()

        # Mostrar los datos TIFF en el eje con su sistema de referencia de coordenadas original
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
        
        gdf = gpd.read_file(ruta600)
        gdf.plot(ax=ax, facecolor='none', edgecolor='red')

        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
                   
        ventana.mainloop()
    except Exception as e:
        gdf = gpd.read_file(ruta600)
        fig, ax = plt.subplots()
        gdf.plot(ax=ax, facecolor='none', edgecolor='magenta')
    
        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
        
        ventana.mainloop()

dfs = {'df00': df00, 'df11': df11, 'df22': df22, 'df33': df33}

for df_name, df in dfs.items():
    for column in df.columns:
        if column != 'Class':
            dfs[df_name][column] = pd.to_numeric(df[column], errors='coerce').astype('float64')

df00, df11, df22, df33 = dfs['df00'], dfs['df11'], dfs['df22'], dfs['df33']

##############################################################################
###############################################################################
def calcular_centroides_de_shapefile(ruta_shapefile):
    """Calcula los centroides de los polígonos en un shapefile.
    Parámetros:
    ruta_shapefile (str): La ruta al archivo shapefile del cual calcular los centroides.
    Retorna:
    list: Una lista de tuplas con las coordenadas (x, y) de los centroides.
    """
    centroides = []
    try:
        with fiona.open(ruta_shapefile) as f:
            for feature in f:
                polygon = shape(feature['geometry'])
                centroid = polygon.centroid
                centroides.append((centroid.x, centroid.y))
    except Exception as e:
        print(f"Error al abrir el shapefile {ruta_shapefile}: {e}")
    
    return centroides


def visualizar_interpolacion_y_tabla(rutas_shapefiles, ruta120, ventana):
    try:
        
        # Cargar la imagen TIFF usando GDAL
        ds = gdal.Open(ruta120)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]
        
        fig, ax = plt.subplots()
        show(data, ax=ax, transform=src.transform)
        
        gdfs = []
        for ruta in rutas_shapefiles:
            try:
                gdf = gpd.read_file(ruta)
                gdfs.append(gdf)
            except Exception as e:
                print(f"Error al leer {ruta}: {e}")
        
        if not gdfs:
            print("No se encontraron archivos válidos.")
            return
        
        points_gdf = pd.concat(gdfs, ignore_index=True)
        points_gdf = points_gdf.dropna(subset=['geometry'])

        if points_gdf.empty:
            print("No hay datos para visualizar.")
            return
        
        min_value = points_gdf['risk_pop'].dropna().min() * 0.001
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_pop'] = min_value
        
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
        xi, yi = np.mgrid[min(x):max(x):500j, min(y):max(y):500j]
        zi = griddata((x, y), z, (xi, yi), method='linear')
        
        levels = np.linspace(0, max(z), 15)
        contours = plt.contour(xi, yi, zi, levels=levels, linewidths=2.0, cmap='jet')
        plt.colorbar(contours, ax=ax)

        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)

        if not points_gdf.empty:
            table = points_gdf[points_gdf.risk_pop != 0]
            table = table[table.Este != 0]
            table = table.dropna(subset=['Nucleado'])
            table2 = table[['people', 'Nucleado', 'risk_pop', 'Este', 'Norte']]
            root5 = table2
            
            Label(text="Table Risk on the Population index             ", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
            pt.show()

    except Exception as e:
        pass



def procesar_centroides_y_points(ruta510, gdf_merge, zone):
    centroids = []  # Lista vacía para almacenar los centroides
    try:
        with fiona.open(ruta510) as f:
            for feature in f:
                polygon = shape(feature['geometry'])
                centroid = polygon.centroid
                centroids.append((centroid.x, centroid.y))
    except Exception as e:
        print(f"Error al procesar {ruta510}: {e}")
        return None  # Puedes decidir retornar algo específico en caso de error

    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
    points = df_concatenado.copy()
    points['geometry'] = points['geometry'].centroid

    if not points.empty:
        points2 = zone.copy()
    else:
        columns = ['geometry', 'people']
        data = []
        points2 = gpd.GeoDataFrame(data, columns=columns, geometry='geometry')
        points2['people'] = points2['people'].astype('float64')

    points2['risk_pop'] = points2['people']
    points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
    df_puntos1 = points2.explode("geometry").reset_index(drop=True)

    points3 = points.append(df_puntos1, ignore_index=True)
    points3['risk_pop'] = points3['risk_pop'].fillna(0)
    points5 = points3[points3.risk_pop != 0]
    points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
    min_value = points6['risk_pop'].min()
    points7 = points6[points6.risk_pop != min_value]
    points7 = points7.drop(['area'], axis=1)

    if not points7.empty:
        points7.to_file(ruta520)  # Asegúrate de que ruta520 esté definida y accesible

def verificar_y_preparar_rutas():
    try:
        # Intenta acceder a las variables globales y verifica su existencia
        global df44, ruta20, ruta10
        if 'df44' not in globals():
            raise NameError("La cobertura de tanques no ha sido cargada")
        if 'ruta20' not in globals():
            raise NameError("Debe cargar la cobertura de Viviendas")
        if 'ruta10' not in globals():
            raise NameError("Debe cargar la cobertura de Construcciones")
    except NameError as e:
        messagebox.showerror("Error", str(e))
        return False  # Devuelve False si falta alguna variable

def visualizar_interpolacion_y_tabla1(rutas_shapefiles, ruta120, ventana):
    try:
        
        # Cargar la imagen TIFF usando GDAL
        ds = gdal.Open(ruta120)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]
        
        fig, ax = plt.subplots()
        show(data, ax=ax, transform=src.transform)
        
        gdfs = []
        
        for ruta in rutas:
            if os.path.exists(ruta):
                # El archivo existe, intenta leerlo y añadirlo a la lista
                gdf = gpd.read_file(ruta)
                gdfs.append(gdf)
            else:
                pass
        
        # Verifica si se leyó al menos un GeoDataFrame
        if gdfs:
            # Unir los GeoDataFrames que se lograron leer
            points_gdf = pd.concat(gdfs, ignore_index=True)
        else:
            points_gdf = gpd.GeoDataFrame()
        
        min_value = points_gdf['risk_pop'].dropna().min() * 0.001
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_pop'] = min_value
        
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
        xi, yi = np.mgrid[min(x):max(x):500j, min(y):max(y):500j]
        zi = griddata((x, y), z, (xi, yi), method='linear')
        
        levels = np.linspace(0, max(z), 15)
        contours = plt.contour(xi, yi, zi, levels=levels, linewidths=2.0, cmap='jet')
        plt.colorbar(contours, ax=ax)

        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)

        if not points_gdf.empty:
            table = points_gdf[points_gdf.risk_pop != 0]
            table = table[table.Este != 0]
            table = table.dropna(subset=['Nucleado'])
            table2 = table[['people', 'Nucleado', 'risk_pop', 'Este', 'Norte']]
            root5 = table2
            
            Label(text="Table Risk on the Population index             ", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
            pt.show()

    except Exception as e:
        pass       


def visualizar_interpolacion_y_tabla_risk_individual(rutas_shapefiles, ruta120, ventana):
    try:
        from matplotlib.colors import ListedColormap
        import numpy as np
        import matplotlib.pyplot as plt
        
        gdfs = []
        for ruta in rutas_shapefiles:
            if os.path.exists(ruta):
                gdf = gpd.read_file(ruta)
                gdfs.append(gdf)

        if gdfs:
            points_gdf = pd.concat(gdfs, ignore_index=True)
        else:
            messagebox.showerror("Error", "No existe traslape con coberturas de personas")
            return

        points_gdf = points_gdf.dropna(subset=['geometry'])
        min_value = points_gdf['risk'].min() * 0.001
        points_gdf.loc[points_gdf['Este'] == 0, 'risk'] = min_value

         # Extracción de coordenadas y valores de riesgo
        x = points_gdf.geometry.x
        y = points_gdf.geometry.y
        z = points_gdf['risk']
        
        # # Ajustando valores mínimos para 'risk'
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
        
        # Creación de la figura y el eje para la trama
        fig, ax = plt.subplots()
        
        # Carga y muestra de datos TIFF como imagen de fondo
        try:
            ds = gdal.Open(ruta120)
            if ds.RasterCount >= 3:
                data = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=-1)
                gt = ds.GetGeoTransform()
                extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
                ax.imshow(data, extent=extent, origin='upper')
        except Exception as e:
            print("Failed to load raster:", e)
        
        # Asumiendo que x, y, z_filled, y ax ya están definidos correctamente
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)  # Ajusta según sea necesario
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.8)
        plt.colorbar(contourf, ax=ax, label='Risk level Individual Risk')

        canvas = tk.Canvas(ventana)
        canvas.pack()

        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        canvas.place(x=715, y=160, width=780, height=530)

        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()

        if not points_gdf.empty:
            table = points_gdf[points_gdf.risk != 0]
            table = table[table.Este != 0]
            table2 = table[['people', 'risk', 'Este', 'Norte']]
            root5 = table2

            Label(text="Table Individual Risk", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
            pt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {e}")



###############################################################################

def proc_people():
    # Comprobación para verificar si ruta40 está definida
    try:
        if not ruta40:  # Asumiendo que 'ruta40' debe haber sido definida previamente
            raise NameError("ruta40 not defined")
    except NameError:
        messagebox.showerror("Error", "La cobertura de población no ha sido cargada")
        return
    ruta2 = ruta.removesuffix('.shp')
    ruta400 = ruta2 + "1.shp"
    ruta405 = ruta2 + "11.shp"
    ruta500 = ruta2 + "100.shp"
    ruta510 = ruta2 + "poly.shp"
    ruta520 = ruta2 + "people.shp" #df00
    ruta521 = ruta2 + "people1.shp" #df11
    ruta522 = ruta2 + "people2.shp" #df22
    ruta523 = ruta2 + "people3.shp" #df33
    ruta530 = ruta2 + "krig.shp"
    ruta540 = ruta2 + "contours.png"
    ruta600 = ruta2 + "200.shp"
    
    import os
    
    # Lista de archivos para verificar y limpiar
    archivos_limpiar = [ruta520, ruta521, ruta522, ruta523]
    
    # Itera sobre cada archivo en la lista
    for archivo in archivos_limpiar:
        if os.path.exists(archivo):
            os.remove(archivo)
    
    def funcion_principal_df00():
        global df00
        df11a = df00
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
                                             
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            if df['Impact Radius'].sum() == 0:
                # Establece el valor de la primera fila de 'Impact Radius' a 1
                df.loc[0, 'Impact Radius'] = 1
            df498 = df.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
            for column in df498.columns:
                if column != 'Class':
                    df498[column] = pd.to_numeric(df498[column], errors='coerce') 
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #For Flash Fire los df31...
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.sjoin(g2, g1, op='within')
                
                inter.to_file(output_shp)
            
            # Define la ruta base
            #ruta2 = ruta40.removesuffix('.tif')
            ruta22 = ruta40.removesuffix('.shp')
            ruta4 = ruta22 + "1.shp"
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta40, ruta70)
            intersect_and_save(shp1, ruta40, ruta71)
            intersect_and_save(shp2, ruta40, ruta72)
            intersect_and_save(shp3, ruta40, ruta73)
            intersect_and_save(shp4, ruta40, ruta74)
            intersect_and_save(shp5, ruta40, ruta75)
            intersect_and_save(shp6, ruta40, ruta76)
            
            df501 = (df498['Probability']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probability'][0]/1
            pr1 = df503['Probability'][1]/1
            pr2 = df503['Probability'][2]/1
            pr3 = df503['Probability'][3]/1
            pr4 = df503['Probability'][4]/1
            pr5 = df503['Probability'][5]/1
            pr6 = df503['Probability'][6]/1
            
                
            v0 = pr * ( (float(frec.get()) ) + 0)
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
           
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk'] = gdf['people'] * pr1
            gdf1['risk'] = gdf1['people'] * pr2
            gdf2['risk'] = gdf2['people'] * pr3
            gdf3['risk'] = gdf3['people'] * pr4
            gdf4['risk'] = gdf4['people'] * pr5
            gdf5['risk'] = gdf5['people'] * pr6
            gdf6['risk'] = gdf6['people'] * pr
            
            gdf['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf1['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf2['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf3['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf4['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf5['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf6['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge.to_file(ruta510)
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            #df_puntos = points2.explode("geometry").reset_index(drop=True)
            points2 = points2[points2.geometry.notnull()]
            df_puntos = points2.explode("geometry").reset_index(drop=True)
                
            points3 = points.append(df_puntos, ignore_index=True)
            points4 = points3.fillna(0)
            points5 = points4[points4.risk != 0]
            min_value = points5['risk'].min()
            points6 = points5[points5.risk != min_value]
            #points6.to_file(ruta520)
            if not points6.empty:
                points6.to_file(ruta520)
            else:
                pass
        else:
            pass
    funcion_principal_df00()
    ############################
    def funcion_principal_df11():
        global df11
        df11a = df11
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            
            gdf.to_file(ruta500)
    
            df = df11a
            if df['Impact Radius'].sum() == 0:
                # Establece el valor de la primera fila de 'Impact Radius' a 1
                df.loc[0, 'Impact Radius'] = 1
            df498 = df.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            # #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
            for column in df498.columns:
                if column != 'Class':
                    df498[column] = pd.to_numeric(df498[column], errors='coerce')        
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #For Flash Fire los df31...
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.sjoin(g2, g1, op='within')
                
                inter.to_file(output_shp)
            
            # Define la ruta base
            #ruta2 = ruta40.removesuffix('.tif')
            ruta22 = ruta40.removesuffix('.shp')
            ruta4 = ruta22 + "1.shp"
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta40, ruta70)
            intersect_and_save(shp1, ruta40, ruta71)
            intersect_and_save(shp2, ruta40, ruta72)
            intersect_and_save(shp3, ruta40, ruta73)
            intersect_and_save(shp4, ruta40, ruta74)
            intersect_and_save(shp5, ruta40, ruta75)
            intersect_and_save(shp6, ruta40, ruta76)
            
            df501 = (df498['Probability']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probability'][0]/1
            pr1 = df503['Probability'][1]/1
            pr2 = df503['Probability'][2]/1
            pr3 = df503['Probability'][3]/1
            pr4 = df503['Probability'][4]/1
            pr5 = df503['Probability'][5]/1
            pr6 = df503['Probability'][6]/1
            
                
            v0 = pr * ( (float(frec.get()) ) + 0)
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
           
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk'] = gdf['people'] * pr1
            gdf1['risk'] = gdf1['people'] * pr2
            gdf2['risk'] = gdf2['people'] * pr3
            gdf3['risk'] = gdf3['people'] * pr4
            gdf4['risk'] = gdf4['people'] * pr5
            gdf5['risk'] = gdf5['people'] * pr6
            gdf6['risk'] = gdf6['people'] * pr
            
            gdf['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf1['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf2['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf3['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf4['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf5['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf6['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge.to_file(ruta510)
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
            filas= points2['geometry']
               
            #df_puntos = points2.explode("geometry").reset_index(drop=True)
            points2 = points2[points2.geometry.notnull()]
            df_puntos = points2.explode("geometry").reset_index(drop=True)
        
                
            points3 = points.append(df_puntos, ignore_index=True)
            points4 = points3.fillna(0)
            points5 = points4[points4.risk != 0]
            min_value = points5['risk'].min()
            points6 = points5[points5.risk != min_value]
            #points6.to_file(ruta521)
            if not points6.empty:
                points6.to_file(ruta521)
            else:
                pass
        else:
            pass
    funcion_principal_df11()
    #############################
    def funcion_principal_df22():
        global df22
        df11a = df22
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            if df['Impact Radius'].sum() == 0:
                # Establece el valor de la primera fila de 'Impact Radius' a 1
                df.loc[0, 'Impact Radius'] = 1
            df498 = df.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
            for column in df498.columns:
                if column != 'Class':
                    df498[column] = pd.to_numeric(df498[column], errors='coerce')          
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #For Flash Fire los df31...
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.sjoin(g2, g1, op='within')
                
                inter.to_file(output_shp)
            
            # Define la ruta base
            #ruta2 = ruta40.removesuffix('.tif')
            ruta22 = ruta40.removesuffix('.shp')
            ruta4 = ruta22 + "1.shp"
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta40, ruta70)
            intersect_and_save(shp1, ruta40, ruta71)
            intersect_and_save(shp2, ruta40, ruta72)
            intersect_and_save(shp3, ruta40, ruta73)
            intersect_and_save(shp4, ruta40, ruta74)
            intersect_and_save(shp5, ruta40, ruta75)
            intersect_and_save(shp6, ruta40, ruta76)
            
            df501 = (df498['Probability']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probability'][0]/1
            pr1 = df503['Probability'][1]/1
            pr2 = df503['Probability'][2]/1
            pr3 = df503['Probability'][3]/1
            pr4 = df503['Probability'][4]/1
            pr5 = df503['Probability'][5]/1
            pr6 = df503['Probability'][6]/1
            
                
            v0 = pr * ( (float(frec.get()) ) + 0)
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
           
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk'] = gdf['people'] * pr1
            gdf1['risk'] = gdf1['people'] * pr2
            gdf2['risk'] = gdf2['people'] * pr3
            gdf3['risk'] = gdf3['people'] * pr4
            gdf4['risk'] = gdf4['people'] * pr5
            gdf5['risk'] = gdf5['people'] * pr6
            gdf6['risk'] = gdf6['people'] * pr
            
            gdf['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf1['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf2['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf3['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf4['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf5['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf6['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge.to_file(ruta510)
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
            filas= points2['geometry']
               
            #df_puntos = points2.explode("geometry").reset_index(drop=True)
            points2 = points2[points2.geometry.notnull()]
            df_puntos = points2.explode("geometry").reset_index(drop=True)
                
            points3 = points.append(df_puntos, ignore_index=True)
            points4 = points3.fillna(0)
            points5 = points4[points4.risk != 0]
            min_value = points5['risk'].min()
            points6 = points5[points5.risk != min_value]
            #points6.to_file(ruta522)
            if not points6.empty:
                points6.to_file(ruta522)
            else:
                pass
        else:
            pass
    funcion_principal_df22()
    #############################
    def funcion_principal_df33():
        global df33
        df11a = df33
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            if df['Impact Radius'].sum() == 0:
                # Establece el valor de la primera fila de 'Impact Radius' a 1
                df.loc[0, 'Impact Radius'] = 1
            df498 = df.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
            for column in df498.columns:
                if column != 'Class':
                    df498[column] = pd.to_numeric(df498[column], errors='coerce') 
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #For Flash Fire los df31...
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.sjoin(g2, g1, op='within')
                # Calcula el área de cada polígono de la intersección
                #inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                #inter = inter[inter['people'] != 255]
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            #ruta2 = ruta40.removesuffix('.tif')
            ruta22 = ruta40.removesuffix('.shp')
            ruta4 = ruta22 + "1.shp"
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta40, ruta70)
            intersect_and_save(shp1, ruta40, ruta71)
            intersect_and_save(shp2, ruta40, ruta72)
            intersect_and_save(shp3, ruta40, ruta73)
            intersect_and_save(shp4, ruta40, ruta74)
            intersect_and_save(shp5, ruta40, ruta75)
            intersect_and_save(shp6, ruta40, ruta76)
            
            df501 = (df498['Probability']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probability'][0]/1
            pr1 = df503['Probability'][1]/1
            pr2 = df503['Probability'][2]/1
            pr3 = df503['Probability'][3]/1
            pr4 = df503['Probability'][4]/1
            pr5 = df503['Probability'][5]/1
            pr6 = df503['Probability'][6]/1
            
                
            v0 = pr * ( (float(frec.get()) ) + 0)
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
           
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk'] = gdf['people'] * pr1
            gdf1['risk'] = gdf1['people'] * pr2
            gdf2['risk'] = gdf2['people'] * pr3
            gdf3['risk'] = gdf3['people'] * pr4
            gdf4['risk'] = gdf4['people'] * pr5
            gdf5['risk'] = gdf5['people'] * pr6
            gdf6['risk'] = gdf6['people'] * pr
            
            gdf['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf1['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf2['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf3['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf4['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf5['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf6['risk'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge.to_file(ruta510)
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
            filas= points2['geometry']
               
            #df_puntos = points2.explode("geometry").reset_index(drop=True)
            points2 = points2[points2.geometry.notnull()]
            df_puntos = points2.explode("geometry").reset_index(drop=True)
                
            points3 = points.append(df_puntos, ignore_index=True)
            points4 = points3.fillna(0)
            points5 = points4[points4.risk != 0]
            min_value = points5['risk'].min()
            points6 = points5[points5.risk != min_value]
            #points6.to_file(ruta523)
            if not points6.empty:
                points6.to_file(ruta523)
            else:
                pass
        else:
            pass
    funcion_principal_df33()
    #############################
    # Definir las rutas a los archivos shapefile y TIFF
    rutas_shapefiles = [ruta520, ruta521, ruta522, ruta523]
    visualizar_interpolacion_y_tabla_risk_individual(rutas_shapefiles, ruta120, ventana)
       
     
############################################################################################
def proc_societal():
    # Comprobación para verificar si ruta40 está definida
    try:
        if not ruta40:  # Asumiendo que 'ruta40' debe haber sido definida previamente
            raise NameError("ruta40 not defined")
    except NameError:
        messagebox.showerror("Error", "La cobertura de población no ha sido cargada")
        return
    ruta2 = ruta.removesuffix('.shp')
    ruta400 = ruta2 + "1.shp"
    ruta405 = ruta2 + "11.shp"
    ruta500 = ruta2 + "100.shp"
    ruta510 = ruta2 + "poly.shp"
    ruta520 = ruta2 + "pointsoc.shp" #df00
    ruta521 = ruta2 + "pointsoc1.shp" #df11
    ruta522 = ruta2 + "pointsoc2.shp" #df22
    ruta523 = ruta2 + "pointsoc3.shp" #df33
    ruta530 = ruta2 + "krig.shp"
    ruta540 = ruta2 + "contours.png"
    ruta600 = ruta2 + "200.shp"
    # Lista de archivos para verificar y limpiar
    archivos_limpiar = [ruta520, ruta521, ruta522, ruta523]
    import os
    # Itera sobre cada archivo en la lista
    for archivo in archivos_limpiar:
        if os.path.exists(archivo):
            os.remove(archivo)
    
    from shapely.geometry import shape, Point        
    import geopandas as gpd
               
    gdf0 = gpd.read_file(ruta400)
    gdf = gdf0[gdf0.IDTK != 255]
    gdf = gdf[gdf.IDTK != 0]
    # gdf00 = gdf.to_crs(3116)#3116/3857
    # gdf000 = gdf00.to_crs(4326)#4326
    gdf.to_file(ruta500)
    df = df00
    if df['Impact Radius'].sum() == 0:
        # Establece el valor de la primera fila de 'Impact Radius' a 1
        df.loc[0, 'Impact Radius'] = 1
    df498 = df.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    #dian = 30
    # def sumar_dian(x):
    #     return x + (float(d.get())) if x > 0 else x
    # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
    for column in df498.columns:
        if column != 'Class':
            df498[column] = pd.to_numeric(df498[column], errors='coerce')         
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    from shapely.geometry import shape, Point
    centroids = [] #Empy
       
    with fiona.open(ruta500) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    # Obtener solo los nombres de archivo sin la ruta
    #shp_filenames = [os.path.basename(f) for f in shp_files]
    #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    def intersect_and_save(shp1, shp2, output_shp):
        # Realiza la intersección entre los shapefiles
        g1 = gpd.GeoDataFrame.from_file(shp1)
        g2 = gpd.GeoDataFrame.from_file(shp2)
        inter = gpd.sjoin(g2, g1, op='within')
        # Calcula el área de cada polígono de la intersección
        #inter['area'] = inter['geometry'].apply(lambda x: x.area)
        # Reorganiza las columnas del GeoDataFrame
        #inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
        #inter = inter[inter['people'] != 255]
        # Guarda el resultado en un archivo shapefile
        inter.to_file(output_shp)
    
    # Define la ruta base
    #ruta2 = ruta40.removesuffix('.tif')
    ruta22 = ruta40.removesuffix('.shp')
    ruta4 = ruta22 + "1.shp"
    # Define la lista de sufijos
    sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
    # Genera las rutas de los archivos shapefile
    rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
    ruta7 = rutas[0]  
    ruta70 = rutas[1]
    ruta71 = rutas[2]
    ruta72 = rutas[3]
    ruta73 = rutas[4]
    ruta74 = rutas[5]
    ruta75 = rutas[6]
    ruta76 = rutas[7]
       
    shp = work[0][0]
    shp1 = work[0][1]
    shp2 = work[0][2]
    shp3 = work[0][3]
    shp4 = work[0][4]
    shp5 = work[0][5]
    shp6 = work1[0][0]
    
    z0 = gpd.read_file(shp)
    z1 = gpd.read_file(shp1)
    z2 = gpd.read_file(shp2)
    z3 = gpd.read_file(shp3)
    z4 = gpd.read_file(shp4)
    z5 = gpd.read_file(shp5)
    z6 = gpd.read_file(shp6)
    zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
    
    intersect_and_save(shp, ruta40, ruta70)
    intersect_and_save(shp1, ruta40, ruta71)
    intersect_and_save(shp2, ruta40, ruta72)
    intersect_and_save(shp3, ruta40, ruta73)
    intersect_and_save(shp4, ruta40, ruta74)
    intersect_and_save(shp5, ruta40, ruta75)
    intersect_and_save(shp6, ruta40, ruta76)
    
    df501 = (df498['Probability']).div(1)
    df503 = pd.DataFrame(df501)
    
    #probabilities value
    pr = df503['Probability'][0]/1
    pr1 = df503['Probability'][1]/1
    pr2 = df503['Probability'][2]/1
    pr3 = df503['Probability'][3]/1
    pr4 = df503['Probability'][4]/1
    pr5 = df503['Probability'][5]/1
    pr6 = df503['Probability'][6]/1
    
    v0 = 0 + (float(frec.get()))
    v1 = pr1 * ( (float(frec.get()) ) + 0)
    v2 = pr2 * ( (float(frec.get()) ) + 0)
    v3 = pr3 * ( (float(frec.get()) ) + 0)
    v4 = pr4 * ( (float(frec.get()) ) + 0)
    v5 = pr5 * ( (float(frec.get()) ) + 0)
    v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
    
    v00 = []
    v00.append (v0)
    v00.append (v1)
    v00.append (v2)
    v00.append (v3)
    v00.append (v4)
    v00.append (v5)
    v00.append (v6)
    
    zone = zone.assign(people=v00)    
    
    gdf = gpd.read_file(ruta70)
    gdf1 = gpd.read_file(ruta71)
    gdf2 = gpd.read_file(ruta72)
    gdf3 = gpd.read_file(ruta73)
    gdf4 = gpd.read_file(ruta74)
    gdf5 = gpd.read_file(ruta75)
    gdf6 = gpd.read_file(ruta76)
    
    gdf['affected'] = gdf['people'] * pr1
    gdf1['affected'] = gdf1['people'] * pr2
    gdf2['affected'] = gdf2['people'] * pr3
    gdf3['affected'] = gdf3['people'] * pr4
    gdf4['affected'] = gdf4['people'] * pr5
    gdf5['affected'] = gdf5['people'] * pr6
    gdf6['affected'] = gdf6['people'] * pr
    
    gdf['affected'] *= ( (float(ffp.get()) ) + 0)
    gdf1['affected'] *= ( (float(ffp.get()) ) + 0)
    gdf2['affected'] *= ( (float(ffp.get()) ) + 0)
    gdf3['affected'] *= ( (float(ffp.get()) ) + 0)
    gdf4['affected'] *= ( (float(ffp.get()) ) + 0)
    gdf5['affected'] *= ( (float(ffp.get()) ) + 0)
    gdf6['affected'] *= ( (float(ffp.get()) ) + 0)
    
    import geopandas as gpd
    from shapely.geometry import Point
    gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
    gdf_merge = gdf_merge.reset_index(drop=True)
    gdf_merge.to_file(ruta510)
    centroids = [] #Empy
       
    with fiona.open(ruta510) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
    
    points = df_concatenado.copy()
    # change geometry 
    points['geometry'] = points['geometry'].centroid
    
    # df4 = df3
    # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
    from shapely.geometry import MultiPoint
    points2 = zone.copy()
    points2['affected'] = points2['people']
    points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
    filas= points2['geometry']
       
    #df_puntos = points2.explode("geometry").reset_index(drop=True)
    points2 = points2[points2.geometry.notnull()]
    df_puntos = points2.explode("geometry").reset_index(drop=True)
        
    points3 = points.append(df_puntos, ignore_index=True)
    points4 = points3.fillna(0)
    points5 = points4[points4.affected != 0]
    min_value = points5['affected'].min()
    points6 = points5[points5.affected != min_value]
    points6.to_file(ruta520)
    ##########################
    df = df11
    if df['Impact Radius'].sum() == 0:
        # Establece el valor de la primera fila de 'Impact Radius' a 1
        df.loc[0, 'Impact Radius'] = 1
    df498 = df.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    #dian = 30
    # def sumar_dian(x):
    #     return x + (float(d.get())) if x > 0 else x
    # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
    for column in df498.columns:
        if column != 'Class':
            df498[column] = pd.to_numeric(df498[column], errors='coerce')        
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    from shapely.geometry import shape, Point
    centroids = [] #Empy
       
    with fiona.open(ruta500) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    # Obtener solo los nombres de archivo sin la ruta
    #shp_filenames = [os.path.basename(f) for f in shp_files]
    #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    def intersect_and_save(shp1, shp2, output_shp):
        # Realiza la intersección entre los shapefiles
        g1 = gpd.GeoDataFrame.from_file(shp1)
        g2 = gpd.GeoDataFrame.from_file(shp2)
        inter = gpd.sjoin(g2, g1, op='within')
        # Calcula el área de cada polígono de la intersección
        #inter['area'] = inter['geometry'].apply(lambda x: x.area)
        # Reorganiza las columnas del GeoDataFrame
        #inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
        #inter = inter[inter['people'] != 255]
        # Guarda el resultado en un archivo shapefile
        inter.to_file(output_shp)
    
    # Define la ruta base
    #ruta2 = ruta40.removesuffix('.tif')
    ruta22 = ruta40.removesuffix('.shp')
    ruta4 = ruta22 + "1.shp"
    # Define la lista de sufijos
    sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
    # Genera las rutas de los archivos shapefile
    rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
    ruta7 = rutas[0]  
    ruta70 = rutas[1]
    ruta71 = rutas[2]
    ruta72 = rutas[3]
    ruta73 = rutas[4]
    ruta74 = rutas[5]
    ruta75 = rutas[6]
    ruta76 = rutas[7]
       
    shp = work[0][0]
    shp1 = work[0][1]
    shp2 = work[0][2]
    shp3 = work[0][3]
    shp4 = work[0][4]
    shp5 = work[0][5]
    shp6 = work1[0][0]
    
    z0 = gpd.read_file(shp)
    z1 = gpd.read_file(shp1)
    z2 = gpd.read_file(shp2)
    z3 = gpd.read_file(shp3)
    z4 = gpd.read_file(shp4)
    z5 = gpd.read_file(shp5)
    z6 = gpd.read_file(shp6)
    zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
    
    intersect_and_save(shp, ruta40, ruta70)
    intersect_and_save(shp1, ruta40, ruta71)
    intersect_and_save(shp2, ruta40, ruta72)
    intersect_and_save(shp3, ruta40, ruta73)
    intersect_and_save(shp4, ruta40, ruta74)
    intersect_and_save(shp5, ruta40, ruta75)
    intersect_and_save(shp6, ruta40, ruta76)
    
    df501 = (df498['Probability']).div(1)
    df503 = pd.DataFrame(df501)
    
    #probabilities value
    pr = df503['Probability'][0]/1
    pr1 = df503['Probability'][1]/1
    pr2 = df503['Probability'][2]/1
    pr3 = df503['Probability'][3]/1
    pr4 = df503['Probability'][4]/1
    pr5 = df503['Probability'][5]/1
    pr6 = df503['Probability'][6]/1
    
    v0 = 0 + (float(frec.get()))
    v1 = pr1 * ( (float(frec.get()) ) + 0)
    v2 = pr2 * ( (float(frec.get()) ) + 0)
    v3 = pr3 * ( (float(frec.get()) ) + 0)
    v4 = pr4 * ( (float(frec.get()) ) + 0)
    v5 = pr5 * ( (float(frec.get()) ) + 0)
    v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
    
    v00 = []
    v00.append (v0)
    v00.append (v1)
    v00.append (v2)
    v00.append (v3)
    v00.append (v4)
    v00.append (v5)
    v00.append (v6)
    
    zone = zone.assign(people=v00)    
    
    gdf = gpd.read_file(ruta70)
    gdf1 = gpd.read_file(ruta71)
    gdf2 = gpd.read_file(ruta72)
    gdf3 = gpd.read_file(ruta73)
    gdf4 = gpd.read_file(ruta74)
    gdf5 = gpd.read_file(ruta75)
    gdf6 = gpd.read_file(ruta76)
    
    gdf['affected'] = gdf['people'] * pr1
    gdf1['affected'] = gdf1['people'] * pr2
    gdf2['affected'] = gdf2['people'] * pr3
    gdf3['affected'] = gdf3['people'] * pr4
    gdf4['affected'] = gdf4['people'] * pr5
    gdf5['affected'] = gdf5['people'] * pr6
    gdf6['affected'] = gdf6['people'] * pr
    
    gdf['affected'] *= ( (float(jfp.get()) ) + 0)
    gdf1['affected'] *= ( (float(jfp.get()) ) + 0)
    gdf2['affected'] *= ( (float(jfp.get()) ) + 0)
    gdf3['affected'] *= ( (float(jfp.get()) ) + 0)
    gdf4['affected'] *= ( (float(jfp.get()) ) + 0)
    gdf5['affected'] *= ( (float(jfp.get()) ) + 0)
    gdf6['affected'] *= ( (float(jfp.get()) ) + 0)
    
    import geopandas as gpd
    from shapely.geometry import Point
    gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
    gdf_merge = gdf_merge.reset_index(drop=True)
    gdf_merge.to_file(ruta510)
    centroids = [] #Empy
       
    with fiona.open(ruta510) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
    
    points = df_concatenado.copy()
    # change geometry 
    points['geometry'] = points['geometry'].centroid
    
    # df4 = df3
    # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
    from shapely.geometry import MultiPoint
    points2 = zone.copy()
    points2['affected'] = points2['people']
    points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
    filas= points2['geometry']
       
    #df_puntos = points2.explode("geometry").reset_index(drop=True)
    points2 = points2[points2.geometry.notnull()]
    df_puntos = points2.explode("geometry").reset_index(drop=True)
        
    points3 = points.append(df_puntos, ignore_index=True)
    points4 = points3.fillna(0)
    points5 = points4[points4.affected != 0]
    min_value = points5['affected'].min()
    points6 = points5[points5.affected != min_value]
    points6.to_file(ruta521)
    ###########################
    df = df22
    if df['Impact Radius'].sum() == 0:
        # Establece el valor de la primera fila de 'Impact Radius' a 1
        df.loc[0, 'Impact Radius'] = 1
    df498 = df.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    #dian = 30
    # def sumar_dian(x):
    #     return x + (float(d.get())) if x > 0 else x
    # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
    for column in df498.columns:
        if column != 'Class':
            df498[column] = pd.to_numeric(df498[column], errors='coerce')        
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    from shapely.geometry import shape, Point
    centroids = [] #Empy
       
    with fiona.open(ruta500) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    # Obtener solo los nombres de archivo sin la ruta
    #shp_filenames = [os.path.basename(f) for f in shp_files]
    #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    def intersect_and_save(shp1, shp2, output_shp):
        # Realiza la intersección entre los shapefiles
        g1 = gpd.GeoDataFrame.from_file(shp1)
        g2 = gpd.GeoDataFrame.from_file(shp2)
        inter = gpd.sjoin(g2, g1, op='within')
        inter.to_file(output_shp)
    
    # Define la ruta base
    #ruta2 = ruta40.removesuffix('.tif')
    ruta22 = ruta40.removesuffix('.shp')
    ruta4 = ruta22 + "1.shp"
    # Define la lista de sufijos
    sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
    # Genera las rutas de los archivos shapefile
    rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
    ruta7 = rutas[0]  
    ruta70 = rutas[1]
    ruta71 = rutas[2]
    ruta72 = rutas[3]
    ruta73 = rutas[4]
    ruta74 = rutas[5]
    ruta75 = rutas[6]
    ruta76 = rutas[7]
       
    shp = work[0][0]
    shp1 = work[0][1]
    shp2 = work[0][2]
    shp3 = work[0][3]
    shp4 = work[0][4]
    shp5 = work[0][5]
    shp6 = work1[0][0]
    
    z0 = gpd.read_file(shp)
    z1 = gpd.read_file(shp1)
    z2 = gpd.read_file(shp2)
    z3 = gpd.read_file(shp3)
    z4 = gpd.read_file(shp4)
    z5 = gpd.read_file(shp5)
    z6 = gpd.read_file(shp6)
    zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
    
    intersect_and_save(shp, ruta40, ruta70)
    intersect_and_save(shp1, ruta40, ruta71)
    intersect_and_save(shp2, ruta40, ruta72)
    intersect_and_save(shp3, ruta40, ruta73)
    intersect_and_save(shp4, ruta40, ruta74)
    intersect_and_save(shp5, ruta40, ruta75)
    intersect_and_save(shp6, ruta40, ruta76)
    
    df501 = (df498['Probability']).div(1)
    df503 = pd.DataFrame(df501)
    
    #probabilities value
    pr = df503['Probability'][0]/1
    pr1 = df503['Probability'][1]/1
    pr2 = df503['Probability'][2]/1
    pr3 = df503['Probability'][3]/1
    pr4 = df503['Probability'][4]/1
    pr5 = df503['Probability'][5]/1
    pr6 = df503['Probability'][6]/1
    
    v0 = 0 + (float(frec.get()))
    v1 = pr1 * ( (float(frec.get()) ) + 0)
    v2 = pr2 * ( (float(frec.get()) ) + 0)
    v3 = pr3 * ( (float(frec.get()) ) + 0)
    v4 = pr4 * ( (float(frec.get()) ) + 0)
    v5 = pr5 * ( (float(frec.get()) ) + 0)
    v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
    
    v00 = []
    v00.append (v0)
    v00.append (v1)
    v00.append (v2)
    v00.append (v3)
    v00.append (v4)
    v00.append (v5)
    v00.append (v6)
    
    zone = zone.assign(people=v00)    
    
    gdf = gpd.read_file(ruta70)
    gdf1 = gpd.read_file(ruta71)
    gdf2 = gpd.read_file(ruta72)
    gdf3 = gpd.read_file(ruta73)
    gdf4 = gpd.read_file(ruta74)
    gdf5 = gpd.read_file(ruta75)
    gdf6 = gpd.read_file(ruta76)
    
    gdf['affected'] = gdf['people'] * pr1
    gdf1['affected'] = gdf1['people'] * pr2
    gdf2['affected'] = gdf2['people'] * pr3
    gdf3['affected'] = gdf3['people'] * pr4
    gdf4['affected'] = gdf4['people'] * pr5
    gdf5['affected'] = gdf5['people'] * pr6
    gdf6['affected'] = gdf6['people'] * pr
    
    gdf['affected'] *= ( (float(pfp.get()) ) + 0)
    gdf1['affected'] *= ( (float(pfp.get()) ) + 0)
    gdf2['affected'] *= ( (float(pfp.get()) ) + 0)
    gdf3['affected'] *= ( (float(pfp.get()) ) + 0)
    gdf4['affected'] *= ( (float(pfp.get()) ) + 0)
    gdf5['affected'] *= ( (float(pfp.get()) ) + 0)
    gdf6['affected'] *= ( (float(pfp.get()) ) + 0)
    
    import geopandas as gpd
    from shapely.geometry import Point
    gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
    gdf_merge = gdf_merge.reset_index(drop=True)
    gdf_merge.to_file(ruta510)
    centroids = [] #Empy
       
    with fiona.open(ruta510) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
    
    points = df_concatenado.copy()
    # change geometry 
    points['geometry'] = points['geometry'].centroid
    
    # df4 = df3
    # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
    from shapely.geometry import MultiPoint
    points2 = zone.copy()
    points2['affected'] = points2['people']
    points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
    filas= points2['geometry']
       
    #df_puntos = points2.explode("geometry").reset_index(drop=True)
    points2 = points2[points2.geometry.notnull()]
    df_puntos = points2.explode("geometry").reset_index(drop=True)
        
    points3 = points.append(df_puntos, ignore_index=True)
    points4 = points3.fillna(0)
    points5 = points4[points4.affected != 0]
    min_value = points5['affected'].min()
    points6 = points5[points5.affected != min_value]
    points6.to_file(ruta522)
    ###########################
    df = df33
    if df['Impact Radius'].sum() == 0:
        # Establece el valor de la primera fila de 'Impact Radius' a 1
        df.loc[0, 'Impact Radius'] = 1
    df498 = df.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    
    for column in df498.columns:
        if column != 'Class':
            df498[column] = pd.to_numeric(df498[column], errors='coerce')        
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    from shapely.geometry import shape, Point
    centroids = [] #Empy
       
    with fiona.open(ruta500) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    # Obtener solo los nombres de archivo sin la ruta
    #shp_filenames = [os.path.basename(f) for f in shp_files]
    #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    def intersect_and_save(shp1, shp2, output_shp):
        # Realiza la intersección entre los shapefiles
        g1 = gpd.GeoDataFrame.from_file(shp1)
        g2 = gpd.GeoDataFrame.from_file(shp2)
        inter = gpd.sjoin(g2, g1, op='within')
        
        inter.to_file(output_shp)
    
    # Define la ruta base
    #ruta2 = ruta40.removesuffix('.tif')
    ruta22 = ruta40.removesuffix('.shp')
    ruta4 = ruta22 + "1.shp"
    # Define la lista de sufijos
    sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
    # Genera las rutas de los archivos shapefile
    rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
    ruta7 = rutas[0]  
    ruta70 = rutas[1]
    ruta71 = rutas[2]
    ruta72 = rutas[3]
    ruta73 = rutas[4]
    ruta74 = rutas[5]
    ruta75 = rutas[6]
    ruta76 = rutas[7]
       
    shp = work[0][0]
    shp1 = work[0][1]
    shp2 = work[0][2]
    shp3 = work[0][3]
    shp4 = work[0][4]
    shp5 = work[0][5]
    shp6 = work1[0][0]
    
    z0 = gpd.read_file(shp)
    z1 = gpd.read_file(shp1)
    z2 = gpd.read_file(shp2)
    z3 = gpd.read_file(shp3)
    z4 = gpd.read_file(shp4)
    z5 = gpd.read_file(shp5)
    z6 = gpd.read_file(shp6)
    zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
    
    intersect_and_save(shp, ruta40, ruta70)
    intersect_and_save(shp1, ruta40, ruta71)
    intersect_and_save(shp2, ruta40, ruta72)
    intersect_and_save(shp3, ruta40, ruta73)
    intersect_and_save(shp4, ruta40, ruta74)
    intersect_and_save(shp5, ruta40, ruta75)
    intersect_and_save(shp6, ruta40, ruta76)
    
    df501 = (df498['Probability']).div(1)
    df503 = pd.DataFrame(df501)
    
    #probabilities value
    pr = df503['Probability'][0]/1
    pr1 = df503['Probability'][1]/1
    pr2 = df503['Probability'][2]/1
    pr3 = df503['Probability'][3]/1
    pr4 = df503['Probability'][4]/1
    pr5 = df503['Probability'][5]/1
    pr6 = df503['Probability'][6]/1
    
    v0 = 0 + (float(frec.get()))
    v1 = pr1 * ( (float(frec.get()) ) + 0)
    v2 = pr2 * ( (float(frec.get()) ) + 0)
    v3 = pr3 * ( (float(frec.get()) ) + 0)
    v4 = pr4 * ( (float(frec.get()) ) + 0)
    v5 = pr5 * ( (float(frec.get()) ) + 0)
    v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
    
    v00 = []
    v00.append (v0)
    v00.append (v1)
    v00.append (v2)
    v00.append (v3)
    v00.append (v4)
    v00.append (v5)
    v00.append (v6)
    
    zone = zone.assign(people=v00)    
    
    gdf = gpd.read_file(ruta70)
    gdf1 = gpd.read_file(ruta71)
    gdf2 = gpd.read_file(ruta72)
    gdf3 = gpd.read_file(ruta73)
    gdf4 = gpd.read_file(ruta74)
    gdf5 = gpd.read_file(ruta75)
    gdf6 = gpd.read_file(ruta76)
    
    gdf['affected'] = gdf['people'] * pr1
    gdf1['affected'] = gdf1['people'] * pr2
    gdf2['affected'] = gdf2['people'] * pr3
    gdf3['affected'] = gdf3['people'] * pr4
    gdf4['affected'] = gdf4['people'] * pr5
    gdf5['affected'] = gdf5['people'] * pr6
    gdf6['affected'] = gdf6['people'] * pr
    
    gdf['affected'] *= ( (float(opp.get()) ) + 0)
    gdf1['affected'] *= ( (float(opp.get()) ) + 0)
    gdf2['affected'] *= ( (float(opp.get()) ) + 0)
    gdf3['affected'] *= ( (float(opp.get()) ) + 0)
    gdf4['affected'] *= ( (float(opp.get()) ) + 0)
    gdf5['affected'] *= ( (float(opp.get()) ) + 0)
    gdf6['affected'] *= ( (float(opp.get()) ) + 0)
    
    import geopandas as gpd
    from shapely.geometry import Point
    gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
    gdf_merge = gdf_merge.reset_index(drop=True)
    gdf_merge.to_file(ruta510)
    centroids = [] #Empy
       
    with fiona.open(ruta510) as f:
        # Iterates over all the entries in the shapefile
        for feature in f:
            # Gets the geometry of the polygon
            polygon = shape(feature['geometry'])
            # Calculate the centroid of the polygon
            centroid = polygon.centroid
            # Stores the coordinates of the centroid in the list
            centroids.append((centroid.x, centroid.y))

    # DataFrame centroids
    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
    
    points = df_concatenado.copy()
    # change geometry 
    points['geometry'] = points['geometry'].centroid
    
    # df4 = df3
    # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
    
    points2 = zone.copy()
    points2['affected'] = points2['people']
    points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
    filas= points2['geometry']
       
    #df_puntos = points2.explode("geometry").reset_index(drop=True)
    points2 = points2[points2.geometry.notnull()]
    df_puntos = points2.explode("geometry").reset_index(drop=True)
        
    points3 = points.append(df_puntos, ignore_index=True)
    points4 = points3.fillna(0)
    points5 = points4[points4.affected != 0]
    min_value = points5['affected'].min()
    points6 = points5[points5.affected != min_value]
    points6.to_file(ruta523)
    ##################################
            
    ############################    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        # Cargando y concatenando tus GeoDataFrames
        gdf1 = gpd.read_file(ruta520)
        gdf2 = gpd.read_file(ruta521)
        gdf3 = gpd.read_file(ruta522)
        gdf4 = gpd.read_file(ruta523)
        points_gdf = pd.concat([gdf1, gdf2, gdf3, gdf4], ignore_index=True)
        points_gdf = points_gdf.dropna(subset=['geometry'])
        # Suponiendo que points_gdf es tu DataFrame
        min_value = (points_gdf['affected'].min())*0.001
        points_gdf.loc[points_gdf['Este'] == 0, 'risk'] = min_value

        
        # Extracción de coordenadas y valores de riesgo
        x = points_gdf.geometry.x
        y = points_gdf.geometry.y
        z = points_gdf['affected']
        
        # # Ajustando valores mínimos para 'risk'
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
        
        # Creación de la figura y el eje para la trama
        fig, ax = plt.subplots()
        
        
        ds = gdal.Open(ruta120)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
        
        
        from matplotlib.colors import ListedColormap
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Asumiendo que x, y, z_filled, y ax ya están definidos correctamente
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)  # Ajusta según sea necesario
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Societal Risk')
        
        plt.show()
        import tkinter as tk
               
        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
                   
                
        table = points_gdf
        table1 = table[table.affected != 0]
        table1 = table1[table1.Este != 0]
        table2 = table1[['people', 'affected', 'Este', 'Norte']]
        root5 = table2
        
        Label(text = "Table Number of people affected                ", fg = 'black', font= ("Times New Roman",10)).place(x=24, y=144)
        frame = tk.Frame(ventana)
        frame.pack(fill='both', expand=True)
        frame.place(x=20, y=170, width=650, height=560)
        pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
        pt.show()
        ventana.mainloop()            
            
    except Exception as e:
        messagebox.showerror("No existe afectación Social", str(e))
                
        
###############################################################################
def visualizar_interpolacion_y_tablapop(rutas_shapefiles, ruta120, ventana, ruta5233):
    try:
        from matplotlib.colors import ListedColormap
        import numpy as np
        import matplotlib.pyplot as plt
        from osgeo import gdal
        
        # Cargar la imagen TIFF usando GDAL
        ds = gdal.Open(ruta120)
        if ds is None:
            messagebox.showerror("Error", f"No se pudo abrir el archivo TIFF {ruta120}")
            return
        
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]

        fig, ax = plt.subplots()
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
                
        gdfs = []
        for ruta in rutas_shapefiles:
            try:
                gdf = gpd.read_file(ruta)
                gdfs.append(gdf)
            except Exception as e:
                print(f"Error al leer {ruta}: {e}")
        
        if not gdfs:
            print("No se encontraron archivos válidos.")
            return
        
        points_gdf = pd.concat(gdfs, ignore_index=True)
        points_gdf = points_gdf.dropna(subset=['geometry'])

        if points_gdf.empty:
            print("No hay datos para visualizar.")
            return
        
        min_value = points_gdf['risk_pop'].dropna().min() * 0.001
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_pop'] = min_value
        
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
        # # Ajustando valores mínimos para 'risk'
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
        
        # Creación de la figura y el eje para la trama
        fig, ax = plt.subplots()
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
        
        # Asumiendo que x, y, z_filled, y ax ya están definidos correctamente
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)  # Ajusta según sea necesario
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Population index')

        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)

        if not points_gdf.empty:
            points_gdf.to_file(ruta5233, driver='ESRI Shapefile')
            table = points_gdf[points_gdf.risk_pop != 0]
            table = table[table.Este != 0]
            table = table.dropna(subset=['Nucleado'])
            table2 = table[['people', 'Nucleado', 'risk_pop', 'Este', 'Norte']]
            root5 = table2
            
            Label(text="Table Risk on the Population index             ", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
            pt.show()

    except Exception as e:
        pass
def proc_socioeconomicpop():
    rutas_dict = verificar_y_preparar_rutas()
    
    ruta2 = ruta.removesuffix('.shp') #Tanks
    ruta3 = ruta20.removesuffix('.shp') #households
    ruta4 = ruta10.removesuffix('.shp') #buildings
    ruta400 = ruta2 + "1.shp"
    ruta402 = ruta4 + "1.shp"
    ruta405 = ruta2 + "11.shp"
    ruta500 = ruta2 + "100.shp"
    ruta510 = ruta4 + "index.shp"
    ruta600 = ruta4 + "index001.shp"
    ruta520 = ruta4 + "salidassoc.shp"
    ruta521 = ruta4 + "salidassoc1.shp"
    ruta522 = ruta4 + "salidassoc2.shp"
    ruta523 = ruta4 + "salidassoc3.shp"
    ruta5233 = ruta4 + "salidaspopulationindex.shp"
    
    import os

    def borrar_shapefiles(*rutas):
        for ruta in rutas:
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
                    
                else:
                    pass
            except Exception as e:
                continue
    
    rutas_shapefiles = [
        ruta510, ruta600, ruta520, ruta521, ruta522, ruta523
    ]
    
    borrar_shapefiles(*rutas_shapefiles)

    
    from shapely.geometry import shape, Point        
    import geopandas as gpd
    gdf0 = gpd.read_file(ruta400)
    gdf = gdf0[gdf0.IDTK != 255]
    gdf = gdf[gdf.IDTK != 0]
    gdf.to_file(ruta500)
    df = df00
    if df['Impact Radius'].sum() == 0:
        # Establece el valor de la primera fila de 'Impact Radius' a 1
        df.loc[0, 'Impact Radius'] = 1
    df498 = df.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    
    for column in df498.columns:
        if column != 'Class':
            df498[column] = pd.to_numeric(df498[column], errors='coerce')         
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    from shapely.geometry import shape, Point
    centroids = [] #Empy
    centroids = calcular_centroides_de_shapefile(ruta500) #Funcion centroides
       
    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    #df4 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    def intersect_and_save(shp1, shp2, output_shp):
        # Realiza la intersección entre los shapefiles
        g1 = gpd.GeoDataFrame.from_file(shp1)
        g2 = gpd.GeoDataFrame.from_file(shp2)
        # Calcula el área de cada polígono de la intersección
        g2['area'] = g2['geometry'].apply(lambda x: x.area)
        inter = gpd.overlay(g1, g2, how='intersection')
        # Reorganiza las columnas del GeoDataFrame
        inter = inter.reindex(columns=['index', 'people', 'Nucleado', 'area', 'geometry'])
        inter = inter[inter['people'] != 0]
        inter['density'] = inter['people']/(inter['area']/100)
        inter['risk_pop'] = inter['density']
        # Guarda el resultado en un archivo shapefile
        inter.to_file(output_shp)
    
    # Define la ruta base
    ruta22 = ruta10.removesuffix('.shp')
    ruta4 = ruta10
    
    poligonos = gpd.read_file(ruta10)

    # Unite all polygons into a single geometric object
    union_geometria = poligonos.unary_union

    # Create a new polygon that covers all the space within the shapefile
    xmin, ymin, xmax, ymax = union_geometria.bounds
    nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    # Create a new layer for the filled polygons
    poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})

    # Merge the original and filled polygon layers
    poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')

    # Create a new polygon that covers all the space inside the shapefile, but twice the size
    xmin, ymin, xmax, ymax = nuevo_poligono.bounds
    doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                              (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])

    # Create a new layer for the double polygon
    doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})

    # Merge the full polygon layers and the double polygon into a single layer
    fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))

    # Save the entire layer as a new shapefile
    fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
    # load the two shapefiles to be merged
    shp1 = gpd.read_file(ruta20)
    shp2 = gpd.read_file(ruta405)
    #shp3 = gpd.read_file(ruta10)
    
    # Perform spatial merge of the two shapefiles
    fusion = gpd.GeoDataFrame(pd.concat([shp1, shp2], ignore_index=True), crs=shp1.crs)
    fusion['people'] = fusion['people'].fillna(0)
    fusion['Nucleado'] = fusion['Nucleado'].fillna(0)
    from shapely.geometry import MultiPoint
    fusion1 = fusion.explode("geometry").reset_index(drop=True)
    fusion = fusion1.drop(['Id', 'H'], axis=1)
    # Save the entire layer as a new shapefile
    fusion.to_file(ruta500, driver='ESRI Shapefile')
    ruta4 = ruta500
        
    # Define la lista de sufijos
    sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
    # Genera las rutas de los archivos shapefile
    rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
    ruta7 = rutas[0]  
    ruta70 = rutas[1]
    ruta71 = rutas[2]
    ruta72 = rutas[3]
    ruta73 = rutas[4]
    ruta74 = rutas[5]
    ruta75 = rutas[6]
    ruta76 = rutas[7]
    
    shp = work[0][0]
    shp1 = work[0][1]
    shp2 = work[0][2]
    shp3 = work[0][3]
    shp4 = work[0][4]
    shp5 = work[0][5]
    shp6 = work1[0][0]
    
    z0 = gpd.read_file(shp)
    z1 = gpd.read_file(shp1)
    z2 = gpd.read_file(shp2)
    z3 = gpd.read_file(shp3)
    z4 = gpd.read_file(shp4)
    z5 = gpd.read_file(shp5)
    z6 = gpd.read_file(shp6)
    zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
    
    intersect_and_save(shp, ruta4, ruta70)
    intersect_and_save(shp1, ruta4, ruta71)
    intersect_and_save(shp2, ruta4, ruta72)
    intersect_and_save(shp3, ruta4, ruta73)
    intersect_and_save(shp4, ruta4, ruta74)
    intersect_and_save(shp5, ruta4, ruta75)
    intersect_and_save(shp6, ruta4, ruta76)
    
    df501 = (df498['Probit People']).div(1)
    df503 = pd.DataFrame(df501)
    
    #probabilities value
    pr = df503['Probit People'][0]/1
    pr1 = df503['Probit People'][1]/1
    pr2 = df503['Probit People'][2]/1
    pr3 = df503['Probit People'][3]/1
    pr4 = df503['Probit People'][4]/1
    pr5 = df503['Probit People'][5]/1
    pr6 = df503['Probit People'][6]/1
    
    v0 = 0 + (float(frec.get()))
    v1 = pr1 * ( (float(frec.get()) ) + 0)
    v2 = pr2 * ( (float(frec.get()) ) + 0)
    v3 = pr3 * ( (float(frec.get()) ) + 0)
    v4 = pr4 * ( (float(frec.get()) ) + 0)
    v5 = pr5 * ( (float(frec.get()) ) + 0)
    v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
    
    v00 = []
    v00.append (v0)
    v00.append (v1)
    v00.append (v2)
    v00.append (v3)
    v00.append (v4)
    v00.append (v5)
    v00.append (v6)
    
    zone = zone.assign(people=v00)
    
    gdf = gpd.read_file(ruta70)
    gdf1 = gpd.read_file(ruta71)
    gdf2 = gpd.read_file(ruta72)
    gdf3 = gpd.read_file(ruta73)
    gdf4 = gpd.read_file(ruta74)
    gdf5 = gpd.read_file(ruta75)
    gdf6 = gpd.read_file(ruta76)
    
    gdf['risk_pop'] *= pr1
    gdf1['risk_pop'] *= pr2
    gdf2['risk_pop'] *= pr3
    gdf3['risk_pop'] *= pr4
    gdf4['risk_pop'] *= pr5
    gdf5['risk_pop'] *= pr6
    gdf6['risk_pop'] *= pr
    
    gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
    gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
    gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
    gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
    gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
    gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
    gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
    
    import geopandas as gpd
    from shapely.geometry import Point
    gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
    gdf_merge = gdf_merge.reset_index(drop=True)
    gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
    gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
    gdf_mergef = gdf_merge.loc[gdf_merge['Nucleado'] == 1]
    if not gdf_mergef.empty:
        sumaareas = (gdf_mergef['areap'].sum()) / 10000
        sumapersonas = gdf_mergef['people'].sum()
        densitotal = sumapersonas / sumaareas if sumaareas != 0 else 0
    else:
        pass

        
    gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
    gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
    gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
    # Verificar si gdf_mergedis1 está vacío
    if not gdf_mergedis1.empty:
        # Realizar las operaciones si gdf_mergedis1 no está vacío
        gdf_mergedis1['people'] = (gdf_mergedis1['areap'] / 100) * gdf_mergedis1['density']
        gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
        sumapersonasdis = gdf_mergedis1['people'].sum()
    
        # Para evitar la división por cero, verifica si sumapersonasdis no es igual a cero
        if sumapersonasdis != 0:
            gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
        else:
            gdf_mergedis1['risk_pop'] = 0
    
        gdf_mergedis1.fillna(0, inplace=True)
    else:
        pass
    
    
    if not gdf_mergef.empty and sumaareas != 0 and densitotal != 0:
        gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap'] / 100)) / (sumaareas * densitotal)
    else:
        pass

    #gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap']/100))/(sumaareas * densitotal)
       
    gdf_merge = gdf_mergedis1.append (gdf_mergef)
    gdf_merge = gdf_merge.reset_index(drop=True)
    if not gdf_merge.empty:
        gdf_merge.to_file(ruta510)
    else:
        pass
    
    
    #procesar_centroides_y_points(ruta510, gdf_merge, zone) #Funcion salidas puntos
    centroids = [] #Empy
       
    try:
             
        with fiona.open(ruta510) as f:
            # Itera sobre todas las entradas del shapefile
            for feature in f:
                # Obtiene la geometría del polígono
                polygon = shape(feature['geometry'])
                # Calcula el centroide del polígono
                centroid = polygon.centroid
                # Almacena las coordenadas del centroide en la lista
                centroids.append((centroid.x, centroid.y))
    except Exception as e:
        pass

    # DataFrame centroids
    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
        
    points = df_concatenado.copy()
    # change geometry 
    points['geometry'] = points['geometry'].centroid
    
    
    from shapely.geometry import MultiPoint
    import geopandas as gpd
    
    # Suponiendo que 'zone' es un GeoDataFrame existente
    # zone = ...
    
    if not zone.empty:
        points2 = zone.copy()
        points2['risk_pop'] = points2['people']
        points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
        filas= points2['geometry']
    else:
        pass
    
    points2 = points2[~points2['geometry'].isna() & (points2['geometry'] != '')]
    df_puntos1 = points2.explode("geometry").reset_index(drop=True)
    
    df_puntos = df_puntos1
        
    points3 = points.append(df_puntos, ignore_index=True)
    points3['risk_pop'] = points3['risk_pop'].fillna(0)
    points5 = points3[points3.risk_pop != 0]
    points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
    min_value = points6['risk_pop'].min()
    points7 = points6[points6.risk_pop != min_value]
    points7 = points7.drop(['area'], axis=1)
    if 'level_0' in points7.columns:
        points7 = points7.drop(columns=['level_0'])
    
    # Cambiar la columna 'index' a tipo object si existe
    if 'index' in points7.columns:
        points7['index'] = points7['index'].astype(object)
    
    # Cambiar todas las otras columnas a float64, excepto 'geometry'
    for column in points7.columns:
        if column != 'geometry' and column != 'index':
            points7[column] = points7[column].astype(float)
    #points7.to_file(ruta520)
    
    if not points7.empty:
        points7.to_file(ruta520)
    else:
        pass
    ##################################
    rutas_shapefiles = [
        ruta521
    ]
    
    borrar_shapefiles(*rutas_shapefiles)
    gdf0 = gpd.read_file(ruta400)
    gdf = gdf0[gdf0.IDTK != 255]
    gdf = gdf[gdf.IDTK != 0]
    gdf.to_file(ruta500)
    df = df11
    if df['Impact Radius'].sum() == 0:
        # Establece el valor de la primera fila de 'Impact Radius' a 1
        df.loc[0, 'Impact Radius'] = 1
    df498 = df.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    
    for column in df498.columns:
        if column != 'Class':
            df498[column] = pd.to_numeric(df498[column], errors='coerce')         
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    from shapely.geometry import shape, Point
    centroids = [] #Empy
    centroids = calcular_centroides_de_shapefile(ruta500) #Funcion centroides
       
    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    #df4 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    def intersect_and_save(shp1, shp2, output_shp):
        # Realiza la intersección entre los shapefiles
        g1 = gpd.GeoDataFrame.from_file(shp1)
        g2 = gpd.GeoDataFrame.from_file(shp2)
        # Calcula el área de cada polígono de la intersección
        g2['area'] = g2['geometry'].apply(lambda x: x.area)
        inter = gpd.overlay(g1, g2, how='intersection')
        # Reorganiza las columnas del GeoDataFrame
        inter = inter.reindex(columns=['index', 'people', 'Nucleado', 'area', 'geometry'])
        inter = inter[inter['people'] != 0]
        inter['density'] = inter['people']/(inter['area']/100)
        inter['risk_pop'] = inter['density']
        # Guarda el resultado en un archivo shapefile
        inter.to_file(output_shp)
    
    # Define la ruta base
    ruta22 = ruta10.removesuffix('.shp')
    ruta4 = ruta10
    
    poligonos = gpd.read_file(ruta10)

    # Unite all polygons into a single geometric object
    union_geometria = poligonos.unary_union

    # Create a new polygon that covers all the space within the shapefile
    xmin, ymin, xmax, ymax = union_geometria.bounds
    nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    # Create a new layer for the filled polygons
    poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})

    # Merge the original and filled polygon layers
    poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')

    # Create a new polygon that covers all the space inside the shapefile, but twice the size
    xmin, ymin, xmax, ymax = nuevo_poligono.bounds
    doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                              (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])

    # Create a new layer for the double polygon
    doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})

    # Merge the full polygon layers and the double polygon into a single layer
    fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))

    # Save the entire layer as a new shapefile
    fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
    # load the two shapefiles to be merged
    shp1 = gpd.read_file(ruta20)
    shp2 = gpd.read_file(ruta405)
    #shp3 = gpd.read_file(ruta10)
    
    # Perform spatial merge of the two shapefiles
    fusion = gpd.GeoDataFrame(pd.concat([shp1, shp2], ignore_index=True), crs=shp1.crs)
    fusion['people'] = fusion['people'].fillna(0)
    fusion['Nucleado'] = fusion['Nucleado'].fillna(0)
    from shapely.geometry import MultiPoint
    fusion1 = fusion.explode("geometry").reset_index(drop=True)
    fusion = fusion1.drop(['Id', 'H'], axis=1)
    # Save the entire layer as a new shapefile
    fusion.to_file(ruta500, driver='ESRI Shapefile')
    ruta4 = ruta500
        
    # Define la lista de sufijos
    sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
    # Genera las rutas de los archivos shapefile
    rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
    ruta7 = rutas[0]  
    ruta70 = rutas[1]
    ruta71 = rutas[2]
    ruta72 = rutas[3]
    ruta73 = rutas[4]
    ruta74 = rutas[5]
    ruta75 = rutas[6]
    ruta76 = rutas[7]
    
    shp = work[0][0]
    shp1 = work[0][1]
    shp2 = work[0][2]
    shp3 = work[0][3]
    shp4 = work[0][4]
    shp5 = work[0][5]
    shp6 = work1[0][0]
    
    z0 = gpd.read_file(shp)
    z1 = gpd.read_file(shp1)
    z2 = gpd.read_file(shp2)
    z3 = gpd.read_file(shp3)
    z4 = gpd.read_file(shp4)
    z5 = gpd.read_file(shp5)
    z6 = gpd.read_file(shp6)
    zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
    
    intersect_and_save(shp, ruta4, ruta70)
    intersect_and_save(shp1, ruta4, ruta71)
    intersect_and_save(shp2, ruta4, ruta72)
    intersect_and_save(shp3, ruta4, ruta73)
    intersect_and_save(shp4, ruta4, ruta74)
    intersect_and_save(shp5, ruta4, ruta75)
    intersect_and_save(shp6, ruta4, ruta76)
    
    df501 = (df498['Probit People']).div(1)
    df503 = pd.DataFrame(df501)
    
    #probabilities value
    pr = df503['Probit People'][0]/1
    pr1 = df503['Probit People'][1]/1
    pr2 = df503['Probit People'][2]/1
    pr3 = df503['Probit People'][3]/1
    pr4 = df503['Probit People'][4]/1
    pr5 = df503['Probit People'][5]/1
    pr6 = df503['Probit People'][6]/1
    
    v0 = 0 + (float(frec.get()))
    v1 = pr1 * ( (float(frec.get()) ) + 0)
    v2 = pr2 * ( (float(frec.get()) ) + 0)
    v3 = pr3 * ( (float(frec.get()) ) + 0)
    v4 = pr4 * ( (float(frec.get()) ) + 0)
    v5 = pr5 * ( (float(frec.get()) ) + 0)
    v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
    
    v00 = []
    v00.append (v0)
    v00.append (v1)
    v00.append (v2)
    v00.append (v3)
    v00.append (v4)
    v00.append (v5)
    v00.append (v6)
    
    zone = zone.assign(people=v00)
    
    gdf = gpd.read_file(ruta70)
    gdf1 = gpd.read_file(ruta71)
    gdf2 = gpd.read_file(ruta72)
    gdf3 = gpd.read_file(ruta73)
    gdf4 = gpd.read_file(ruta74)
    gdf5 = gpd.read_file(ruta75)
    gdf6 = gpd.read_file(ruta76)
    
    gdf['risk_pop'] *= pr1
    gdf1['risk_pop'] *= pr2
    gdf2['risk_pop'] *= pr3
    gdf3['risk_pop'] *= pr4
    gdf4['risk_pop'] *= pr5
    gdf5['risk_pop'] *= pr6
    gdf6['risk_pop'] *= pr
    
    gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
    gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
    gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
    gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
    gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
    gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
    gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
    
    import geopandas as gpd
    from shapely.geometry import Point
    gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
    gdf_merge = gdf_merge.reset_index(drop=True)
    gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
    gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
    gdf_mergef = gdf_merge.loc[gdf_merge['Nucleado'] == 1]
    if not gdf_mergef.empty:
        sumaareas = (gdf_mergef['areap'].sum()) / 10000
        sumapersonas = gdf_mergef['people'].sum()
        densitotal = sumapersonas / sumaareas if sumaareas != 0 else 0
    else:
        pass

        
    gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
    gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
    gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
    # Verificar si gdf_mergedis1 está vacío
    if not gdf_mergedis1.empty:
        # Realizar las operaciones si gdf_mergedis1 no está vacío
        gdf_mergedis1['people'] = (gdf_mergedis1['areap'] / 100) * gdf_mergedis1['density']
        gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
        sumapersonasdis = gdf_mergedis1['people'].sum()
    
        # Para evitar la división por cero, verifica si sumapersonasdis no es igual a cero
        if sumapersonasdis != 0:
            gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
        else:
            gdf_mergedis1['risk_pop'] = 0
    
        gdf_mergedis1.fillna(0, inplace=True)
    else:
        pass
    
    
    if not gdf_mergef.empty and sumaareas != 0 and densitotal != 0:
        gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap'] / 100)) / (sumaareas * densitotal)
    else:
        pass

    #gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap']/100))/(sumaareas * densitotal)
       
    gdf_merge = gdf_mergedis1.append (gdf_mergef)
    gdf_merge = gdf_merge.reset_index(drop=True)
    if not gdf_merge.empty:
        gdf_merge.to_file(ruta510)
    else:
        pass
    
    
    #procesar_centroides_y_points(ruta510, gdf_merge, zone) #Funcion salidas puntos
    centroids = [] #Empy
       
    try:
             
        with fiona.open(ruta510) as f:
            # Itera sobre todas las entradas del shapefile
            for feature in f:
                # Obtiene la geometría del polígono
                polygon = shape(feature['geometry'])
                # Calcula el centroide del polígono
                centroid = polygon.centroid
                # Almacena las coordenadas del centroide en la lista
                centroids.append((centroid.x, centroid.y))
    except Exception as e:
        pass

    # DataFrame centroids
    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
        
    points = df_concatenado.copy()
    # change geometry 
    points['geometry'] = points['geometry'].centroid
    
    
    from shapely.geometry import MultiPoint
    import geopandas as gpd
    
    # Suponiendo que 'zone' es un GeoDataFrame existente
    # zone = ...
    
    if not zone.empty:
        points2 = zone.copy()
        points2['risk_pop'] = points2['people']
        points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
        filas= points2['geometry']
    else:
        pass
    
        
    points2 = points2[~points2['geometry'].isna() & (points2['geometry'] != '')]
    
    df_puntos1 = points2.explode("geometry").reset_index(drop=True)
    
    df_puntos = df_puntos1
        
    points3 = points.append(df_puntos, ignore_index=True)
    points3['risk_pop'] = points3['risk_pop'].fillna(0)
    points5 = points3[points3.risk_pop != 0]
    points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
    min_value = points6['risk_pop'].min()
    points7 = points6[points6.risk_pop != min_value]
    points7 = points7.drop(['area'], axis=1)
    if 'level_0' in points7.columns:
        points7 = points7.drop(columns=['level_0'])
    
    # Cambiar la columna 'index' a tipo object si existe
    if 'index' in points7.columns:
        points7['index'] = points7['index'].astype(object)
    
    # Cambiar todas las otras columnas a float64, excepto 'geometry'
    for column in points7.columns:
        if column != 'geometry' and column != 'index':
            points7[column] = points7[column].astype(float)
    

    #points7.to_file(ruta520)
    if not points7.empty:
        points7.to_file(ruta521)
    else:
        pass
    #################################
    rutas_shapefiles = [
        ruta522
    ]
    
    borrar_shapefiles(*rutas_shapefiles)
    gdf0 = gpd.read_file(ruta400)
    gdf = gdf0[gdf0.IDTK != 255]
    gdf = gdf[gdf.IDTK != 0]
    gdf.to_file(ruta500)
    df = df22
    if df['Impact Radius'].sum() == 0:
        # Establece el valor de la primera fila de 'Impact Radius' a 1
        df.loc[0, 'Impact Radius'] = 1
    df498 = df.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    
    for column in df498.columns:
        if column != 'Class':
            df498[column] = pd.to_numeric(df498[column], errors='coerce')         
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    from shapely.geometry import shape, Point
    centroids = [] #Empy
    centroids = calcular_centroides_de_shapefile(ruta500) #Funcion centroides
       
    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    #df4 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    def intersect_and_save(shp1, shp2, output_shp):
        # Realiza la intersección entre los shapefiles
        g1 = gpd.GeoDataFrame.from_file(shp1)
        g2 = gpd.GeoDataFrame.from_file(shp2)
        # Calcula el área de cada polígono de la intersección
        g2['area'] = g2['geometry'].apply(lambda x: x.area)
        inter = gpd.overlay(g1, g2, how='intersection')
        # Reorganiza las columnas del GeoDataFrame
        inter = inter.reindex(columns=['index', 'people', 'Nucleado', 'area', 'geometry'])
        inter = inter[inter['people'] != 0]
        inter['density'] = inter['people']/(inter['area']/100)
        inter['risk_pop'] = inter['density']
        # Guarda el resultado en un archivo shapefile
        inter.to_file(output_shp)
    
    # Define la ruta base
    ruta22 = ruta10.removesuffix('.shp')
    ruta4 = ruta10
    
    poligonos = gpd.read_file(ruta10)

    # Unite all polygons into a single geometric object
    union_geometria = poligonos.unary_union

    # Create a new polygon that covers all the space within the shapefile
    xmin, ymin, xmax, ymax = union_geometria.bounds
    nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    # Create a new layer for the filled polygons
    poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})

    # Merge the original and filled polygon layers
    poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')

    # Create a new polygon that covers all the space inside the shapefile, but twice the size
    xmin, ymin, xmax, ymax = nuevo_poligono.bounds
    doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                              (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])

    # Create a new layer for the double polygon
    doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})

    # Merge the full polygon layers and the double polygon into a single layer
    fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))

    # Save the entire layer as a new shapefile
    fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
    # load the two shapefiles to be merged
    shp1 = gpd.read_file(ruta20)
    shp2 = gpd.read_file(ruta405)
    #shp3 = gpd.read_file(ruta10)
    
    # Perform spatial merge of the two shapefiles
    fusion = gpd.GeoDataFrame(pd.concat([shp1, shp2], ignore_index=True), crs=shp1.crs)
    fusion['people'] = fusion['people'].fillna(0)
    fusion['Nucleado'] = fusion['Nucleado'].fillna(0)
    from shapely.geometry import MultiPoint
    fusion1 = fusion.explode("geometry").reset_index(drop=True)
    fusion = fusion1.drop(['Id', 'H'], axis=1)
    # Save the entire layer as a new shapefile
    fusion.to_file(ruta500, driver='ESRI Shapefile')
    ruta4 = ruta500
        
    # Define la lista de sufijos
    sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
    # Genera las rutas de los archivos shapefile
    rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
    ruta7 = rutas[0]  
    ruta70 = rutas[1]
    ruta71 = rutas[2]
    ruta72 = rutas[3]
    ruta73 = rutas[4]
    ruta74 = rutas[5]
    ruta75 = rutas[6]
    ruta76 = rutas[7]
    
    shp = work[0][0]
    shp1 = work[0][1]
    shp2 = work[0][2]
    shp3 = work[0][3]
    shp4 = work[0][4]
    shp5 = work[0][5]
    shp6 = work1[0][0]
    
    z0 = gpd.read_file(shp)
    z1 = gpd.read_file(shp1)
    z2 = gpd.read_file(shp2)
    z3 = gpd.read_file(shp3)
    z4 = gpd.read_file(shp4)
    z5 = gpd.read_file(shp5)
    z6 = gpd.read_file(shp6)
    zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
    
    intersect_and_save(shp, ruta4, ruta70)
    intersect_and_save(shp1, ruta4, ruta71)
    intersect_and_save(shp2, ruta4, ruta72)
    intersect_and_save(shp3, ruta4, ruta73)
    intersect_and_save(shp4, ruta4, ruta74)
    intersect_and_save(shp5, ruta4, ruta75)
    intersect_and_save(shp6, ruta4, ruta76)
    
    df501 = (df498['Probit People']).div(1)
    df503 = pd.DataFrame(df501)
    
    #probabilities value
    pr = df503['Probit People'][0]/1
    pr1 = df503['Probit People'][1]/1
    pr2 = df503['Probit People'][2]/1
    pr3 = df503['Probit People'][3]/1
    pr4 = df503['Probit People'][4]/1
    pr5 = df503['Probit People'][5]/1
    pr6 = df503['Probit People'][6]/1
    
    v0 = 0 + (float(frec.get()))
    v1 = pr1 * ( (float(frec.get()) ) + 0)
    v2 = pr2 * ( (float(frec.get()) ) + 0)
    v3 = pr3 * ( (float(frec.get()) ) + 0)
    v4 = pr4 * ( (float(frec.get()) ) + 0)
    v5 = pr5 * ( (float(frec.get()) ) + 0)
    v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
    
    v00 = []
    v00.append (v0)
    v00.append (v1)
    v00.append (v2)
    v00.append (v3)
    v00.append (v4)
    v00.append (v5)
    v00.append (v6)
    
    zone = zone.assign(people=v00)
    
    gdf = gpd.read_file(ruta70)
    gdf1 = gpd.read_file(ruta71)
    gdf2 = gpd.read_file(ruta72)
    gdf3 = gpd.read_file(ruta73)
    gdf4 = gpd.read_file(ruta74)
    gdf5 = gpd.read_file(ruta75)
    gdf6 = gpd.read_file(ruta76)
    
    gdf['risk_pop'] *= pr1
    gdf1['risk_pop'] *= pr2
    gdf2['risk_pop'] *= pr3
    gdf3['risk_pop'] *= pr4
    gdf4['risk_pop'] *= pr5
    gdf5['risk_pop'] *= pr6
    gdf6['risk_pop'] *= pr
    
    gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
    gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
    gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
    gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
    gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
    gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
    gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
    
    import geopandas as gpd
    from shapely.geometry import Point
    gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
    gdf_merge = gdf_merge.reset_index(drop=True)
    gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
    gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
    gdf_mergef = gdf_merge.loc[gdf_merge['Nucleado'] == 1]
    if not gdf_mergef.empty:
        sumaareas = (gdf_mergef['areap'].sum()) / 10000
        sumapersonas = gdf_mergef['people'].sum()
        densitotal = sumapersonas / sumaareas if sumaareas != 0 else 0
    else:
        pass

        
    gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
    gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
    gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
    # Verificar si gdf_mergedis1 está vacío
    if not gdf_mergedis1.empty:
        # Realizar las operaciones si gdf_mergedis1 no está vacío
        gdf_mergedis1['people'] = (gdf_mergedis1['areap'] / 100) * gdf_mergedis1['density']
        gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
        sumapersonasdis = gdf_mergedis1['people'].sum()
    
        # Para evitar la división por cero, verifica si sumapersonasdis no es igual a cero
        if sumapersonasdis != 0:
            gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
        else:
            gdf_mergedis1['risk_pop'] = 0
    
        gdf_mergedis1.fillna(0, inplace=True)
    else:
        pass
    
    
    if not gdf_mergef.empty and sumaareas != 0 and densitotal != 0:
        gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap'] / 100)) / (sumaareas * densitotal)
    else:
        pass

    #gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap']/100))/(sumaareas * densitotal)
       
    gdf_merge = gdf_mergedis1.append (gdf_mergef)
    gdf_merge = gdf_merge.reset_index(drop=True)
    if not gdf_merge.empty:
        gdf_merge.to_file(ruta510)
    else:
        pass
    
    
    #procesar_centroides_y_points(ruta510, gdf_merge, zone) #Funcion salidas puntos
    centroids = [] #Empy
       
    try:
             
        with fiona.open(ruta510) as f:
            # Itera sobre todas las entradas del shapefile
            for feature in f:
                # Obtiene la geometría del polígono
                polygon = shape(feature['geometry'])
                # Calcula el centroide del polígono
                centroid = polygon.centroid
                # Almacena las coordenadas del centroide en la lista
                centroids.append((centroid.x, centroid.y))
    except Exception as e:
        pass

    # DataFrame centroids
    # DataFrame centroids
    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
        
    points = df_concatenado.copy()
    # change geometry 
    points['geometry'] = points['geometry'].centroid
    
    
    from shapely.geometry import MultiPoint
    import geopandas as gpd
    
    # Suponiendo que 'zone' es un GeoDataFrame existente
    # zone = ...
    
    if not zone.empty:
        points2 = zone.copy()
        points2['risk_pop'] = points2['people']
        points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
        filas= points2['geometry']
    else:
        pass
    points2 = points2[~points2['geometry'].isna() & (points2['geometry'] != '')]

    df_puntos1 = points2.explode("geometry").reset_index(drop=True)
    
    df_puntos = df_puntos1
        
    points3 = points.append(df_puntos, ignore_index=True)
    points3['risk_pop'] = points3['risk_pop'].fillna(0)
    points5 = points3[points3.risk_pop != 0]
    points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
    min_value = points6['risk_pop'].min()
    points7 = points6[points6.risk_pop != min_value]
    points7 = points7.drop(['area'], axis=1)
    if 'level_0' in points7.columns:
        points7 = points7.drop(columns=['level_0'])
    
    # Cambiar la columna 'index' a tipo object si existe
    if 'index' in points7.columns:
        points7['index'] = points7['index'].astype(object)
    
    # Cambiar todas las otras columnas a float64, excepto 'geometry'
    for column in points7.columns:
        if column != 'geometry' and column != 'index':
            points7[column] = points7[column].astype(float)
    #points7.to_file(ruta520)
    if not points7.empty:
        points7.to_file(ruta522)
    else:
        pass
    ##################################
    rutas_shapefiles = [
        ruta523
    ]
    gdf0 = gpd.read_file(ruta400)
    gdf = gdf0[gdf0.IDTK != 255]
    gdf = gdf[gdf.IDTK != 0]
    gdf.to_file(ruta500)
    df = df33
    if df['Impact Radius'].sum() == 0:
        # Establece el valor de la primera fila de 'Impact Radius' a 1
        df.loc[0, 'Impact Radius'] = 1
    df498 = df.sort_values(by='Impact Radius')
    df498 = df498.reset_index(inplace=False, drop=True)
    files = len(df498)
    file = 7 - files
    for i in range(file):
        df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
    
    for column in df498.columns:
        if column != 'Class':
            df498[column] = pd.to_numeric(df498[column], errors='coerce')         
    df499 = (df498['Impact Radius']).div(1) #35971.22302158273
    df500 = pd.DataFrame(df499)
    import os
    parent_dir = ruta2.rsplit('/', 1)[0]
    os.chdir(parent_dir)
    
    from shapely.geometry import shape, Point
    centroids = [] #Empy
    centroids = calcular_centroides_de_shapefile(ruta500) #Funcion centroides
       
    # DataFrame centroids
    df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    shapefile = gpd.read_file(ruta500)
    df_concatenado = pd.concat([shapefile, df5], axis=1)
        
    shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
    shapefile02 = shapefile01.reset_index(drop=True)
    shapefile03 = shapefile02[shapefile02.IDTK != 255]
    shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
    shapefile03['IDTK'] = shapefile03['IDTK']  + 1
    
    df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

    df3 = df600
    #df4 = df600
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    #generate the rip buffer
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df3{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
        
        shapefile1 = gpd.read_file(filename)
        if index < len(df) - 1:
            shapefile2 = gpd.read_file(f"df3{index+2}.shp")
        else:
            # si es el último shapefile generado, no hacemos nada más
            continue
        geometry1 = shapefile1["geometry"]
        geometry2 = shapefile2["geometry"]
        
        # convertimos las columnas a GeoSeries
        geoseries1 = gpd.GeoSeries(geometry1)
        geoseries2 = gpd.GeoSeries(geometry2)
        
        # realizamos el clip usando la función difference de geopandas
        clipped_shapefile = geoseries2.difference(geoseries1, align=True)
        
        # guardamos el resultado en un archivo .shp nuevo
        clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    for index, row in df500.iterrows():
        r = row[0]
        buffer = df3.buffer(r)
        filename = f"df55{index+1}.shp"
        buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
    
    import os
    import glob
    # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
    shp_files = glob.glob(parent_dir + '/*.shp')
    for i, shp_file in enumerate(shp_files):
        shp_files[i] = shp_file.replace("\\", "/")
            
    work = [w for w in shp_files if w.find('df3') != -1]
    work1 = [w for w in shp_files if w.find('df551') != -1]
    work = pd.DataFrame(work)
    work1 = pd.DataFrame(work1)
    
    def intersect_and_save(shp1, shp2, output_shp):
        # Realiza la intersección entre los shapefiles
        g1 = gpd.GeoDataFrame.from_file(shp1)
        g2 = gpd.GeoDataFrame.from_file(shp2)
        # Calcula el área de cada polígono de la intersección
        g2['area'] = g2['geometry'].apply(lambda x: x.area)
        inter = gpd.overlay(g1, g2, how='intersection')
        # Reorganiza las columnas del GeoDataFrame
        inter = inter.reindex(columns=['index', 'people', 'Nucleado', 'area', 'geometry'])
        inter = inter[inter['people'] != 0]
        inter['density'] = inter['people']/(inter['area']/100)
        inter['risk_pop'] = inter['density']
        # Guarda el resultado en un archivo shapefile
        inter.to_file(output_shp)
    
    # Define la ruta base
    ruta22 = ruta10.removesuffix('.shp')
    ruta4 = ruta10
    
    poligonos = gpd.read_file(ruta10)

    # Unite all polygons into a single geometric object
    union_geometria = poligonos.unary_union

    # Create a new polygon that covers all the space within the shapefile
    xmin, ymin, xmax, ymax = union_geometria.bounds
    nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    # Create a new layer for the filled polygons
    poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})

    # Merge the original and filled polygon layers
    poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')

    # Create a new polygon that covers all the space inside the shapefile, but twice the size
    xmin, ymin, xmax, ymax = nuevo_poligono.bounds
    doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                              (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])

    # Create a new layer for the double polygon
    doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})

    # Merge the full polygon layers and the double polygon into a single layer
    fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))

    # Save the entire layer as a new shapefile
    fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
    # load the two shapefiles to be merged
    shp1 = gpd.read_file(ruta20)
    shp2 = gpd.read_file(ruta405)
    #shp3 = gpd.read_file(ruta10)
    
    # Perform spatial merge of the two shapefiles
    fusion = gpd.GeoDataFrame(pd.concat([shp1, shp2], ignore_index=True), crs=shp1.crs)
    fusion['people'] = fusion['people'].fillna(0)
    fusion['Nucleado'] = fusion['Nucleado'].fillna(0)
    from shapely.geometry import MultiPoint
    fusion1 = fusion.explode("geometry").reset_index(drop=True)
    fusion = fusion1.drop(['Id', 'H'], axis=1)
    # Save the entire layer as a new shapefile
    fusion.to_file(ruta500, driver='ESRI Shapefile')
    ruta4 = ruta500
        
    # Define la lista de sufijos
    sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
    # Genera las rutas de los archivos shapefile
    rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
    ruta7 = rutas[0]  
    ruta70 = rutas[1]
    ruta71 = rutas[2]
    ruta72 = rutas[3]
    ruta73 = rutas[4]
    ruta74 = rutas[5]
    ruta75 = rutas[6]
    ruta76 = rutas[7]
    
    shp = work[0][0]
    shp1 = work[0][1]
    shp2 = work[0][2]
    shp3 = work[0][3]
    shp4 = work[0][4]
    shp5 = work[0][5]
    shp6 = work1[0][0]
    
    z0 = gpd.read_file(shp)
    z1 = gpd.read_file(shp1)
    z2 = gpd.read_file(shp2)
    z3 = gpd.read_file(shp3)
    z4 = gpd.read_file(shp4)
    z5 = gpd.read_file(shp5)
    z6 = gpd.read_file(shp6)
    zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
    
    intersect_and_save(shp, ruta4, ruta70)
    intersect_and_save(shp1, ruta4, ruta71)
    intersect_and_save(shp2, ruta4, ruta72)
    intersect_and_save(shp3, ruta4, ruta73)
    intersect_and_save(shp4, ruta4, ruta74)
    intersect_and_save(shp5, ruta4, ruta75)
    intersect_and_save(shp6, ruta4, ruta76)
    
    df501 = (df498['Probit People']).div(1)
    df503 = pd.DataFrame(df501)
    
    #probabilities value
    pr = df503['Probit People'][0]/1
    pr1 = df503['Probit People'][1]/1
    pr2 = df503['Probit People'][2]/1
    pr3 = df503['Probit People'][3]/1
    pr4 = df503['Probit People'][4]/1
    pr5 = df503['Probit People'][5]/1
    pr6 = df503['Probit People'][6]/1
    
    v0 = 0 + (float(frec.get()))
    v1 = pr1 * ( (float(frec.get()) ) + 0)
    v2 = pr2 * ( (float(frec.get()) ) + 0)
    v3 = pr3 * ( (float(frec.get()) ) + 0)
    v4 = pr4 * ( (float(frec.get()) ) + 0)
    v5 = pr5 * ( (float(frec.get()) ) + 0)
    v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
    
    v00 = []
    v00.append (v0)
    v00.append (v1)
    v00.append (v2)
    v00.append (v3)
    v00.append (v4)
    v00.append (v5)
    v00.append (v6)
    
    zone = zone.assign(people=v00)
    
    gdf = gpd.read_file(ruta70)
    gdf1 = gpd.read_file(ruta71)
    gdf2 = gpd.read_file(ruta72)
    gdf3 = gpd.read_file(ruta73)
    gdf4 = gpd.read_file(ruta74)
    gdf5 = gpd.read_file(ruta75)
    gdf6 = gpd.read_file(ruta76)
    
    gdf['risk_pop'] *= pr1
    gdf1['risk_pop'] *= pr2
    gdf2['risk_pop'] *= pr3
    gdf3['risk_pop'] *= pr4
    gdf4['risk_pop'] *= pr5
    gdf5['risk_pop'] *= pr6
    gdf6['risk_pop'] *= pr
    
    gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
    gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
    gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
    gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
    gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
    gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
    gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
    
    import geopandas as gpd
    from shapely.geometry import Point
    gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
    gdf_merge = gdf_merge.reset_index(drop=True)
    gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
    gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
    gdf_mergef = gdf_merge.loc[gdf_merge['Nucleado'] == 1]
    if not gdf_mergef.empty:
        sumaareas = (gdf_mergef['areap'].sum()) / 10000
        sumapersonas = gdf_mergef['people'].sum()
        densitotal = sumapersonas / sumaareas if sumaareas != 0 else 0
    else:
        pass

        
    gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
    gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
    gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
    # Verificar si gdf_mergedis1 está vacío
    if not gdf_mergedis1.empty:
        # Realizar las operaciones si gdf_mergedis1 no está vacío
        gdf_mergedis1['people'] = (gdf_mergedis1['areap'] / 100) * gdf_mergedis1['density']
        gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
        sumapersonasdis = gdf_mergedis1['people'].sum()
    
        # Para evitar la división por cero, verifica si sumapersonasdis no es igual a cero
        if sumapersonasdis != 0:
            gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
        else:
            gdf_mergedis1['risk_pop'] = 0
    
        gdf_mergedis1.fillna(0, inplace=True)
    else:
        pass
    
    
    if not gdf_mergef.empty and sumaareas != 0 and densitotal != 0:
        gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap'] / 100)) / (sumaareas * densitotal)
    else:
        pass

    #gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap']/100))/(sumaareas * densitotal)
       
    gdf_merge = gdf_mergedis1.append (gdf_mergef)
    gdf_merge = gdf_merge.reset_index(drop=True)
    if not gdf_merge.empty:
        gdf_merge.to_file(ruta510)
    else:
        pass
    
    
    #procesar_centroides_y_points(ruta510, gdf_merge, zone) #Funcion salidas puntos
    centroids = [] #Empy
       
    try:
             
        with fiona.open(ruta510) as f:
            # Itera sobre todas las entradas del shapefile
            for feature in f:
                # Obtiene la geometría del polígono
                polygon = shape(feature['geometry'])
                # Calcula el centroide del polígono
                centroid = polygon.centroid
                # Almacena las coordenadas del centroide en la lista
                centroids.append((centroid.x, centroid.y))
    except Exception as e:
        pass

    # DataFrame centroids
    df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
    df_concatenado = pd.concat([gdf_merge, df6], axis=1)
        
    points = df_concatenado.copy()
    # change geometry 
    points['geometry'] = points['geometry'].centroid
    
    
    from shapely.geometry import MultiPoint
    import geopandas as gpd
    
    # Suponiendo que 'zone' es un GeoDataFrame existente
    # zone = ...
    
    if not zone.empty:
        points2 = zone.copy()
        points2['risk_pop'] = points2['people']
        points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
        filas= points2['geometry']
    else:
        pass
    points2 = points2[~points2['geometry'].isna() & (points2['geometry'] != '')]

    df_puntos1 = points2.explode("geometry").reset_index(drop=True)
    
    df_puntos = df_puntos1
        
    points3 = points.append(df_puntos, ignore_index=True)
    points3['risk_pop'] = points3['risk_pop'].fillna(0)
    points5 = points3[points3.risk_pop != 0]
    points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
    min_value = points6['risk_pop'].min()
    points7 = points6[points6.risk_pop != min_value]
    points7 = points7.drop(['area'], axis=1)
    if 'level_0' in points7.columns:
        points7 = points7.drop(columns=['level_0'])
    
    # Cambiar la columna 'index' a tipo object si existe
    if 'index' in points7.columns:
        points7['index'] = points7['index'].astype(object)
    
    # Cambiar todas las otras columnas a float64, excepto 'geometry'
    for column in points7.columns:
        if column != 'geometry' and column != 'index':
            points7[column] = points7[column].astype(float)
    #points7.to_file(ruta520)
    if not points7.empty:
        points7.to_file(ruta523)
    else:
        pass
        
    ##################################
    rutas_shapefiles = [ruta520, ruta521, ruta522, ruta523]
    visualizar_interpolacion_y_tablapop(rutas_shapefiles, ruta120, ventana, ruta5233) #Realiza la graficacion
         
    

############################################################################################

def visualizar_indices_hogares2(ruta120, rutas, ventana, ruta547):
    try:
        from matplotlib.colors import ListedColormap
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Cargar y mostrar datos TIFF como imagen de fondo usando GDAL
        ds = gdal.Open(ruta120)
        if ds is None:
            messagebox.showerror("Error", f"No se pudo abrir el archivo TIFF {ruta120}")
            return

        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]

        # Crear figura y eje para la trama
        fig, ax = plt.subplots()
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
        
        gdfs = []
        for ruta in rutas:
            try:
                gdf = gpd.read_file(ruta)
                gdfs.append(gdf)
            except Exception as e:
                print(f"No se pudo cargar el archivo {ruta}: {e}")
        
        if gdfs:
            points_gdf = pd.concat(gdfs, ignore_index=True)
        else:
            points_gdf = pd.DataFrame()
        
        points_gdf = points_gdf.dropna(subset=['geometry'])
        min_value = points_gdf['risk_pop'].dropna().min() * 0.001
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_pop'] = min_value
        
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
        # # Ajustando valores mínimos para 'risk'
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
        
        
        
        # Asumiendo que x, y, z_filled, y ax ya están definidos correctamente
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)  # Ajusta según sea necesario
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Households index')
        
        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)

        if not points_gdf.empty:
            points_gdf.to_file(ruta547, driver='ESRI Shapefile')
            table = points_gdf[points_gdf.risk_pop != 0]
            table = table[table.Este != 0]
            table = table.dropna(subset=['Nucleado'])
            table2 = table[['people', 'Nucleado', 'risk_pop', 'Este', 'Norte']]
            root5 = table2
            
            Label(text="Table Risk on the Households index             ", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
            pt.show()

    except Exception as e:
        messagebox.showerror(f"No se presenta afectacion")
        pass


def proc_socioeconomicpop1():
    ruta2 = ruta.removesuffix('.shp') #Tanks
    ruta400 = ruta2 + "1.shp"
    ruta3 = ruta20.removesuffix('.shp') #households
    ruta401 = ruta3 + "1.shp"
    ruta4 = ruta20.removesuffix('.shp') #buildings - ruta10
    ruta402 = ruta4 + "1.shp"
    ruta405 = ruta2 + "11.shp"
    ruta500 = ruta2 + "100.shp"
    ruta510 = ruta4 + "index.shp"
    ruta600 = ruta4 + "index001.shp"
    ruta520 = ruta4 + "salidaspopb.shp"
    ruta521 = ruta4 + "salidaspopb1.shp" #df11
    ruta522 = ruta4 + "salidaspopb2.shp" #df22
    ruta523 = ruta4 + "salidaspopb3.shp" #df33
    ruta547 = ruta4 + "salidaspopbindex.shp" #Indice
    ruta548 = ruta4 + "salidasinfraestructureindex.shp" #Indice
    # Lista de archivos para verificar y limpiar
    def borrar_shapefiles(*rutas):
        for ruta in rutas:
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
                    
                else:
                    pass
            except Exception as e:
                continue
    
    rutas_shapefiles = [
        ruta510, ruta520, ruta521, ruta522, ruta523, ruta600
    ]
    
    borrar_shapefiles(*rutas_shapefiles)
    
    def funcion_principal_df00():
        global df00
        df11a = df00
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            
            from shapely.geometry import shape, Point        
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                # Calcula el área de cada polígono de la intersección
                g2['area'] = g2['geometry'].apply(lambda x: x.area)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'Nucleado', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['people'].fillna(0, inplace=True)
                inter['area'].fillna(0, inplace=True)
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['density']
                inter['risk_pop'].fillna(0, inplace=True)
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            ruta22 = ruta20.removesuffix('.shp')
            ruta4 = ruta20
            
            poligonos = gpd.read_file(ruta20)
        
            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union
        
            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        
            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})
        
            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')
        
            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])
        
            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})
        
            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))
        
            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta20)
            shp2 = gpd.read_file(ruta405)
            #shp3 = gpd.read_file(ruta10)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([shp1, shp2], ignore_index=True), crs=shp1.crs)
            fusion['people'] = fusion['people'].fillna(0)
            fusion['Nucleado'] = fusion['Nucleado'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            fusion = fusion1.drop(['Id', 'H'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
            
            # dens = gpd.read_file(ruta4)
            
            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit House']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit House'][0]/1
            pr1 = df503['Probit House'][1]/1
            pr2 = df503['Probit House'][2]/1
            pr3 = df503['Probit House'][3]/1
            pr4 = df503['Probit House'][4]/1
            pr5 = df503['Probit House'][5]/1
            pr6 = df503['Probit House'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
            gdf_mergef = gdf_merge.loc[gdf_merge['Nucleado'] == 1]
            sumaareas = (gdf_mergef['areap'].sum())/10000
            sumapersonas = gdf_mergef['people'].sum()
            densitotal = sumapersonas / sumaareas
            try:
                gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
                gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
                
                # Asegurarse de que las columnas 'areap' y 'density' no generen errores
                if 'geometry' in gdf_mergedis1.columns:
                    gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
                else:
                    gdf_mergedis1['areap'] = 0
            
                # Rellenar NaNs en la columna 'density'
                gdf_mergedis1['density'].fillna(0, inplace=True)
            
                # Calcular el número de personas solo si 'density' no está vacío
                if not gdf_mergedis1['density'].empty:
                    gdf_mergedis1['people'] = ((gdf_mergedis1['areap'] / 100) * gdf_mergedis1['density']) / 5
                else:
                    gdf_mergedis1['people'] = 0
            
                # Rellenar NaNs en la columna 'people'
                gdf_mergedis1['people'].fillna(0, inplace=True)
            
                # Calcular 'risk_pop' solo si 'density' no está vacío
                if not gdf_mergedis1['density'].empty:
                    gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
                else:
                    gdf_mergedis1['risk_pop'] = 0
            
                # Rellenar NaNs en la columna 'risk_pop'
                gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            
                # Calcular la suma de personas solo si 'people' no está vacío
                if not gdf_mergedis1['people'].empty:
                    sumapersonasdis = (gdf_mergedis1['people'].sum()) / 5
                else:
                    sumapersonasdis = 1  # Evitar división por cero
            
                # Calcular 'risk_pop' nuevamente
                gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
            
                # Rellenar NaNs en las columnas 'risk_pop' y 'areap'
                gdf_mergef['risk_pop'].fillna(0, inplace=True)
                gdf_merge['areap'].fillna(0, inplace=True)
            
                # Calcular 'risk_pop' para gdf_mergef
                if 'areap' in gdf_mergef.columns and 'risk_pop' in gdf_mergef.columns:
                    gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap'] / 100)) / (sumaareas * densitotal)
            
                # Rellenar NaNs en las columnas 'risk_pop' de gdf_mergedis1
                gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            
                # Combinar los GeoDataFrames y restablecer el índice
                gdf_merge = gdf_mergedis1.append(gdf_mergef)
                gdf_merge = gdf_merge.reset_index(drop=True)
            
                # Guardar el GeoDataFrame combinado en un archivo
                gdf_merge.to_file(ruta510)
            
            except Exception as e:
                print(f"Error procesando los datos: {e}")
                pass    
            # gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
            # gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
            # gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
            # gdf_mergedis1['people'] = ((gdf_mergedis1['areap']/100) * gdf_mergedis1['density'])/5 #average number of people
            # gdf_mergedis1['people'].fillna(0, inplace=True)
            # gdf_mergedis1['density'].fillna(0, inplace=True)
            # gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
            # sumapersonasdis = (gdf_mergedis1['people'].sum())/5 #average number of people
            # gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
            
            # gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap']/100))/(sumaareas * densitotal)
               
            # gdf_merge = gdf_mergedis1.append (gdf_mergef)
            # gdf_merge = gdf_merge.reset_index(drop=True)
            # gdf_merge['areap'].fillna(0, inplace=True)
            # gdf_mergef['risk_pop'].fillna(0, inplace=True)
            # gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            # gdf_merge.to_file(ruta510)
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
                
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['FID', 'area'], axis=1)
            #points7.to_file(ruta520)
            
            if not points7.empty:
                points7.to_file(ruta520)
            else:
                pass
        else:
            pass
    
    funcion_principal_df00()            
    ###################################
    def funcion_principal_df11():
        global df11
        df11a = df11
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            
            from shapely.geometry import shape, Point        
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
                                
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                # Calcula el área de cada polígono de la intersección
                g2['area'] = g2['geometry'].apply(lambda x: x.area)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'Nucleado', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['people'].fillna(0, inplace=True)
                inter['area'].fillna(0, inplace=True)
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['density']
                inter['risk_pop'].fillna(0, inplace=True)
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            ruta22 = ruta20.removesuffix('.shp')
            ruta4 = ruta20
            
            poligonos = gpd.read_file(ruta20)
        
            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union
        
            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        
            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})
        
            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')
        
            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])
        
            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})
        
            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))
        
            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta20)
            shp2 = gpd.read_file(ruta405)
            #shp3 = gpd.read_file(ruta10)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([shp1, shp2], ignore_index=True), crs=shp1.crs)
            fusion['people'] = fusion['people'].fillna(0)
            fusion['Nucleado'] = fusion['Nucleado'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            fusion = fusion1.drop(['Id', 'H'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
            
            # dens = gpd.read_file(ruta4)
            
            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit House']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit House'][0]/1
            pr1 = df503['Probit House'][1]/1
            pr2 = df503['Probit House'][2]/1
            pr3 = df503['Probit House'][3]/1
            pr4 = df503['Probit House'][4]/1
            pr5 = df503['Probit House'][5]/1
            pr6 = df503['Probit House'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
            gdf_mergef = gdf_merge.loc[gdf_merge['Nucleado'] == 1]
            sumaareas = (gdf_mergef['areap'].sum())/10000
            sumapersonas = gdf_mergef['people'].sum()
            densitotal = sumapersonas / sumaareas
            
            try:
                gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
                gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
                
                # Asegurarse de que las columnas 'areap' y 'density' no generen errores
                if 'geometry' in gdf_mergedis1.columns:
                    gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
                else:
                    gdf_mergedis1['areap'] = 0
            
                # Rellenar NaNs en la columna 'density'
                gdf_mergedis1['density'].fillna(0, inplace=True)
            
                # Calcular el número de personas solo si 'density' no está vacío
                if not gdf_mergedis1['density'].empty:
                    gdf_mergedis1['people'] = ((gdf_mergedis1['areap'] / 100) * gdf_mergedis1['density']) / 5
                else:
                    gdf_mergedis1['people'] = 0
            
                # Rellenar NaNs en la columna 'people'
                gdf_mergedis1['people'].fillna(0, inplace=True)
            
                # Calcular 'risk_pop' solo si 'density' no está vacío
                if not gdf_mergedis1['density'].empty:
                    gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
                else:
                    gdf_mergedis1['risk_pop'] = 0
            
                # Rellenar NaNs en la columna 'risk_pop'
                gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            
                # Calcular la suma de personas solo si 'people' no está vacío
                if not gdf_mergedis1['people'].empty:
                    sumapersonasdis = (gdf_mergedis1['people'].sum()) / 5
                else:
                    sumapersonasdis = 1  # Evitar división por cero
            
                # Calcular 'risk_pop' nuevamente
                gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
            
                # Rellenar NaNs en las columnas 'risk_pop' y 'areap'
                gdf_mergef['risk_pop'].fillna(0, inplace=True)
                gdf_merge['areap'].fillna(0, inplace=True)
            
                # Calcular 'risk_pop' para gdf_mergef
                if 'areap' in gdf_mergef.columns and 'risk_pop' in gdf_mergef.columns:
                    gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap'] / 100)) / (sumaareas * densitotal)
            
                # Rellenar NaNs en las columnas 'risk_pop' de gdf_mergedis1
                gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            
                # Combinar los GeoDataFrames y restablecer el índice
                gdf_merge = gdf_mergedis1.append(gdf_mergef)
                gdf_merge = gdf_merge.reset_index(drop=True)
            
                # Guardar el GeoDataFrame combinado en un archivo
                gdf_merge.to_file(ruta510)
            
            except Exception as e:
                print(f"Error procesando los datos: {e}")
                pass
            
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
                
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['FID', 'area'], axis=1)
            #points7.to_file(ruta520)
            
            if not points7.empty:
                points7.to_file(ruta521)
            else:
                pass
        else:
            pass
    
    funcion_principal_df11()
    ##################################
    def funcion_principal_df22():
        global df22
        df11a = df22
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            
            from shapely.geometry import shape, Point        
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                # Calcula el área de cada polígono de la intersección
                g2['area'] = g2['geometry'].apply(lambda x: x.area)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'Nucleado', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['people'].fillna(0, inplace=True)
                inter['area'].fillna(0, inplace=True)
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['density']
                inter['risk_pop'].fillna(0, inplace=True)
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            ruta22 = ruta20.removesuffix('.shp')
            ruta4 = ruta20
            
            poligonos = gpd.read_file(ruta20)
        
            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union
        
            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        
            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})
        
            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')
        
            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])
        
            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})
        
            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))
        
            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta20)
            shp2 = gpd.read_file(ruta405)
            #shp3 = gpd.read_file(ruta10)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([shp1, shp2], ignore_index=True), crs=shp1.crs)
            fusion['people'] = fusion['people'].fillna(0)
            fusion['Nucleado'] = fusion['Nucleado'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            fusion = fusion1.drop(['Id', 'H'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
            
            # dens = gpd.read_file(ruta4)
            
            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit House']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit House'][0]/1
            pr1 = df503['Probit House'][1]/1
            pr2 = df503['Probit House'][2]/1
            pr3 = df503['Probit House'][3]/1
            pr4 = df503['Probit House'][4]/1
            pr5 = df503['Probit House'][5]/1
            pr6 = df503['Probit House'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
            gdf_mergef = gdf_merge.loc[gdf_merge['Nucleado'] == 1]
            sumaareas = (gdf_mergef['areap'].sum())/10000
            sumapersonas = gdf_mergef['people'].sum()
            densitotal = sumapersonas / sumaareas
                
            # gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
            # gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
            # gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
            # gdf_mergedis1['people'] = ((gdf_mergedis1['areap']/100) * gdf_mergedis1['density'])/5 #average number of people
            # gdf_mergedis1['people'].fillna(0, inplace=True)
            # gdf_mergedis1['density'].fillna(0, inplace=True)
            # gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
            # sumapersonasdis = (gdf_mergedis1['people'].sum())/5 #average number of people
            # gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
            
            # gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap']/100))/(sumaareas * densitotal)
               
            # gdf_merge = gdf_mergedis1.append (gdf_mergef)
            # gdf_merge = gdf_merge.reset_index(drop=True)
            # gdf_merge['areap'].fillna(0, inplace=True)
            # gdf_mergef['risk_pop'].fillna(0, inplace=True)
            # gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            # gdf_merge.to_file(ruta510)
            try:
                gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
                gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
                
                # Asegurarse de que las columnas 'areap' y 'density' no generen errores
                if 'geometry' in gdf_mergedis1.columns:
                    gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
                else:
                    gdf_mergedis1['areap'] = 0
            
                # Rellenar NaNs en la columna 'density'
                gdf_mergedis1['density'].fillna(0, inplace=True)
            
                # Calcular el número de personas solo si 'density' no está vacío
                if not gdf_mergedis1['density'].empty:
                    gdf_mergedis1['people'] = ((gdf_mergedis1['areap'] / 100) * gdf_mergedis1['density']) / 5
                else:
                    gdf_mergedis1['people'] = 0
            
                # Rellenar NaNs en la columna 'people'
                gdf_mergedis1['people'].fillna(0, inplace=True)
            
                # Calcular 'risk_pop' solo si 'density' no está vacío
                if not gdf_mergedis1['density'].empty:
                    gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
                else:
                    gdf_mergedis1['risk_pop'] = 0
            
                # Rellenar NaNs en la columna 'risk_pop'
                gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            
                # Calcular la suma de personas solo si 'people' no está vacío
                if not gdf_mergedis1['people'].empty:
                    sumapersonasdis = (gdf_mergedis1['people'].sum()) / 5
                else:
                    sumapersonasdis = 1  # Evitar división por cero
            
                # Calcular 'risk_pop' nuevamente
                gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
            
                # Rellenar NaNs en las columnas 'risk_pop' y 'areap'
                gdf_mergef['risk_pop'].fillna(0, inplace=True)
                gdf_merge['areap'].fillna(0, inplace=True)
            
                # Calcular 'risk_pop' para gdf_mergef
                if 'areap' in gdf_mergef.columns and 'risk_pop' in gdf_mergef.columns:
                    gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap'] / 100)) / (sumaareas * densitotal)
            
                # Rellenar NaNs en las columnas 'risk_pop' de gdf_mergedis1
                gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            
                # Combinar los GeoDataFrames y restablecer el índice
                gdf_merge = gdf_mergedis1.append(gdf_mergef)
                gdf_merge = gdf_merge.reset_index(drop=True)
            
                # Guardar el GeoDataFrame combinado en un archivo
                gdf_merge.to_file(ruta510)
            
            except Exception as e:
                print(f"Error procesando los datos: {e}")
                pass

            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
                
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['FID', 'area'], axis=1)
            #points7.to_file(ruta520)
            
            if not points7.empty:
                points7.to_file(ruta522)
            else:
                pass
        else:
            pass
    
    funcion_principal_df22()
    ###################################
    def funcion_principal_df33():
        global df33
        df11a = df33
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            
            from shapely.geometry import shape, Point        
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                # Calcula el área de cada polígono de la intersección
                g2['area'] = g2['geometry'].apply(lambda x: x.area)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'Nucleado', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['people'].fillna(0, inplace=True)
                inter['area'].fillna(0, inplace=True)
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['density']
                inter['risk_pop'].fillna(0, inplace=True)
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            ruta22 = ruta20.removesuffix('.shp')
            ruta4 = ruta20
            
            poligonos = gpd.read_file(ruta20)
        
            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union
        
            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        
            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})
        
            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')
        
            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])
        
            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})
        
            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))
        
            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta20)
            shp2 = gpd.read_file(ruta405)
            #shp3 = gpd.read_file(ruta10)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([shp1, shp2], ignore_index=True), crs=shp1.crs)
            fusion['people'] = fusion['people'].fillna(0)
            fusion['Nucleado'] = fusion['Nucleado'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            fusion = fusion1.drop(['Id', 'H'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
            
            # dens = gpd.read_file(ruta4)
            
            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit House']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit House'][0]/1
            pr1 = df503['Probit House'][1]/1
            pr2 = df503['Probit House'][2]/1
            pr3 = df503['Probit House'][3]/1
            pr4 = df503['Probit House'][4]/1
            pr5 = df503['Probit House'][5]/1
            pr6 = df503['Probit House'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
            gdf_mergef = gdf_merge.loc[gdf_merge['Nucleado'] == 1]
            sumaareas = (gdf_mergef['areap'].sum())/10000
            sumapersonas = gdf_mergef['people'].sum()
            densitotal = sumapersonas / sumaareas
             
            try:
                gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
                gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
                
                # Asegurarse de que las columnas 'areap' y 'density' no generen errores
                if 'geometry' in gdf_mergedis1.columns:
                    gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
                else:
                    gdf_mergedis1['areap'] = 0
            
                # Rellenar NaNs en la columna 'density'
                gdf_mergedis1['density'].fillna(0, inplace=True)
            
                # Calcular el número de personas solo si 'density' no está vacío
                if not gdf_mergedis1['density'].empty:
                    gdf_mergedis1['people'] = ((gdf_mergedis1['areap'] / 100) * gdf_mergedis1['density']) / 5
                else:
                    gdf_mergedis1['people'] = 0
            
                # Rellenar NaNs en la columna 'people'
                gdf_mergedis1['people'].fillna(0, inplace=True)
            
                # Calcular 'risk_pop' solo si 'density' no está vacío
                if not gdf_mergedis1['density'].empty:
                    gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
                else:
                    gdf_mergedis1['risk_pop'] = 0
            
                # Rellenar NaNs en la columna 'risk_pop'
                gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            
                # Calcular la suma de personas solo si 'people' no está vacío
                if not gdf_mergedis1['people'].empty:
                    sumapersonasdis = (gdf_mergedis1['people'].sum()) / 5
                else:
                    sumapersonasdis = 1  # Evitar división por cero
            
                # Calcular 'risk_pop' nuevamente
                gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
            
                # Rellenar NaNs en las columnas 'risk_pop' y 'areap'
                gdf_mergef['risk_pop'].fillna(0, inplace=True)
                gdf_merge['areap'].fillna(0, inplace=True)
            
                # Calcular 'risk_pop' para gdf_mergef
                if 'areap' in gdf_mergef.columns and 'risk_pop' in gdf_mergef.columns:
                    gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap'] / 100)) / (sumaareas * densitotal)
            
                # Rellenar NaNs en las columnas 'risk_pop' de gdf_mergedis1
                gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            
                # Combinar los GeoDataFrames y restablecer el índice
                gdf_merge = gdf_mergedis1.append(gdf_mergef)
                gdf_merge = gdf_merge.reset_index(drop=True)
            
                # Guardar el GeoDataFrame combinado en un archivo
                gdf_merge.to_file(ruta510)
            
            except Exception as e:
                print(f"Error procesando los datos: {e}")
                pass
            # gdf_mergedis = gdf_merge.loc[gdf_merge['Nucleado'] == 0]
            # gdf_mergedis1 = gdf_mergedis.explode("geometry").reset_index(drop=True)
            # gdf_mergedis1['areap'] = gdf_mergedis1['geometry'].apply(lambda x: x.area)
            # gdf_mergedis1['people'] = ((gdf_mergedis1['areap']/100) * gdf_mergedis1['density'])/5 #average number of people
            # gdf_mergedis1['people'].fillna(0, inplace=True)
            # gdf_mergedis1['density'].fillna(0, inplace=True)
            # gdf_mergedis1['risk_pop'] = gdf_mergedis1['risk_pop'] / gdf_mergedis1['density']
            # sumapersonasdis = (gdf_mergedis1['people'].sum())/5 #average number of people
            # gdf_mergedis1['risk_pop'] = (gdf_mergedis1['people'] * gdf_mergedis1['risk_pop']) / sumapersonasdis
            
            # gdf_mergef['risk_pop'] = (gdf_mergef['risk_pop'] * (gdf_mergef['areap']/100))/(sumaareas * densitotal)
               
            # gdf_merge = gdf_mergedis1.append (gdf_mergef)
            # gdf_merge = gdf_merge.reset_index(drop=True)
            # gdf_merge['areap'].fillna(0, inplace=True)
            # gdf_mergef['risk_pop'].fillna(0, inplace=True)
            # gdf_mergedis1['risk_pop'].fillna(0, inplace=True)
            # gdf_merge.to_file(ruta510)
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
                
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['FID', 'area'], axis=1)
            #points7.to_file(ruta520)
            
            if not points7.empty:
                points7.to_file(ruta523)
            else:
                pass
        else:
            pass
    
    funcion_principal_df33()
    
    ###################################
    rutas = [ruta520, ruta521, ruta522, ruta523]
    visualizar_indices_hogares2(ruta120, rutas, ventana, ruta547)

######################################################################################
def visualizar_indices_servicios_publicos3(ruta120, rutas, ventana, ruta546):
    try:
        from matplotlib.colors import ListedColormap
        import numpy as np
        import matplotlib.pyplot as plt       
        # Cargar la imagen TIFF usando GDAL
        # Cargar y mostrar datos TIFF como imagen de fondo usando GDAL
        ds = gdal.Open(ruta120)
        if ds is None:
            print(f"No se pudo abrir el archivo TIFF {ruta120}")
            return
        
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]

        fig, ax = plt.subplots()
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
        
        gdfs = []
        for ruta in rutas:
            try:
                gdf = gpd.read_file(ruta)
                gdfs.append(gdf)
            except Exception as e:
                print(f"No se pudo cargar el archivo {ruta}: {e}")

        if gdfs:
            points_gdf = pd.concat(gdfs, ignore_index=True)
        else:
            points_gdf = pd.DataFrame()

        points_gdf = points_gdf.dropna(subset=['geometry'])
        min_value = points_gdf['risk_pop'].dropna().min() * 0.001
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_pop'] = min_value

        x = points_gdf.geometry.x
        y = points_gdf.geometry.y
        z = points_gdf.risk_pop
        # # Ajustando valores mínimos para 'risk'
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
        
                
        # Asumiendo que x, y, z_filled, y ax ya están definidos correctamente
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)  # Ajusta según sea necesario
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Public Services Set')
        
                
        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
        
        
        if not points_gdf.empty:
            points_gdf.to_file(ruta546, driver='ESRI Shapefile')
            table = points_gdf[points_gdf.risk_pop != 0]
            table = table[table.Este != 0]
            table = table.dropna(subset=['Este'])
            table2 = table[['people', 'risk_pop', 'Este', 'Norte']]
            root5 = table2
            
            Label(text="Table Public Services Index             ", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
            pt.show()
        ventana.mainloop()

    except Exception as e:
        messagebox.showinfo(f"No existe afectacion a Bienes Publicos")
        pass

def proc_socioeconomicpublic():
    ruta2 = ruta.removesuffix('.shp') #Tanks
    ruta400 = ruta2 + "1.shp"
    ruta3 = ruta60.removesuffix('.shp') #public
    ruta401 = ruta3 + "1.shp"
    ruta4 = ruta90.removesuffix('.shp') #public line
    ruta402 = ruta4 + "1.shp"
    ruta405 = ruta2 + "11.shp"
    ruta500 = ruta2 + "100.shp"
    ruta510 = ruta4 + "infrapoint.shp" 
    ruta520 = ruta4 + "lines.shp"
    ruta530 = ruta4 + "publicj.shp"
    ruta540 = ruta4 + "salidapub.shp"
    ruta541 = ruta4 + "salidapub1.shp"
    ruta542 = ruta4 + "salidapub2.shp"
    ruta543 = ruta4 + "salidapub3.shp"
    ruta546 = ruta4 + "salidapubindex.shp"
    
    import os

    def borrar_shapefiles(*rutas):
        for ruta in rutas:
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
                    
                else:
                    pass
            except Exception as e:
                continue
    
    rutas_shapefiles = [
        ruta510, ruta520, ruta530, ruta540, ruta541, ruta542, ruta543
    ]
    
    borrar_shapefiles(*rutas_shapefiles)
    
    def funcion_principal_df00():
        global df00
        df11a = df00
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Longitud', 'Latitud'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['people']
                # count = inter['people'].value_counts()
                # inter['risk_pop'] = count #inter['people'].value_counts()
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            ruta22 = ruta60.removesuffix('.shp')
            ruta4 = ruta60
            
            
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
            ushp7 = gpd.read_file(ruta60)
            
            
                
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6, ushp7], ignore_index=True), crs=ushp.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            #fusion = fusion1.drop(['Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
        
            
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit INps']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit INps'][0]/1
            pr1 = df503['Probit INps'][1]/1
            pr2 = df503['Probit INps'][2]/1
            pr3 = df503['Probit INps'][3]/1
            pr4 = df503['Probit INps'][4]/1
            pr5 = df503['Probit INps'][5]/1
            pr6 = df503['Probit INps'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
        
            
            gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
        
            gdf_merge = pd.concat(gdfs).reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            
            count = gdf_merge['people'].count()
            
            gdf_mergep_list = []
            for gdfp in gdfs:
                gdfp = gdfp[gdfp['risk_pop'] != 0]
                countgdfp = gdfp['people'].count()
                gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                gdf_mergep_list.append(gdfp)
            
            gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
            gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
          
            gdf_mergep.to_file(ruta510)
           
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_mergep, df6], axis=1)
            #end process to building
            
            
            def intersect_and_save1(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                #inter['length'] = inter['geometry'].apply(lambda x: x.length)
                inter['risk_pop'] = 1
                # Reorganiza las columnas del GeoDataFrame
                # total_people = inter['length'].sum()
                # inter['total'] = total_people
                # inter['risk_pop'] = inter['total'] / 2
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
                
            #Star process to households
            ruta22 = ruta90.removesuffix('.shp')
            ruta4 = ruta90
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
               
            
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6], ignore_index=True), crs=ushp.crs)
            lines = fusion.boundary
            line = pd.DataFrame({'geometry': lines})
            
            
            ushp7 = gpd.read_file(ruta90)
            fusion = gpd.GeoDataFrame(pd.concat([ushp7,line], ignore_index=True), crs=ushp7.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            buffer = fusion1.buffer(0.1)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=lines.crs)
            fusion2 = fusion1['people']
            gdf_concat = pd.concat([fusion2, buffer_gdf], axis=1)
            gdf_concat2 = gpd.GeoDataFrame(gdf_concat, geometry='geometry')
            # Save the entire layer as a new shapefile
            gdf_concat2.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
                
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save1(shp, ruta4, ruta70)
            intersect_and_save1(shp1, ruta4, ruta71)
            intersect_and_save1(shp2, ruta4, ruta72)
            intersect_and_save1(shp3, ruta4, ruta73)
            intersect_and_save1(shp4, ruta4, ruta74)
            intersect_and_save1(shp5, ruta4, ruta75)
            intersect_and_save1(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit LNps']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit LNps'][0]/1
            pr1 = df503['Probit LNps'][1]/1
            pr2 = df503['Probit LNps'][2]/1
            pr3 = df503['Probit LNps'][3]/1
            pr4 = df503['Probit LNps'][4]/1
            pr5 = df503['Probit LNps'][5]/1
            pr6 = df503['Probit LNps'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
        
            gdf_merge = pd.concat(gdfs).reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            
            count = gdf_merge['people'].count()
            
            gdf_mergep_list = []
            for gdfp in gdfs:
                gdfp = gdfp[gdfp['risk_pop'] != 0]
                countgdfp = gdfp['people'].count()
                gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                gdf_mergep_list.append(gdfp)
            
            gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
            gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
                
            gdf_mergep.to_file(ruta520) #export lines shapefile
            
            
            shp200 = gpd.read_file(ruta510)
            shp300 = gpd.read_file(ruta520)
            fusion3 = gpd.GeoDataFrame(pd.concat([shp200, shp300], ignore_index=True), crs=shp200.crs)
            fusion4 = fusion3.explode("geometry").reset_index(drop=True)
            fusion4.to_file(ruta530)
            centroids = [] #Empy
               
            with fiona.open(ruta530) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado2 = pd.concat([fusion4, df6], axis=1)
            points = df_concatenado2.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.people != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            # min_value = points6['risk_pop'].min()
            # points7 = points6[points6.risk_pop != min_value]
            points7 = points6.drop(['FID', 'area', 'index', 'density'], axis=1)
            #points7.to_file(ruta540)
            if not points7.empty:
                points7.to_file(ruta540)
            else:
                pass
        else:
            pass
    funcion_principal_df00()
            
    ###############################
    def funcion_principal_df11():
        global df11
        df11a = df11
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Longitud', 'Latitud'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['people']
                
                inter.to_file(output_shp)
            
            # Define la ruta base
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            ruta22 = ruta60.removesuffix('.shp')
            ruta4 = ruta60
            
            
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
            ushp7 = gpd.read_file(ruta60)
            
            
                
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6, ushp7], ignore_index=True), crs=ushp.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            #fusion = fusion1.drop(['Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
        
            
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit INps']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit INps'][0]/1
            pr1 = df503['Probit INps'][1]/1
            pr2 = df503['Probit INps'][2]/1
            pr3 = df503['Probit INps'][3]/1
            pr4 = df503['Probit INps'][4]/1
            pr5 = df503['Probit INps'][5]/1
            pr6 = df503['Probit INps'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
        
            
            gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
        
            gdf_merge = pd.concat(gdfs).reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            
            count = gdf_merge['people'].count()
            
            gdf_mergep_list = []
            for gdfp in gdfs:
                gdfp = gdfp[gdfp['risk_pop'] != 0]
                countgdfp = gdfp['people'].count()
                gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                gdf_mergep_list.append(gdfp)
            
            gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
            gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
          
            gdf_mergep.to_file(ruta510)
           
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_mergep, df6], axis=1)
            #end process to building
            
            
            def intersect_and_save1(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                #inter['length'] = inter['geometry'].apply(lambda x: x.length)
                inter['risk_pop'] = 1
                # Reorganiza las columnas del GeoDataFrame
                # total_people = inter['length'].sum()
                # inter['total'] = total_people
                # inter['risk_pop'] = inter['total'] / 2
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
                
            #Star process to households
            ruta22 = ruta90.removesuffix('.shp')
            ruta4 = ruta90
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
               
            #shp2 = gpd.read_file(ruta405)
            #shp3 = gpd.read_file(ruta10)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6], ignore_index=True), crs=ushp.crs)
            lines = fusion.boundary
            line = pd.DataFrame({'geometry': lines})
            
            
            ushp7 = gpd.read_file(ruta90)
            fusion = gpd.GeoDataFrame(pd.concat([ushp7,line], ignore_index=True), crs=ushp7.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            buffer = fusion1.buffer(0.1)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=lines.crs)
            fusion2 = fusion1['people']
            gdf_concat = pd.concat([fusion2, buffer_gdf], axis=1)
            gdf_concat2 = gpd.GeoDataFrame(gdf_concat, geometry='geometry')
            # Save the entire layer as a new shapefile
            gdf_concat2.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
                
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save1(shp, ruta4, ruta70)
            intersect_and_save1(shp1, ruta4, ruta71)
            intersect_and_save1(shp2, ruta4, ruta72)
            intersect_and_save1(shp3, ruta4, ruta73)
            intersect_and_save1(shp4, ruta4, ruta74)
            intersect_and_save1(shp5, ruta4, ruta75)
            intersect_and_save1(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit LNps']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit LNps'][0]/1
            pr1 = df503['Probit LNps'][1]/1
            pr2 = df503['Probit LNps'][2]/1
            pr3 = df503['Probit LNps'][3]/1
            pr4 = df503['Probit LNps'][4]/1
            pr5 = df503['Probit LNps'][5]/1
            pr6 = df503['Probit LNps'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
        
            gdf_merge = pd.concat(gdfs).reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            
            count = gdf_merge['people'].count()
            
            gdf_mergep_list = []
            for gdfp in gdfs:
                gdfp = gdfp[gdfp['risk_pop'] != 0]
                countgdfp = gdfp['people'].count()
                gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                gdf_mergep_list.append(gdfp)
            
            gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
            gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
                
            gdf_mergep.to_file(ruta520) #export lines shapefile
            
            
            shp200 = gpd.read_file(ruta510)
            shp300 = gpd.read_file(ruta520)
            fusion3 = gpd.GeoDataFrame(pd.concat([shp200, shp300], ignore_index=True), crs=shp200.crs)
            fusion4 = fusion3.explode("geometry").reset_index(drop=True)
            fusion4.to_file(ruta530)
            centroids = [] #Empy
               
            with fiona.open(ruta530) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado2 = pd.concat([fusion4, df6], axis=1)
            points = df_concatenado2.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.people != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            # min_value = points6['risk_pop'].min()
            # points7 = points6[points6.risk_pop != min_value]
            points7 = points6.drop(['FID', 'area', 'index', 'density'], axis=1)
            #points7.to_file(ruta540)
            if not points7.empty:
                points7.to_file(ruta541)
            else:
                pass
        else:
            pass
    funcion_principal_df11()
    
    ###############################
    def funcion_principal_df22():
        global df22
        df11a = df22
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Longitud', 'Latitud'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['people']
                # count = inter['people'].value_counts()
                # inter['risk_pop'] = count #inter['people'].value_counts()
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            ruta22 = ruta60.removesuffix('.shp')
            ruta4 = ruta60
            
            
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
            ushp7 = gpd.read_file(ruta60)
            
            
                
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6, ushp7], ignore_index=True), crs=ushp.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            #fusion = fusion1.drop(['Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
        
            
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit INps']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit INps'][0]/1
            pr1 = df503['Probit INps'][1]/1
            pr2 = df503['Probit INps'][2]/1
            pr3 = df503['Probit INps'][3]/1
            pr4 = df503['Probit INps'][4]/1
            pr5 = df503['Probit INps'][5]/1
            pr6 = df503['Probit INps'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
        
            
            gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
        
            gdf_merge = pd.concat(gdfs).reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            
            count = gdf_merge['people'].count()
            
            gdf_mergep_list = []
            for gdfp in gdfs:
                gdfp = gdfp[gdfp['risk_pop'] != 0]
                countgdfp = gdfp['people'].count()
                gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                gdf_mergep_list.append(gdfp)
            
            gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
            gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
          
            gdf_mergep.to_file(ruta510)
           
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_mergep, df6], axis=1)
            #end process to building
            
            
            def intersect_and_save1(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                #inter['length'] = inter['geometry'].apply(lambda x: x.length)
                inter['risk_pop'] = 1
                
                inter.to_file(output_shp)
                
            #Star process to households
            ruta22 = ruta90.removesuffix('.shp')
            ruta4 = ruta90
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
               
            #shp2 = gpd.read_file(ruta405)
            #shp3 = gpd.read_file(ruta10)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6], ignore_index=True), crs=ushp.crs)
            lines = fusion.boundary
            line = pd.DataFrame({'geometry': lines})
            
            
            ushp7 = gpd.read_file(ruta90)
            fusion = gpd.GeoDataFrame(pd.concat([ushp7,line], ignore_index=True), crs=ushp7.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            buffer = fusion1.buffer(0.1)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=lines.crs)
            fusion2 = fusion1['people']
            gdf_concat = pd.concat([fusion2, buffer_gdf], axis=1)
            gdf_concat2 = gpd.GeoDataFrame(gdf_concat, geometry='geometry')
            # Save the entire layer as a new shapefile
            gdf_concat2.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
                
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save1(shp, ruta4, ruta70)
            intersect_and_save1(shp1, ruta4, ruta71)
            intersect_and_save1(shp2, ruta4, ruta72)
            intersect_and_save1(shp3, ruta4, ruta73)
            intersect_and_save1(shp4, ruta4, ruta74)
            intersect_and_save1(shp5, ruta4, ruta75)
            intersect_and_save1(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit LNps']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit LNps'][0]/1
            pr1 = df503['Probit LNps'][1]/1
            pr2 = df503['Probit LNps'][2]/1
            pr3 = df503['Probit LNps'][3]/1
            pr4 = df503['Probit LNps'][4]/1
            pr5 = df503['Probit LNps'][5]/1
            pr6 = df503['Probit LNps'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
        
            gdf_merge = pd.concat(gdfs).reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            
            count = gdf_merge['people'].count()
            
            gdf_mergep_list = []
            for gdfp in gdfs:
                gdfp = gdfp[gdfp['risk_pop'] != 0]
                countgdfp = gdfp['people'].count()
                gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                gdf_mergep_list.append(gdfp)
            
            gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
            gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
                
            gdf_mergep.to_file(ruta520) #export lines shapefile
            
            
            shp200 = gpd.read_file(ruta510)
            shp300 = gpd.read_file(ruta520)
            fusion3 = gpd.GeoDataFrame(pd.concat([shp200, shp300], ignore_index=True), crs=shp200.crs)
            fusion4 = fusion3.explode("geometry").reset_index(drop=True)
            fusion4.to_file(ruta530)
            centroids = [] #Empy
               
            with fiona.open(ruta530) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado2 = pd.concat([fusion4, df6], axis=1)
            points = df_concatenado2.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.people != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            # min_value = points6['risk_pop'].min()
            # points7 = points6[points6.risk_pop != min_value]
            points7 = points6.drop(['FID', 'area', 'index', 'density'], axis=1)
            #points7.to_file(ruta540)
            if not points7.empty:
                points7.to_file(ruta542)
            else:
                pass
        else:
            pass
    funcion_principal_df22()
    #####################################
    def funcion_principal_df33():
        global df33
        df11a = df33
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Longitud', 'Latitud'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['people']
                # count = inter['people'].value_counts()
                # inter['risk_pop'] = count #inter['people'].value_counts()
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            ruta22 = ruta60.removesuffix('.shp')
            ruta4 = ruta60
            
            
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
            ushp7 = gpd.read_file(ruta60)
            
            
                
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6, ushp7], ignore_index=True), crs=ushp.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            #fusion = fusion1.drop(['Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
        
            
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit INps']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit INps'][0]/1
            pr1 = df503['Probit INps'][1]/1
            pr2 = df503['Probit INps'][2]/1
            pr3 = df503['Probit INps'][3]/1
            pr4 = df503['Probit INps'][4]/1
            pr5 = df503['Probit INps'][5]/1
            pr6 = df503['Probit INps'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
        
            
            gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
        
            gdf_merge = pd.concat(gdfs).reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            
            count = gdf_merge['people'].count()
            
            gdf_mergep_list = []
            for gdfp in gdfs:
                gdfp = gdfp[gdfp['risk_pop'] != 0]
                countgdfp = gdfp['people'].count()
                gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                gdf_mergep_list.append(gdfp)
            
            gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
            gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
          
            gdf_mergep.to_file(ruta510)
           
            centroids = [] #Empy
               
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_mergep, df6], axis=1)
            #end process to building
            
            
            def intersect_and_save1(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                #inter['length'] = inter['geometry'].apply(lambda x: x.length)
                inter['risk_pop'] = 1
                inter.to_file(output_shp)
                
            #Star process to households
            ruta22 = ruta90.removesuffix('.shp')
            ruta4 = ruta90
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
               
            #shp2 = gpd.read_file(ruta405)
            #shp3 = gpd.read_file(ruta10)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6], ignore_index=True), crs=ushp.crs)
            lines = fusion.boundary
            line = pd.DataFrame({'geometry': lines})
            
            
            ushp7 = gpd.read_file(ruta90)
            fusion = gpd.GeoDataFrame(pd.concat([ushp7,line], ignore_index=True), crs=ushp7.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            buffer = fusion1.buffer(0.1)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=lines.crs)
            fusion2 = fusion1['people']
            gdf_concat = pd.concat([fusion2, buffer_gdf], axis=1)
            gdf_concat2 = gpd.GeoDataFrame(gdf_concat, geometry='geometry')
            # Save the entire layer as a new shapefile
            gdf_concat2.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
                
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save1(shp, ruta4, ruta70)
            intersect_and_save1(shp1, ruta4, ruta71)
            intersect_and_save1(shp2, ruta4, ruta72)
            intersect_and_save1(shp3, ruta4, ruta73)
            intersect_and_save1(shp4, ruta4, ruta74)
            intersect_and_save1(shp5, ruta4, ruta75)
            intersect_and_save1(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit LNps']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit LNps'][0]/1
            pr1 = df503['Probit LNps'][1]/1
            pr2 = df503['Probit LNps'][2]/1
            pr3 = df503['Probit LNps'][3]/1
            pr4 = df503['Probit LNps'][4]/1
            pr5 = df503['Probit LNps'][5]/1
            pr6 = df503['Probit LNps'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
        
            gdf_merge = pd.concat(gdfs).reset_index(drop=True)
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            
            count = gdf_merge['people'].count()
            
            gdf_mergep_list = []
            for gdfp in gdfs:
                gdfp = gdfp[gdfp['risk_pop'] != 0]
                countgdfp = gdfp['people'].count()
                gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                gdf_mergep_list.append(gdfp)
            
            gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
            gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
                
            gdf_mergep.to_file(ruta520) #export lines shapefile
            
            
            shp200 = gpd.read_file(ruta510)
            shp300 = gpd.read_file(ruta520)
            fusion3 = gpd.GeoDataFrame(pd.concat([shp200, shp300], ignore_index=True), crs=shp200.crs)
            fusion4 = fusion3.explode("geometry").reset_index(drop=True)
            fusion4.to_file(ruta530)
            centroids = [] #Empy
               
            with fiona.open(ruta530) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado2 = pd.concat([fusion4, df6], axis=1)
            points = df_concatenado2.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.people != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            # min_value = points6['risk_pop'].min()
            # points7 = points6[points6.risk_pop != min_value]
            points7 = points6.drop(['FID', 'area', 'index', 'density'], axis=1)
            #points7.to_file(ruta540)
            if not points7.empty:
                points7.to_file(ruta543)
            else:
                pass
        else:
            pass
    funcion_principal_df33()
       
    
    #####################################
    rutas = [ruta540, ruta541, ruta542, ruta543]
    visualizar_indices_servicios_publicos3(ruta120, rutas, ventana, ruta546)

###########################################################################
def visualizar_interpolacion_y_tablasocial(rutas_shapefiles, ruta120, ventana, ruta545):
    try:
        from matplotlib.colors import ListedColormap
        import numpy as np
        import matplotlib.pyplot as plt
        # Cargar la imagen TIFF usando GDAL
        ds = gdal.Open(ruta120)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]

        # Crear figura y ejes para la trama
        fig, ax = plt.subplots()

        # Mostrar los datos TIFF en el eje con su sistema de referencia de coordenadas original
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
        
        gdfs = []
        for ruta in rutas_shapefiles:
            try:
                gdf = gpd.read_file(ruta)
                gdfs.append(gdf)
            except Exception as e:
                print(f"Error al leer {ruta}: {e}")
        
        if not gdfs:
            print("No se encontraron archivos válidos.")
            return
        
        points_gdf = pd.concat(gdfs, ignore_index=True)
        points_gdf = points_gdf.dropna(subset=['geometry'])

        if points_gdf.empty:
            print("No hay datos para visualizar.")
            return
        
        min_value = points_gdf['risk_pop'].dropna().min() * 0.001
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_pop'] = min_value
        
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
        # # Ajustando valores mínimos para 'risk'
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
        
        # Creación de la figura y el eje para la trama
        fig, ax = plt.subplots()

        # Mostrar los datos TIFF en el eje con su sistema de referencia de coordenadas original
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
        
        
        
        # Asumiendo que x, y, z_filled, y ax ya están definidos correctamente
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)  # Ajusta según sea necesario
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Social Services Set')

        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)

        if not points_gdf.empty:
            points_gdf.to_file(ruta545, driver='ESRI Shapefile')
            table = points_gdf[points_gdf.risk_pop != 0]
            table = table[table.Este != 0]
            table = table.dropna(subset=['Nucleado'])
            table2 = table[['people', 'Nucleado', 'risk_pop', 'Este', 'Norte']]
            root5 = table2
            
            Label(text="Table Risk on the Social Services Set             ", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
            pt.show()

    except Exception as e:
        pass
    

def proc_socioeconomicsocial():
    ruta2 = ruta.removesuffix('.shp') #Tanks
    ruta400 = ruta2 + "1.shp"
    ruta3 = ruta110.removesuffix('.shp') #social
    ruta401 = ruta3 + "1.shp"
    ruta4 = ruta100.removesuffix('.shp') #social line
    ruta402 = ruta4 + "1.shp"
    ruta405 = ruta2 + "11.shp"
    ruta500 = ruta2 + "100.shp"
    ruta510 = ruta4 + "infrapoint.shp" 
    ruta520 = ruta4 + "lines.shp"
    ruta530 = ruta4 + "publicj.shp"
    ruta540 = ruta4 + "salidasocc.shp"
    ruta541 = ruta4 + "salidasocc1.shp"
    ruta542 = ruta4 + "salidasocc2.shp"
    ruta543 = ruta4 + "salidasocc3.shp"
    ruta545 = ruta4 + "salidasoccindex.shp"
    
    import os

    def borrar_shapefiles(*rutas):
        for ruta in rutas:
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
                    
                else:
                    pass
            except Exception as e:
                continue
    
    rutas_shapefiles = [
        ruta510, ruta520, ruta530, ruta540, ruta541, ruta542, ruta543
    ]
    
    borrar_shapefiles(*rutas_shapefiles)
    def funcion_principal_df00():
        global df00
        df11a = df00
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd

            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            gdf.to_file(ruta500)
            df = df11a
            # if df['Impact Radius'].sum() == 0:
            #     # Establece el valor de la primera fila de 'Impact Radius' a 1
            #     df.loc[0, 'Impact Radius'] = 1
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
                        
            df499 = (df498['Impact Radius']).div(1)
            df499.fillna(0, inplace=True)
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
            centroids = calcular_centroides_de_shapefile(ruta500) #Funcion centroides  
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Longitud', 'Latitud'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
                
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['people']
                # count = inter['people'].value_counts()
                # inter['risk_pop'] = count #inter['people'].value_counts()
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            ruta22 = ruta110.removesuffix('.shp')
            ruta4 = ruta110
            
            
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
            ushp7 = gpd.read_file(ruta110)
            
            
                
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6, ushp7], ignore_index=True), crs=ushp.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            #fusion = fusion1.drop(['Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
        
            
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit INss']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit INss'][0]/1
            pr1 = df503['Probit INss'][1]/1
            pr2 = df503['Probit INss'][2]/1
            pr3 = df503['Probit INss'][3]/1
            pr4 = df503['Probit INss'][4]/1
            pr5 = df503['Probit INss'][5]/1
            pr6 = df503['Probit INss'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
        
            
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            # Verificar si gdf_merge no está vacío antes de proceder
            # Inicialización y procesamiento de gdf_merge.
            if not gdf_merge.empty:
                gdf_merge = gdf_merge.reset_index(drop=True)
                gdf_merge.fillna(0, inplace=True)
                gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
                count = gdf_merge['people'].count() if 'people' in gdf_merge.columns else 0
            else:
                count = 0
            
            # Inicializar gdf_mergep aquí para asegurarse de que siempre esté definido
            gdf_mergep = gpd.GeoDataFrame()
            
            # Procesar y llenar gdf_mergep basado en gdf_merge procesado.
            if count > 0:
                # Asumiendo que la lógica de filtrado y procesamiento para gdf_merge ya ha ocurrido.
                gdfp = gdf_merge[gdf_merge['risk_pop'] != 0].copy()  # Usar .copy() para evitar SettingWithCopyWarning
                countgdfp = gdfp['people'].count() if 'people' in gdfp.columns else 0
                if countgdfp > 0:
                    # Asegurar que no haya división por cero
                    gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                    gdf_mergep_list = [gdfp]  # Crear lista con el DataFrame procesado
                else:
                    # Si gdfp está vacío después del filtrado, gdf_mergep_list se queda como lista vacía
                    gdf_mergep_list = []
            
                if gdf_mergep_list:
                    gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
                    gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
        
        
            
            #gdf_mergep.to_file(ruta510)
            # Verificar si gdf_mergep no está vacío antes de intentar guardarlo en un archivo
            if not gdf_mergep.empty:
                 gdf_mergep.to_file(ruta510)
            else:
                pass
        
            centroids = [] #Empy
               
            try:
                     
                with fiona.open(ruta510) as f:
                    # Itera sobre todas las entradas del shapefile
                    for feature in f:
                        # Obtiene la geometría del polígono
                        polygon = shape(feature['geometry'])
                        # Calcula el centroide del polígono
                        centroid = polygon.centroid
                        # Almacena las coordenadas del centroide en la lista
                        centroids.append((centroid.x, centroid.y))
            except Exception as e:
                pass
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            #end process to building
            
            
            def intersect_and_save1(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                #inter['length'] = inter['geometry'].apply(lambda x: x.length)
                inter['risk_pop'] = 1
                # Reorganiza las columnas del GeoDataFrame
                # total_people = inter['length'].sum()
                # inter['total'] = total_people
                # inter['risk_pop'] = inter['total'] / 2
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
                
            #Star process to households
            ruta22 = ruta100.removesuffix('.shp')
            ruta4 = ruta100
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
               
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6], ignore_index=True), crs=ushp.crs)
            lines = fusion.boundary
            line = pd.DataFrame({'geometry': lines})
            
            
            ushp7 = gpd.read_file(ruta100)
            fusion = gpd.GeoDataFrame(pd.concat([ushp7,line], ignore_index=True), crs=ushp7.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            buffer = fusion1.buffer(0.1)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=lines.crs)
            fusion2 = fusion1['people']
            gdf_concat = pd.concat([fusion2, buffer_gdf], axis=1)
            gdf_concat2 = gpd.GeoDataFrame(gdf_concat, geometry='geometry')
            # Save the entire layer as a new shapefile
            gdf_concat2.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
                
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save1(shp, ruta4, ruta70)
            intersect_and_save1(shp1, ruta4, ruta71)
            intersect_and_save1(shp2, ruta4, ruta72)
            intersect_and_save1(shp3, ruta4, ruta73)
            intersect_and_save1(shp4, ruta4, ruta74)
            intersect_and_save1(shp5, ruta4, ruta75)
            intersect_and_save1(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit LNss']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit LNss'][0]/1
            pr1 = df503['Probit LNss'][1]/1
            pr2 = df503['Probit LNss'][2]/1
            pr3 = df503['Probit LNss'][3]/1
            pr4 = df503['Probit LNss'][4]/1
            pr5 = df503['Probit LNss'][5]/1
            pr6 = df503['Probit LNss'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            # Verificar si gdf_merge no está vacío antes de proceder
            if not gdf_merge.empty:
                gdf_merge = gdf_merge.reset_index(drop=True)
                gdf_merge.fillna(0, inplace=True)
                gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
                count = gdf_merge['people'].count() if 'people' in gdf_merge else 0  # Asegura que la columna 'people' exista
                gdf_mergep_list = []
                
                # Solo proceder si gdf_merge tiene registros después de filtrar por 'risk_pop'
                if not gdf_merge.empty:
                    gdfp = gdf_merge[gdf_merge['risk_pop'] != 0]
                    countgdfp = gdfp['people'].count() if 'people' in gdfp else 0  # Asegura que la columna 'people' exista en gdfp
                    if countgdfp > 0 and count > 0:  # Asegura que no haya división por cero
                        gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                    gdf_mergep_list.append(gdfp)
                    
                    gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
                    gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
            else:
                pass
            
            #gdf_mergep.to_file(ruta510)
            # Verificar si gdf_mergep no está vacío antes de intentar guardarlo en un archivo
            if not gdf_mergep.empty:
                gdf_mergep.to_file(ruta520) #export lines shapefile
            else:
                pass
                
            # Inicializar GeoDataFrames vacíos
            shp200 = gpd.GeoDataFrame()
            shp300 = gpd.GeoDataFrame()
            
            # Intentar cargar shp200 si el archivo existe
            if os.path.exists(ruta510):
                try:
                    shp200 = gpd.read_file(ruta510)
                except Exception as e:
                    pass
            
            # Intentar cargar shp300 si el archivo existe
            if os.path.exists(ruta520):
                try:
                    shp300 = gpd.read_file(ruta520)
                except Exception as e:
                    pass
            
            # Proceder solo si al menos uno de los GeoDataFrames no está vacío
            import geopandas as gpd
            
            fusion4 = pd.DataFrame(columns=['people', 'area', 'density', 'risk_pop'])
            # Convertir el DataFrame a GeoDataFrame sin especificar geometría
            fusion4 = gpd.GeoDataFrame(fusion4)
            fusion4['geometry'] = None
            fusion4 = fusion4.set_geometry('geometry')
            
                
            if not shp200.empty or not shp300.empty:
                # Determinar el sistema de referencia de coordenadas (CRS) a partir de shp200 o shp300, el que esté disponible
                crs = shp200.crs if not shp200.empty else shp300.crs
                
                # Concatenar shp200 y shp300, considerando el CRS obtenido
                fusion3 = gpd.GeoDataFrame(pd.concat([shp200, shp300], ignore_index=True), crs=crs)
                fusion4 = fusion3.explode("geometry").reset_index(drop=True)
                
                # Guardar fusion4 si no está vacío
                if not fusion4.empty:
                    fusion4.to_file(ruta530)
                else:
                    pass
            else:
                pass
        
            centroids = [] # Lista vacía para almacenar centroides, si es necesario en pasos posteriores
        
               
            try:
                with fiona.open(ruta530) as f:
                    for feature in f:
                        polygon = shape(feature['geometry'])
                        centroid = polygon.centroid
                        centroids.append((centroid.x, centroid.y))
                               
            except fiona.errors.DriverError:
                pass
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            # Suponiendo que fusion4 es un DataFrame o GeoDataFrame previamente definido
            df_concatenado2 = pd.concat([fusion4, df6], axis=1)
            points = df_concatenado2.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            if not zone.empty:
                points2 = zone.copy()
                points2['risk_pop'] = points2['people']
                points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
                filas= points2['geometry']
            else:
                pass
            
            points2 = points2[~points2['geometry'].isna() & (points2['geometry'] != '')]
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['area'], axis=1)
            if 'level_0' in points7.columns:
                points7 = points7.drop(columns=['level_0'])
            
            # Cambiar la columna 'index' a tipo object si existe
            if 'index' in points7.columns:
                points7['index'] = points7['index'].astype(object)
            
            # Cambiar todas las otras columnas a float64, excepto 'geometry'
            for column in points7.columns:
                if column != 'geometry' and column != 'index':
                    points7[column] = points7[column].astype(float)
            #points7.to_file(ruta520)
            
            if not points7.empty:
                points7.to_file(ruta540)
            else:
                pass
        else:
            pass
    funcion_principal_df00()
    #################################
                  
    def funcion_principal_df11():
        global df11
        df11a = df11
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd

            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            gdf.to_file(ruta500)
            df = df11a
            # if df['Impact Radius'].sum() == 0:
            #     # Establece el valor de la primera fila de 'Impact Radius' a 1
            #     df.loc[0, 'Impact Radius'] = 1
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
                        
            df499 = (df498['Impact Radius']).div(1)
            df499.fillna(0, inplace=True)
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
            centroids = calcular_centroides_de_shapefile(ruta500) #Funcion centroides  
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Longitud', 'Latitud'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
                
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['people']
                # count = inter['people'].value_counts()
                # inter['risk_pop'] = count #inter['people'].value_counts()
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            ruta22 = ruta110.removesuffix('.shp')
            ruta4 = ruta110
            
            
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
            ushp7 = gpd.read_file(ruta110)
            
            
                
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6, ushp7], ignore_index=True), crs=ushp.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            #fusion = fusion1.drop(['Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
        
            
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit INss']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit INss'][0]/1
            pr1 = df503['Probit INss'][1]/1
            pr2 = df503['Probit INss'][2]/1
            pr3 = df503['Probit INss'][3]/1
            pr4 = df503['Probit INss'][4]/1
            pr5 = df503['Probit INss'][5]/1
            pr6 = df503['Probit INss'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
        
            
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            # Verificar si gdf_merge no está vacío antes de proceder
            # Inicialización y procesamiento de gdf_merge.
            if not gdf_merge.empty:
                gdf_merge = gdf_merge.reset_index(drop=True)
                gdf_merge.fillna(0, inplace=True)
                gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
                count = gdf_merge['people'].count() if 'people' in gdf_merge.columns else 0
            else:
                count = 0
            
            # Inicializar gdf_mergep aquí para asegurarse de que siempre esté definido
            gdf_mergep = gpd.GeoDataFrame()
            
            # Procesar y llenar gdf_mergep basado en gdf_merge procesado.
            if count > 0:
                # Asumiendo que la lógica de filtrado y procesamiento para gdf_merge ya ha ocurrido.
                gdfp = gdf_merge[gdf_merge['risk_pop'] != 0].copy()  # Usar .copy() para evitar SettingWithCopyWarning
                countgdfp = gdfp['people'].count() if 'people' in gdfp.columns else 0
                if countgdfp > 0:
                    # Asegurar que no haya división por cero
                    gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                    gdf_mergep_list = [gdfp]  # Crear lista con el DataFrame procesado
                else:
                    # Si gdfp está vacío después del filtrado, gdf_mergep_list se queda como lista vacía
                    gdf_mergep_list = []
            
                if gdf_mergep_list:
                    gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
                    gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
        
        
            
            #gdf_mergep.to_file(ruta510)
            # Verificar si gdf_mergep no está vacío antes de intentar guardarlo en un archivo
            if not gdf_mergep.empty:
                 gdf_mergep.to_file(ruta510)
            else:
                pass
        
            centroids = [] #Empy
               
            try:
                     
                with fiona.open(ruta510) as f:
                    # Itera sobre todas las entradas del shapefile
                    for feature in f:
                        # Obtiene la geometría del polígono
                        polygon = shape(feature['geometry'])
                        # Calcula el centroide del polígono
                        centroid = polygon.centroid
                        # Almacena las coordenadas del centroide en la lista
                        centroids.append((centroid.x, centroid.y))
            except Exception as e:
                pass
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            #end process to building
            
            
            def intersect_and_save1(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                #inter['length'] = inter['geometry'].apply(lambda x: x.length)
                inter['risk_pop'] = 1
                # Reorganiza las columnas del GeoDataFrame
                # total_people = inter['length'].sum()
                # inter['total'] = total_people
                # inter['risk_pop'] = inter['total'] / 2
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
                
            #Star process to households
            ruta22 = ruta100.removesuffix('.shp')
            ruta4 = ruta100
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
               
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6], ignore_index=True), crs=ushp.crs)
            lines = fusion.boundary
            line = pd.DataFrame({'geometry': lines})
            
            
            ushp7 = gpd.read_file(ruta100)
            fusion = gpd.GeoDataFrame(pd.concat([ushp7,line], ignore_index=True), crs=ushp7.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            buffer = fusion1.buffer(0.1)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=lines.crs)
            fusion2 = fusion1['people']
            gdf_concat = pd.concat([fusion2, buffer_gdf], axis=1)
            gdf_concat2 = gpd.GeoDataFrame(gdf_concat, geometry='geometry')
            # Save the entire layer as a new shapefile
            gdf_concat2.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
                
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save1(shp, ruta4, ruta70)
            intersect_and_save1(shp1, ruta4, ruta71)
            intersect_and_save1(shp2, ruta4, ruta72)
            intersect_and_save1(shp3, ruta4, ruta73)
            intersect_and_save1(shp4, ruta4, ruta74)
            intersect_and_save1(shp5, ruta4, ruta75)
            intersect_and_save1(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit LNss']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit LNss'][0]/1
            pr1 = df503['Probit LNss'][1]/1
            pr2 = df503['Probit LNss'][2]/1
            pr3 = df503['Probit LNss'][3]/1
            pr4 = df503['Probit LNss'][4]/1
            pr5 = df503['Probit LNss'][5]/1
            pr6 = df503['Probit LNss'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            # Verificar si gdf_merge no está vacío antes de proceder
            if not gdf_merge.empty:
                gdf_merge = gdf_merge.reset_index(drop=True)
                gdf_merge.fillna(0, inplace=True)
                gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
                count = gdf_merge['people'].count() if 'people' in gdf_merge else 0  # Asegura que la columna 'people' exista
                gdf_mergep_list = []
                
                # Solo proceder si gdf_merge tiene registros después de filtrar por 'risk_pop'
                if not gdf_merge.empty:
                    gdfp = gdf_merge[gdf_merge['risk_pop'] != 0]
                    countgdfp = gdfp['people'].count() if 'people' in gdfp else 0  # Asegura que la columna 'people' exista en gdfp
                    if countgdfp > 0 and count > 0:  # Asegura que no haya división por cero
                        gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                    gdf_mergep_list.append(gdfp)
                    
                    gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
                    gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
            else:
                pass
            
            #gdf_mergep.to_file(ruta510)
            # Verificar si gdf_mergep no está vacío antes de intentar guardarlo en un archivo
            if not gdf_mergep.empty:
                gdf_mergep.to_file(ruta520) #export lines shapefile
            else:
                pass
                
            # Inicializar GeoDataFrames vacíos
            shp200 = gpd.GeoDataFrame()
            shp300 = gpd.GeoDataFrame()
            
            # Intentar cargar shp200 si el archivo existe
            if os.path.exists(ruta510):
                try:
                    shp200 = gpd.read_file(ruta510)
                except Exception as e:
                    pass
            
            # Intentar cargar shp300 si el archivo existe
            if os.path.exists(ruta520):
                try:
                    shp300 = gpd.read_file(ruta520)
                except Exception as e:
                    pass
            
            # Proceder solo si al menos uno de los GeoDataFrames no está vacío
            import geopandas as gpd
            
            fusion4 = pd.DataFrame(columns=['people', 'area', 'density', 'risk_pop'])
            # Convertir el DataFrame a GeoDataFrame sin especificar geometría
            fusion4 = gpd.GeoDataFrame(fusion4)
            fusion4['geometry'] = None
            fusion4 = fusion4.set_geometry('geometry')
            
                
            if not shp200.empty or not shp300.empty:
                # Determinar el sistema de referencia de coordenadas (CRS) a partir de shp200 o shp300, el que esté disponible
                crs = shp200.crs if not shp200.empty else shp300.crs
                
                # Concatenar shp200 y shp300, considerando el CRS obtenido
                fusion3 = gpd.GeoDataFrame(pd.concat([shp200, shp300], ignore_index=True), crs=crs)
                fusion4 = fusion3.explode("geometry").reset_index(drop=True)
                
                # Guardar fusion4 si no está vacío
                if not fusion4.empty:
                    fusion4.to_file(ruta530)
                else:
                    pass
            else:
                pass
        
            centroids = [] # Lista vacía para almacenar centroides, si es necesario en pasos posteriores
        
               
            try:
                with fiona.open(ruta530) as f:
                    for feature in f:
                        polygon = shape(feature['geometry'])
                        centroid = polygon.centroid
                        centroids.append((centroid.x, centroid.y))
                               
            except fiona.errors.DriverError:
                pass
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            # Suponiendo que fusion4 es un DataFrame o GeoDataFrame previamente definido
            df_concatenado2 = pd.concat([fusion4, df6], axis=1)
            points = df_concatenado2.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            if not zone.empty:
                points2 = zone.copy()
                points2['risk_pop'] = points2['people']
                points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
                filas= points2['geometry']
            else:
                pass
            
            points2 = points2[~points2['geometry'].isna() & (points2['geometry'] != '')]
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['area'], axis=1)
            if 'level_0' in points7.columns:
                points7 = points7.drop(columns=['level_0'])
            
            # Cambiar la columna 'index' a tipo object si existe
            if 'index' in points7.columns:
                points7['index'] = points7['index'].astype(object)
            
            # Cambiar todas las otras columnas a float64, excepto 'geometry'
            for column in points7.columns:
                if column != 'geometry' and column != 'index':
                    points7[column] = points7[column].astype(float)
            #points7.to_file(ruta520)
            
            if not points7.empty:
                points7.to_file(ruta541)
            else:
                pass
        else:
            pass
    
            
    funcion_principal_df11()
    #################################
    def funcion_principal_df22():
        global df22
        df11a = df22
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd

            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            gdf.to_file(ruta500)
            df = df11a
            # if df['Impact Radius'].sum() == 0:
            #     # Establece el valor de la primera fila de 'Impact Radius' a 1
            #     df.loc[0, 'Impact Radius'] = 1
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
                        
            df499 = (df498['Impact Radius']).div(1)
            df499.fillna(0, inplace=True)
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
            centroids = calcular_centroides_de_shapefile(ruta500) #Funcion centroides  
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Longitud', 'Latitud'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
                
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['people']
                # count = inter['people'].value_counts()
                # inter['risk_pop'] = count #inter['people'].value_counts()
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            ruta22 = ruta110.removesuffix('.shp')
            ruta4 = ruta110
            
            
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
            ushp7 = gpd.read_file(ruta110)
            
            
                
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6, ushp7], ignore_index=True), crs=ushp.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            #fusion = fusion1.drop(['Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
        
            
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit INss']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit INss'][0]/1
            pr1 = df503['Probit INss'][1]/1
            pr2 = df503['Probit INss'][2]/1
            pr3 = df503['Probit INss'][3]/1
            pr4 = df503['Probit INss'][4]/1
            pr5 = df503['Probit INss'][5]/1
            pr6 = df503['Probit INss'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
        
            
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            # Verificar si gdf_merge no está vacío antes de proceder
            # Inicialización y procesamiento de gdf_merge.
            if not gdf_merge.empty:
                gdf_merge = gdf_merge.reset_index(drop=True)
                gdf_merge.fillna(0, inplace=True)
                gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
                count = gdf_merge['people'].count() if 'people' in gdf_merge.columns else 0
            else:
                count = 0
            
            # Inicializar gdf_mergep aquí para asegurarse de que siempre esté definido
            gdf_mergep = gpd.GeoDataFrame()
            
            # Procesar y llenar gdf_mergep basado en gdf_merge procesado.
            if count > 0:
                # Asumiendo que la lógica de filtrado y procesamiento para gdf_merge ya ha ocurrido.
                gdfp = gdf_merge[gdf_merge['risk_pop'] != 0].copy()  # Usar .copy() para evitar SettingWithCopyWarning
                countgdfp = gdfp['people'].count() if 'people' in gdfp.columns else 0
                if countgdfp > 0:
                    # Asegurar que no haya división por cero
                    gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                    gdf_mergep_list = [gdfp]  # Crear lista con el DataFrame procesado
                else:
                    # Si gdfp está vacío después del filtrado, gdf_mergep_list se queda como lista vacía
                    gdf_mergep_list = []
            
                if gdf_mergep_list:
                    gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
                    gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
        
        
            
            #gdf_mergep.to_file(ruta510)
            # Verificar si gdf_mergep no está vacío antes de intentar guardarlo en un archivo
            if not gdf_mergep.empty:
                 gdf_mergep.to_file(ruta510)
            else:
                pass
        
            centroids = [] #Empy
               
            try:
                     
                with fiona.open(ruta510) as f:
                    # Itera sobre todas las entradas del shapefile
                    for feature in f:
                        # Obtiene la geometría del polígono
                        polygon = shape(feature['geometry'])
                        # Calcula el centroide del polígono
                        centroid = polygon.centroid
                        # Almacena las coordenadas del centroide en la lista
                        centroids.append((centroid.x, centroid.y))
            except Exception as e:
                pass
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            #end process to building
            
            
            def intersect_and_save1(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                #inter['length'] = inter['geometry'].apply(lambda x: x.length)
                inter['risk_pop'] = 1
                
                inter.to_file(output_shp)
                
            #Star process to households
            ruta22 = ruta100.removesuffix('.shp')
            ruta4 = ruta100
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
               
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6], ignore_index=True), crs=ushp.crs)
            lines = fusion.boundary
            line = pd.DataFrame({'geometry': lines})
            
            
            ushp7 = gpd.read_file(ruta100)
            fusion = gpd.GeoDataFrame(pd.concat([ushp7,line], ignore_index=True), crs=ushp7.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            buffer = fusion1.buffer(0.1)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=lines.crs)
            fusion2 = fusion1['people']
            gdf_concat = pd.concat([fusion2, buffer_gdf], axis=1)
            gdf_concat2 = gpd.GeoDataFrame(gdf_concat, geometry='geometry')
            # Save the entire layer as a new shapefile
            gdf_concat2.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
                
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save1(shp, ruta4, ruta70)
            intersect_and_save1(shp1, ruta4, ruta71)
            intersect_and_save1(shp2, ruta4, ruta72)
            intersect_and_save1(shp3, ruta4, ruta73)
            intersect_and_save1(shp4, ruta4, ruta74)
            intersect_and_save1(shp5, ruta4, ruta75)
            intersect_and_save1(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit LNss']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit LNss'][0]/1
            pr1 = df503['Probit LNss'][1]/1
            pr2 = df503['Probit LNss'][2]/1
            pr3 = df503['Probit LNss'][3]/1
            pr4 = df503['Probit LNss'][4]/1
            pr5 = df503['Probit LNss'][5]/1
            pr6 = df503['Probit LNss'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            # Verificar si gdf_merge no está vacío antes de proceder
            if not gdf_merge.empty:
                gdf_merge = gdf_merge.reset_index(drop=True)
                gdf_merge.fillna(0, inplace=True)
                gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
                count = gdf_merge['people'].count() if 'people' in gdf_merge else 0  # Asegura que la columna 'people' exista
                gdf_mergep_list = []
                
                # Solo proceder si gdf_merge tiene registros después de filtrar por 'risk_pop'
                if not gdf_merge.empty:
                    gdfp = gdf_merge[gdf_merge['risk_pop'] != 0]
                    countgdfp = gdfp['people'].count() if 'people' in gdfp else 0  # Asegura que la columna 'people' exista en gdfp
                    if countgdfp > 0 and count > 0:  # Asegura que no haya división por cero
                        gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                    gdf_mergep_list.append(gdfp)
                    
                    gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
                    gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
            else:
                pass
            
            #gdf_mergep.to_file(ruta510)
            # Verificar si gdf_mergep no está vacío antes de intentar guardarlo en un archivo
            if not gdf_mergep.empty:
                gdf_mergep.to_file(ruta520) #export lines shapefile
            else:
                pass
                
            # Inicializar GeoDataFrames vacíos
            shp200 = gpd.GeoDataFrame()
            shp300 = gpd.GeoDataFrame()
            
            # Intentar cargar shp200 si el archivo existe
            if os.path.exists(ruta510):
                try:
                    shp200 = gpd.read_file(ruta510)
                except Exception as e:
                    pass
            
            # Intentar cargar shp300 si el archivo existe
            if os.path.exists(ruta520):
                try:
                    shp300 = gpd.read_file(ruta520)
                except Exception as e:
                    pass
            
            # Proceder solo si al menos uno de los GeoDataFrames no está vacío
            import geopandas as gpd
            
            fusion4 = pd.DataFrame(columns=['people', 'area', 'density', 'risk_pop'])
            # Convertir el DataFrame a GeoDataFrame sin especificar geometría
            fusion4 = gpd.GeoDataFrame(fusion4)
            fusion4['geometry'] = None
            fusion4 = fusion4.set_geometry('geometry')
            
                
            if not shp200.empty or not shp300.empty:
                # Determinar el sistema de referencia de coordenadas (CRS) a partir de shp200 o shp300, el que esté disponible
                crs = shp200.crs if not shp200.empty else shp300.crs
                
                # Concatenar shp200 y shp300, considerando el CRS obtenido
                fusion3 = gpd.GeoDataFrame(pd.concat([shp200, shp300], ignore_index=True), crs=crs)
                fusion4 = fusion3.explode("geometry").reset_index(drop=True)
                
                # Guardar fusion4 si no está vacío
                if not fusion4.empty:
                    fusion4.to_file(ruta530)
                else:
                    pass
            else:
                pass
        
            centroids = [] # Lista vacía para almacenar centroides, si es necesario en pasos posteriores
        
               
            try:
                with fiona.open(ruta530) as f:
                    for feature in f:
                        polygon = shape(feature['geometry'])
                        centroid = polygon.centroid
                        centroids.append((centroid.x, centroid.y))
                               
            except fiona.errors.DriverError:
                pass
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            # Suponiendo que fusion4 es un DataFrame o GeoDataFrame previamente definido
            df_concatenado2 = pd.concat([fusion4, df6], axis=1)
            points = df_concatenado2.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            if not zone.empty:
                points2 = zone.copy()
                points2['risk_pop'] = points2['people']
                points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
                filas= points2['geometry']
            else:
                pass
            
            points2 = points2[~points2['geometry'].isna() & (points2['geometry'] != '')]
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['area'], axis=1)
            if 'level_0' in points7.columns:
                points7 = points7.drop(columns=['level_0'])
            
            # Cambiar la columna 'index' a tipo object si existe
            if 'index' in points7.columns:
                points7['index'] = points7['index'].astype(object)
            
            # Cambiar todas las otras columnas a float64, excepto 'geometry'
            for column in points7.columns:
                if column != 'geometry' and column != 'index':
                    points7[column] = points7[column].astype(float)
            #points7.to_file(ruta520)
            
            if not points7.empty:
                points7.to_file(ruta542)
            else:
                pass
        else:
            pass
        
    
    funcion_principal_df22()
    #################################
    def funcion_principal_df33():
        global df33
        df11a = df33
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd

            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            gdf.to_file(ruta500)
            df = df11a
            # if df['Impact Radius'].sum() == 0:
            #     # Establece el valor de la primera fila de 'Impact Radius' a 1
            #     df.loc[0, 'Impact Radius'] = 1
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
                        
            df499 = (df498['Impact Radius']).div(1)
            df499.fillna(0, inplace=True)
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
            centroids = calcular_centroides_de_shapefile(ruta500) #Funcion centroides  
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Longitud', 'Latitud'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
                
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                inter = inter.reindex(columns=['index', 'people', 'area', 'geometry'])
                inter = inter[inter['people'] != 0]
                inter['density'] = inter['people']/(inter['area']/100)
                inter['risk_pop'] = inter['people']
                # count = inter['people'].value_counts()
                # inter['risk_pop'] = count #inter['people'].value_counts()
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            ruta22 = ruta110.removesuffix('.shp')
            ruta4 = ruta110
            
            
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
            ushp7 = gpd.read_file(ruta110)
            
            
                
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6, ushp7], ignore_index=True), crs=ushp.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            #fusion = fusion1.drop(['Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
        
            
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit INss']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit INss'][0]/1
            pr1 = df503['Probit INss'][1]/1
            pr2 = df503['Probit INss'][2]/1
            pr3 = df503['Probit INss'][3]/1
            pr4 = df503['Probit INss'][4]/1
            pr5 = df503['Probit INss'][5]/1
            pr6 = df503['Probit INss'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
        
            
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            # Verificar si gdf_merge no está vacío antes de proceder
            # Inicialización y procesamiento de gdf_merge.
            if not gdf_merge.empty:
                gdf_merge = gdf_merge.reset_index(drop=True)
                gdf_merge.fillna(0, inplace=True)
                gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
                count = gdf_merge['people'].count() if 'people' in gdf_merge.columns else 0
            else:
                count = 0
            
            # Inicializar gdf_mergep aquí para asegurarse de que siempre esté definido
            gdf_mergep = gpd.GeoDataFrame()
            
            # Procesar y llenar gdf_mergep basado en gdf_merge procesado.
            if count > 0:
                # Asumiendo que la lógica de filtrado y procesamiento para gdf_merge ya ha ocurrido.
                gdfp = gdf_merge[gdf_merge['risk_pop'] != 0].copy()  # Usar .copy() para evitar SettingWithCopyWarning
                countgdfp = gdfp['people'].count() if 'people' in gdfp.columns else 0
                if countgdfp > 0:
                    # Asegurar que no haya división por cero
                    gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                    gdf_mergep_list = [gdfp]  # Crear lista con el DataFrame procesado
                else:
                    # Si gdfp está vacío después del filtrado, gdf_mergep_list se queda como lista vacía
                    gdf_mergep_list = []
            
                if gdf_mergep_list:
                    gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
                    gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
        
        
            
            #gdf_mergep.to_file(ruta510)
            # Verificar si gdf_mergep no está vacío antes de intentar guardarlo en un archivo
            if not gdf_mergep.empty:
                 gdf_mergep.to_file(ruta510)
            else:
                pass
        
            centroids = [] #Empy
               
            try:
                     
                with fiona.open(ruta510) as f:
                    # Itera sobre todas las entradas del shapefile
                    for feature in f:
                        # Obtiene la geometría del polígono
                        polygon = shape(feature['geometry'])
                        # Calcula el centroide del polígono
                        centroid = polygon.centroid
                        # Almacena las coordenadas del centroide en la lista
                        centroids.append((centroid.x, centroid.y))
            except Exception as e:
                pass
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            #end process to building
            
            
            def intersect_and_save1(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                #inter['length'] = inter['geometry'].apply(lambda x: x.length)
                inter['risk_pop'] = 1
                
                inter.to_file(output_shp)
                
            #Star process to households
            ruta22 = ruta100.removesuffix('.shp')
            ruta4 = ruta100
            ushp = gpd.read_file(shp)
            ushp1 = gpd.read_file(shp1)
            ushp2 = gpd.read_file(shp2)
            ushp3 = gpd.read_file(shp3)
            ushp4 = gpd.read_file(shp4)
            ushp5 = gpd.read_file(shp5)
            ushp6 = gpd.read_file(shp6)
               
            # Perform spatial merge of the two shapefiles
            fusion = gpd.GeoDataFrame(pd.concat([ushp, ushp1, ushp2, ushp3, ushp4, ushp5, ushp6], ignore_index=True), crs=ushp.crs)
            lines = fusion.boundary
            line = pd.DataFrame({'geometry': lines})
            
            
            ushp7 = gpd.read_file(ruta100)
            fusion = gpd.GeoDataFrame(pd.concat([ushp7,line], ignore_index=True), crs=ushp7.crs)
            fusion['people'] = fusion['people'].fillna(0)
            from shapely.geometry import MultiPoint
            fusion1 = fusion.explode("geometry").reset_index(drop=True)
            buffer = fusion1.buffer(0.1)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=lines.crs)
            fusion2 = fusion1['people']
            gdf_concat = pd.concat([fusion2, buffer_gdf], axis=1)
            gdf_concat2 = gpd.GeoDataFrame(gdf_concat, geometry='geometry')
            # Save the entire layer as a new shapefile
            gdf_concat2.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
                
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save1(shp, ruta4, ruta70)
            intersect_and_save1(shp1, ruta4, ruta71)
            intersect_and_save1(shp2, ruta4, ruta72)
            intersect_and_save1(shp3, ruta4, ruta73)
            intersect_and_save1(shp4, ruta4, ruta74)
            intersect_and_save1(shp5, ruta4, ruta75)
            intersect_and_save1(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit LNss']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit LNss'][0]/1
            pr1 = df503['Probit LNss'][1]/1
            pr2 = df503['Probit LNss'][2]/1
            pr3 = df503['Probit LNss'][3]/1
            pr4 = df503['Probit LNss'][4]/1
            pr5 = df503['Probit LNss'][5]/1
            pr6 = df503['Probit LNss'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            # Verificar si gdf_merge no está vacío antes de proceder
            if not gdf_merge.empty:
                gdf_merge = gdf_merge.reset_index(drop=True)
                gdf_merge.fillna(0, inplace=True)
                gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
                count = gdf_merge['people'].count() if 'people' in gdf_merge else 0  # Asegura que la columna 'people' exista
                gdf_mergep_list = []
                
                # Solo proceder si gdf_merge tiene registros después de filtrar por 'risk_pop'
                if not gdf_merge.empty:
                    gdfp = gdf_merge[gdf_merge['risk_pop'] != 0]
                    countgdfp = gdfp['people'].count() if 'people' in gdfp else 0  # Asegura que la columna 'people' exista en gdfp
                    if countgdfp > 0 and count > 0:  # Asegura que no haya división por cero
                        gdfp['risk_pop'] = (gdfp['risk_pop'] * countgdfp) / count
                    gdf_mergep_list.append(gdfp)
                    
                    gdf_mergep = pd.concat(gdf_mergep_list).reset_index(drop=True)
                    gdf_mergep = gdf_mergep[gdf_mergep['risk_pop'] != 0]
            else:
                pass
            
            #gdf_mergep.to_file(ruta510)
            # Verificar si gdf_mergep no está vacío antes de intentar guardarlo en un archivo
            if not gdf_mergep.empty:
                gdf_mergep.to_file(ruta520) #export lines shapefile
            else:
                pass
                
            # Inicializar GeoDataFrames vacíos
            shp200 = gpd.GeoDataFrame()
            shp300 = gpd.GeoDataFrame()
            
            # Intentar cargar shp200 si el archivo existe
            if os.path.exists(ruta510):
                try:
                    shp200 = gpd.read_file(ruta510)
                except Exception as e:
                    pass
            
            # Intentar cargar shp300 si el archivo existe
            if os.path.exists(ruta520):
                try:
                    shp300 = gpd.read_file(ruta520)
                except Exception as e:
                    pass
            
            # Proceder solo si al menos uno de los GeoDataFrames no está vacío
            import geopandas as gpd
            
            fusion4 = pd.DataFrame(columns=['people', 'area', 'density', 'risk_pop'])
            # Convertir el DataFrame a GeoDataFrame sin especificar geometría
            fusion4 = gpd.GeoDataFrame(fusion4)
            fusion4['geometry'] = None
            fusion4 = fusion4.set_geometry('geometry')
            
                
            if not shp200.empty or not shp300.empty:
                # Determinar el sistema de referencia de coordenadas (CRS) a partir de shp200 o shp300, el que esté disponible
                crs = shp200.crs if not shp200.empty else shp300.crs
                
                # Concatenar shp200 y shp300, considerando el CRS obtenido
                fusion3 = gpd.GeoDataFrame(pd.concat([shp200, shp300], ignore_index=True), crs=crs)
                fusion4 = fusion3.explode("geometry").reset_index(drop=True)
                
                # Guardar fusion4 si no está vacío
                if not fusion4.empty:
                    fusion4.to_file(ruta530)
                else:
                    pass
            else:
                pass
        
            centroids = [] # Lista vacía para almacenar centroides, si es necesario en pasos posteriores
        
               
            try:
                with fiona.open(ruta530) as f:
                    for feature in f:
                        polygon = shape(feature['geometry'])
                        centroid = polygon.centroid
                        centroids.append((centroid.x, centroid.y))
                               
            except fiona.errors.DriverError:
                pass
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            # Suponiendo que fusion4 es un DataFrame o GeoDataFrame previamente definido
            df_concatenado2 = pd.concat([fusion4, df6], axis=1)
            points = df_concatenado2.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            if not zone.empty:
                points2 = zone.copy()
                points2['risk_pop'] = points2['people']
                points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)) if x is not None else None)
                filas= points2['geometry']
            else:
                pass
            
            points2 = points2[~points2['geometry'].isna() & (points2['geometry'] != '')]
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['area'], axis=1)
            if 'level_0' in points7.columns:
                points7 = points7.drop(columns=['level_0'])
            
            # Cambiar la columna 'index' a tipo object si existe
            if 'index' in points7.columns:
                points7['index'] = points7['index'].astype(object)
            
            # Cambiar todas las otras columnas a float64, excepto 'geometry'
            for column in points7.columns:
                if column != 'geometry' and column != 'index':
                    points7[column] = points7[column].astype(float)
            #points7.to_file(ruta520)
            
            if not points7.empty:
                points7.to_file(ruta543)
            else:
                pass
        else:
            pass
    funcion_principal_df33()
    #################################
    rutas_shapefiles = [ruta540, ruta541, ruta542, ruta543]
    visualizar_interpolacion_y_tablasocial(rutas_shapefiles, ruta120, ventana, ruta545)    
    
    
def proc_infrastructure():
    import geopandas as gpd
    from shapely.geometry import Point
    ruta4a = ruta100.removesuffix('.shp') ## social index
    ruta545 = ruta4a + "salidasoccindex.shp"
    ruta4b = ruta90.removesuffix('.shp') # public index
    ruta546 = ruta4b + "salidapubindex.shp"
    ruta4c = ruta10.removesuffix('.shp') #buildings
    ruta547 = ruta4c + "salidaspopbindex.shp" # Households and Building index
    ruta548 = ruta4c + "salidasinfraestructureindex.shp" #Indice infraestructure

    
    gdfs = []
    
    # Intenta leer cada archivo y añádelo a la lista si es exitoso
    for ruta in [ruta545, ruta546, ruta547]:
        try:
            gdf = gpd.read_file(ruta)
            gdfs.append(gdf)
        except Exception as e:
            print(f"No se pudo cargar el archivo {ruta}: {e}")
    
    # Verifica si se leyeron GeoDataFrames y concaténalos
    if gdfs:
        gdf_merge1 = pd.concat(gdfs, ignore_index=True)
        gdf_merge1 = gdf_merge1.reset_index(drop=True)
        gdf_merge1.to_file(ruta548)
        df_concatenado2 = gdf_merge1
    else:
        print("No se encontraron archivos para procesar.")
        df_concatenado2 = pd.DataFrame()  # Crea un DataFrame vacío si no se encontraron archivos

    
    try:
        # Cargar y mostrar datos TIFF como imagen de fondo usando GDAL
        ds = gdal.Open(ruta120)
        if ds is None:
            messagebox.showerror("Error", f"No se pudo abrir el archivo TIFF {ruta120}")
            return
    
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
    
        # Crear figura y eje para la trama
        fig, ax = plt.subplots()
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')

        
        points_gdf = gpd.read_file(ruta548)
        points_gdf = points_gdf.dropna(subset=['geometry'])
        # Encontrar el valor más pequeño no nulo en 'Environmen'
        min_value = points_gdf['risk_pop'].dropna().min()
        min_value = min_value * 0.001
        # Llenar los valores nulos en 'Environmen' con el valor más pequeño encontrado
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_pop'] = min_value
       
        # Interpola los puntos en una malla
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
        # # Ajustando valores mínimos para 'risk'
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
        
                
        # Asumiendo que x, y, z_filled, y ax ya están definidos correctamente
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)  # Ajusta según sea necesario
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Infrastructure index')

        
        import tkinter as tk
               
        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
                   
                
        table = df_concatenado2
        table['Este'] = table['Este'].fillna(0)
        table = table[table.Este != 0]
        table1 = table[table.risk_pop != 0]
        table2 = table1[['people', 'risk_pop', 'Este', 'Norte']]
        root5 = table2
        
        Label(text = "Table Risk on Infrastructure index             ", fg = 'black', font= ("Times New Roman",10)).place(x=24, y=144)
        frame = tk.Frame(ventana)
        frame.pack(fill='both', expand=True)
        frame.place(x=20, y=170, width=650, height=560)
        pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
        pt.show()
        ventana.mainloop()
    except Exception as e:
        points_gdf = gpd.read_file(ruta548)
        # Interpola los puntos en una malla
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
        xi, yi = np.mgrid[min(x):max(x):500j, min(y):max(y):500j]
        zi = griddata((x, y), z, (xi, yi), method='linear')
        
        # Visualiza las curvas de nivel
        contours = plt.contour(xi, yi, zi, 15, linewidths=2.0, cmap='jet')
        plt.colorbar(contours)
        # plt.clabel(contours, inline=True, fontsize=10)
        # plt.show()
        
        import tkinter as tk
            
        
        
        canvas = tk.Canvas(ventana)
        canvas.pack()
        
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
        
        table = df_concatenado2
        table['Este'] = table['Este'].fillna(0)
        table = table[table.Este != 0]
        table1 = table[table.risk_pop != 0]
        table2 = table1[['people', 'risk_pop', 'Este', 'Norte']]
        root5 = table2
        
        Label(text = "Table Risk on Infrastructure index             ", fg = 'black', font= ("Times New Roman",10)).place(x=24, y=144)
        frame = tk.Frame(ventana)
        frame.pack(fill='both', expand=True)
        frame.place(x=20, y=170, width=650, height=560)
        pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
        pt.show()
        ventana.mainloop()   
##############################################################################################

def visualizar_interpolacion_y_tablaaffected5(rutas_shapefiles, ruta120, ventana, ruta724):
    try:
        from matplotlib.colors import ListedColormap
        import numpy as np
        import matplotlib.pyplot as plt
        # Cargar la imagen TIFF usando GDAL
        ds = gdal.Open(ruta120)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]

        # Crear figura y ejes para la trama
        fig, ax = plt.subplots()

        # Mostrar los datos TIFF en el eje con su sistema de referencia de coordenadas original
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')

        gdfs = []
        for ruta in rutas_shapefiles:
            try:
                gdf = gpd.read_file(ruta)
                gdfs.append(gdf)
            except Exception as e:
                print(f"Error al leer {ruta}: {e}")

        if not gdfs:
            print("No se encontraron archivos válidos.")
            return

        points_gdf = pd.concat(gdfs, ignore_index=True) if gdfs else pd.DataFrame()
        points_gdf = points_gdf.dropna(subset=['geometry'])

        if points_gdf.empty:
            print("No hay datos para visualizar.")
            return

        # Ajustar 'affected' según sea necesario
        min_value = points_gdf['risk_prod'].dropna().min() * 0.001
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_prod'] = min_value

        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_prod']
        # # Ajustando valores mínimos para 'risk'
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
        
                
        # Asumiendo que x, y, z_filled, y ax ya están definidos correctamente
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)  # Ajusta según sea necesario
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Productive Activities Index')

        canvas = tk.Canvas(ventana)
        canvas.pack()

        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)

        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)

        # Exportar points_gdf si no está vacío
        if not points_gdf.empty:
            points_gdf.to_file(ruta724, driver='ESRI Shapefile')
            print(f"Archivo exportado a {ruta724}")

            table = points_gdf[points_gdf.level != 0]
            table = points_gdf[points_gdf.level != 0].dropna()
            table2 = table[['level', 'risk_prod', 'Este', 'Norte']]
            root50 = table2

            Label(text="Table productive activities affected", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root50, showtoolbar=True, showstatusbar=True)
            pt.show()

    except Exception as e:
        pass

    
    
def proc_activities():
    from shapely.geometry import shape, Point        
    import geopandas as gpd
    ruta2 = ruta.removesuffix('.shp') #Tanks
    ruta400 = ruta2 + "1.shp"
    ruta3 = ruta80.removesuffix('.shp') #productive
    ruta401 = ruta3 + "1.shp"
    ruta405 = ruta2 + "11.shp"
    ruta500 = ruta2 + "100.shp"
    ruta510 = ruta3 + "index2.shp"
    ruta720 = ruta3 + "salida.shp"
    ruta721 = ruta3 + "salida1.shp"
    ruta722 = ruta3 + "salida2.shp"
    ruta723 = ruta3 + "salida3.shp"
    ruta724 = ruta3 + "salidaactivitiesindex.shp"
    import os

    def borrar_shapefiles(*rutas):
        for ruta in rutas:
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
                    
                else:
                    pass
            except Exception as e:
                continue
    
    rutas_shapefiles = [
        ruta510, ruta720, ruta721, ruta722, ruta723, ruta724
    ]
    
    borrar_shapefiles(*rutas_shapefiles)
       
    def funcion_principal_df00():
        global df00
        df11a = df00
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'level', 'area', 'geometry'])
                # inter['people'] = inter['level']
                inter = inter[inter['level'] != 0]
                inter['risk_pop'] = 1
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            
            ruta22 = ruta80.removesuffix('.shp')
            ruta4 = ruta80
            poligonos = gpd.read_file(ruta400)
        
            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union
        
            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        
            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})
        
            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')
        
            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])
        
            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})
        
            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))
        
            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta80)
            shp2 = gpd.read_file(ruta405)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.overlay(shp1, shp2, how='union')
            fusion['level'] = fusion['level'].fillna(0)
            fusion = fusion.drop(['Id_1', 'Clase', 'Este', 'Norte', 'IDTK', 'Id_2', 'Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
            
            
            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit IAp']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit IAp'][0]/1
            pr1 = df503['Probit IAp'][1]/1
            pr2 = df503['Probit IAp'][2]/1
            pr3 = df503['Probit IAp'][3]/1
            pr4 = df503['Probit IAp'][4]/1
            pr5 = df503['Probit IAp'][5]/1
            pr6 = df503['Probit IAp'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(ffp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
            sumatoria = (gdf_merge['areap'].sum())/10000
            gdf_merge['risk_pop'] = (gdf_merge['risk_pop'] * (gdf_merge['area']/100))/sumatoria
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge.to_file(ruta510)
            centroids = [] #Empy
            
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            # df_puntos1['Env_risk'] = df_puntos1['Env_risk'] * 1000000
            # df_puntos = df_puntos1.drop(index=df_puntos1[df_puntos1['Env_risk'] == 1].index)
            
            # df_puntos = df_puntos1.loc[df_puntos1['people'] != 0.000001]
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['FID', 'area', 'people'], axis=1)
            points7 = points7.rename(columns={'risk_pop': 'risk_prod'})
            #points7.to_file(ruta520)
            if not points7.empty:
                points7.to_file(ruta720)
            else:
                pass
        else:
            pass
    funcion_principal_df00()
    ############################################
    def funcion_principal_df11():
        global df11
        df11a = df11
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'level', 'area', 'geometry'])
                # inter['people'] = inter['level']
                inter = inter[inter['level'] != 0]
                inter['risk_pop'] = 1
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            
            ruta22 = ruta80.removesuffix('.shp')
            ruta4 = ruta80
            poligonos = gpd.read_file(ruta400)
        
            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union
        
            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        
            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})
        
            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')
        
            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])
        
            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})
        
            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))
        
            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta80)
            shp2 = gpd.read_file(ruta405)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.overlay(shp1, shp2, how='union')
            fusion['level'] = fusion['level'].fillna(0)
            fusion = fusion.drop(['Id_1', 'Clase', 'Este', 'Norte', 'IDTK', 'Id_2', 'Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
            
            
            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit IAp']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit IAp'][0]/1
            pr1 = df503['Probit IAp'][1]/1
            pr2 = df503['Probit IAp'][2]/1
            pr3 = df503['Probit IAp'][3]/1
            pr4 = df503['Probit IAp'][4]/1
            pr5 = df503['Probit IAp'][5]/1
            pr6 = df503['Probit IAp'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(jfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
            sumatoria = (gdf_merge['areap'].sum())/10000
            gdf_merge['risk_pop'] = (gdf_merge['risk_pop'] * (gdf_merge['area']/100))/sumatoria
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge.to_file(ruta510)
            centroids = [] #Empy
            
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            # df_puntos1['Env_risk'] = df_puntos1['Env_risk'] * 1000000
            # df_puntos = df_puntos1.drop(index=df_puntos1[df_puntos1['Env_risk'] == 1].index)
            
            # df_puntos = df_puntos1.loc[df_puntos1['people'] != 0.000001]
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['FID', 'area', 'people'], axis=1)
            points7 = points7.rename(columns={'risk_pop': 'risk_prod'})
            #points7.to_file(ruta520)
            if not points7.empty:
                points7.to_file(ruta721)
            else:
                pass
        else:
            pass
    funcion_principal_df11()
    
    #################################
    def funcion_principal_df22():
        global df22
        df11a = df22
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'level', 'area', 'geometry'])
                # inter['people'] = inter['level']
                inter = inter[inter['level'] != 0]
                inter['risk_pop'] = 1
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            
            ruta22 = ruta80.removesuffix('.shp')
            ruta4 = ruta80
            poligonos = gpd.read_file(ruta400)
        
            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union
        
            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        
            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})
        
            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')
        
            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])
        
            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})
        
            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))
        
            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta80)
            shp2 = gpd.read_file(ruta405)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.overlay(shp1, shp2, how='union')
            fusion['level'] = fusion['level'].fillna(0)
            fusion = fusion.drop(['Id_1', 'Clase', 'Este', 'Norte', 'IDTK', 'Id_2', 'Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
            
            
            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit IAp']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit IAp'][0]/1
            pr1 = df503['Probit IAp'][1]/1
            pr2 = df503['Probit IAp'][2]/1
            pr3 = df503['Probit IAp'][3]/1
            pr4 = df503['Probit IAp'][4]/1
            pr5 = df503['Probit IAp'][5]/1
            pr6 = df503['Probit IAp'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(pfp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
            sumatoria = (gdf_merge['areap'].sum())/10000
            gdf_merge['risk_pop'] = (gdf_merge['risk_pop'] * (gdf_merge['area']/100))/sumatoria
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge.to_file(ruta510)
            centroids = [] #Empy
            
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            # df_puntos1['Env_risk'] = df_puntos1['Env_risk'] * 1000000
            # df_puntos = df_puntos1.drop(index=df_puntos1[df_puntos1['Env_risk'] == 1].index)
            
            # df_puntos = df_puntos1.loc[df_puntos1['people'] != 0.000001]
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['FID', 'area', 'people'], axis=1)
            points7 = points7.rename(columns={'risk_pop': 'risk_prod'})
            #points7.to_file(ruta520)
            if not points7.empty:
                points7.to_file(ruta722)
            else:
                pass
        else:
            pass
    funcion_principal_df22()
    
    #################################
    def funcion_principal_df33():
        global df33
        df11a = df33
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
            gdf0 = gpd.read_file(ruta400)
            gdf = gdf0[gdf0.IDTK != 255]
            gdf = gdf[gdf.IDTK != 0]
            # gdf00 = gdf.to_crs(3116)#3116/3857
            # gdf000 = gdf00.to_crs(4326)#4326
            gdf.to_file(ruta500)
            df = df11a
            df498 = df44.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0}, ignore_index=True)
            #dian = 30
            # def sumar_dian(x):
            #     return x + (float(d.get())) if x > 0 else x
            # df498['Impact Radius'] = df498['Impact Radius'].apply(sumar_dian)
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
               
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]
        
            df3 = df600
            #df4 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'level', 'area', 'geometry'])
                # inter['people'] = inter['level']
                inter = inter[inter['level'] != 0]
                inter['risk_pop'] = 1
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Define la ruta base
            
            ruta22 = ruta80.removesuffix('.shp')
            ruta4 = ruta80
            poligonos = gpd.read_file(ruta400)
        
            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union
        
            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        
            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})
        
            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')
        
            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])
        
            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})
        
            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))
        
            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta80)
            shp2 = gpd.read_file(ruta405)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.overlay(shp1, shp2, how='union')
            fusion['level'] = fusion['level'].fillna(0)
            fusion = fusion.drop(['Id_1', 'Clase', 'Este', 'Norte', 'IDTK', 'Id_2', 'Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500
            
            
            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit IAp']).div(1)
            df503 = pd.DataFrame(df501)
            
            #probabilities value
            pr = df503['Probit IAp'][0]/1
            pr1 = df503['Probit IAp'][1]/1
            pr2 = df503['Probit IAp'][2]/1
            pr3 = df503['Probit IAp'][3]/1
            pr4 = df503['Probit IAp'][4]/1
            pr5 = df503['Probit IAp'][5]/1
            pr6 = df503['Probit IAp'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['risk_pop'] *= pr1
            gdf1['risk_pop'] *= pr2
            gdf2['risk_pop'] *= pr3
            gdf3['risk_pop'] *= pr4
            gdf4['risk_pop'] *= pr5
            gdf5['risk_pop'] *= pr6
            gdf6['risk_pop'] *= pr
            
            gdf['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf1['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf2['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf3['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf4['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf5['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            gdf6['risk_pop'] *= ( (float(frec.get()) ) + 0) * ( (float(opp.get()) ) + 0)
            
            import geopandas as gpd
            from shapely.geometry import Point
            gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
            gdf_merge = gdf_merge[gdf_merge['risk_pop'] != 0]
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge['areap'] = gdf_merge['geometry'].apply(lambda x: x.area)
            sumatoria = (gdf_merge['areap'].sum())/10000
            gdf_merge['risk_pop'] = (gdf_merge['risk_pop'] * (gdf_merge['area']/100))/sumatoria
            gdf_merge = gdf_merge.reset_index(drop=True)
            gdf_merge.to_file(ruta510)
            centroids = [] #Empy
            
            with fiona.open(ruta510) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))
        
            # DataFrame centroids
            df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
            points = df_concatenado.copy()
            # change geometry 
            points['geometry'] = points['geometry'].centroid
            
            # df4 = df3
            # df4 = df4.assign(index='0', people= 0.0001) #(float(f.get()))
            from shapely.geometry import MultiPoint
            points2 = zone.copy()
            points2['risk_pop'] = points2['people']
            points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
            filas= points2['geometry']
               
            df_puntos1 = points2.explode("geometry").reset_index(drop=True)
            # df_puntos1['Env_risk'] = df_puntos1['Env_risk'] * 1000000
            # df_puntos = df_puntos1.drop(index=df_puntos1[df_puntos1['Env_risk'] == 1].index)
            
            # df_puntos = df_puntos1.loc[df_puntos1['people'] != 0.000001]
            df_puntos = df_puntos1
                
            points3 = points.append(df_puntos, ignore_index=True)
            points3['risk_pop'] = points3['risk_pop'].fillna(0)
            points5 = points3[points3.risk_pop != 0]
            points6 = points5[points5.risk_pop != 1.0000000000000002e-06]
            min_value = points6['risk_pop'].min()
            points7 = points6[points6.risk_pop != min_value]
            points7 = points7.drop(['FID', 'area', 'people'], axis=1)
            points7 = points7.rename(columns={'risk_pop': 'risk_prod'})
            #points7.to_file(ruta520)
            if not points7.empty:
                points7.to_file(ruta723)
            else:
                pass
        else:
            pass
    funcion_principal_df33()
    
    #################################
    rutas_shapefiles = [ruta720, ruta721, ruta722, ruta723]
    visualizar_interpolacion_y_tablaaffected5(rutas_shapefiles, ruta120, ventana, ruta724)  
        
############################################################################################
def proc_socioeconomic():
    import geopandas as gpd
    from shapely.geometry import Point
    import geopandas as gpd
        
    def try_read_file(filepath):
        try:
            return gpd.read_file(filepath)
        except Exception as e:
            return gpd.GeoDataFrame()
    
    
    ruta4f = ruta10.removesuffix('.shp') #buildings
    ruta5233 = ruta4f + "salidaspopulationindex.shp" #Population

    ruta4g = ruta10.removesuffix('.shp') #buildings
    ruta548 = ruta4g + "salidasinfraestructureindex.shp" #Indice infraestructure

    ruta3h = ruta80.removesuffix('.shp') #productive
    ruta724 = ruta3h + "salidaactivitiesindex.shp"
    
    gdf0 = try_read_file(ruta724)
    gdf0 = gdf0.rename(columns={'risk_prod': 'risk_pop'})
    gdf1 = try_read_file(ruta548)
    gdf2 = try_read_file(ruta5233)
    
    # Realiza las operaciones necesarias solo si los DataFrames no están vacíos
    if not gdf0.empty:
        gdf0['risk_pop'] *= (float(pfac.get()) + 0)
    if not gdf1.empty:
        gdf1['risk_pop'] *= (float(ifac.get()) + 0)
    if not gdf2.empty:
        gdf2['risk_pop'] *= (float(popfac.get()) + 0)
    
    # Concatena los GeoDataFrames que se lograron leer y procesar
    gdfs = [df for df in [gdf0, gdf1, gdf2] if not df.empty]
    if gdfs:
        gdf_merge1 = pd.concat(gdfs, ignore_index=True)
        gdf_merge1 = gdf_merge1.reset_index(drop=True)
        gdf_merge1.to_file(ruta548)
        df_concatenado2 = gdf_merge1
    else:
        df_concatenado2 = pd.DataFrame()  # Crea un DataFrame vacío si no se encontraron archivos
    
    try:
        from osgeo import gdal
        from matplotlib.colors import ListedColormap
        import numpy as np
        import matplotlib.pyplot as plt
    
        # Cargar y mostrar datos TIFF como imagen de fondo usando GDAL
        ds = gdal.Open(ruta120)
        if ds is None:
            messagebox.showerror("Error", f"No se pudo abrir el archivo TIFF {ruta120}")
            return
    
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
    
        # Crear figura y eje para la trama
        fig, ax = plt.subplots()
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
    
        points_gdf = gpd.read_file(ruta548)
        points_gdf = points_gdf.dropna(subset=['geometry'])
        
        # Encontrar el valor más pequeño no nulo en 'risk_pop'
        min_value = points_gdf['risk_pop'].dropna().min() * 0.001
        points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'risk_pop'] = min_value
    
        # Interpolar los puntos en una malla
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
    
        # Ajustar valores mínimos para 'risk_pop'
        z_filled = np.where(z > 0, z, min_value)
    
        # Crear niveles y mapa de colores
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
    
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Socioeconomic Risk')
    
        import tkinter as tk
    
        canvas = tk.Canvas(ventana)
        canvas.pack()
    
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
    
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
    
        table = df_concatenado2
        table['Este'] = table['Este'].fillna(0)
        table = table[table.Este != 0]
        table1 = table[table.risk_pop != 0]
        table2 = table1[['risk_pop', 'Este', 'Norte']]
        root5 = table2
    
        Label(text="Table Socioeconomic risk index                 ", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
        frame = tk.Frame(ventana)
        frame.pack(fill='both', expand=True)
        frame.place(x=20, y=170, width=650, height=560)
        pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
        pt.show()
        ventana.mainloop()
    
    except Exception as e:
        points_gdf = gpd.read_file(ruta548)
        # Interpolar los puntos en una malla
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['risk_pop']
        xi, yi = np.mgrid[min(x):max(x):500j, min(y):max(y):500j]
        zi = griddata((x, y), z, (xi, yi), method='linear')
    
        # Visualizar las curvas de nivel
        contours = plt.contour(xi, yi, zi, 15, linewidths=2.0, cmap='jet')
        plt.colorbar(contours)
        
        canvas = tk.Canvas(ventana)
        canvas.pack()
    
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
    
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
    
        table = df_concatenado2
        table['Este'] = table['Este'].fillna(0)
        table = table[table.Este != 0]
        table1 = table[table.risk_pop != 0]
        table2 = table1[['risk_pop', 'Este', 'Norte']]
        root5 = table2
    
        Label(text="Table Socioeconomic risk index                  ", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
        frame = tk.Frame(ventana)
        frame.pack(fill='both', expand=True)
        frame.place(x=20, y=170, width=650, height=560)
        pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
        pt.show()
        ventana.mainloop()

        

def proc_enviromental():
    try:
        if 'df44' not in globals():  # Comprueba si df44 está definido globalmente
            raise NameError("df44")
        if 'ruta50' not in globals():  # Comprueba si ruta50 está definido globalmente
            raise NameError("ruta50")
    except NameError as e:
        error_message = "Error desconocido"
        if str(e) == "df44":
            error_message = "La cobertura de tanques no ha sido cargada"
        elif str(e) == "ruta50":
            error_message = "La cobertura ambiental no ha sido cargada"
        messagebox.showerror("Error", error_message)
        return  # Sale de la función si df44 o ruta50 no están definidos
    
    ruta2 = ruta.removesuffix('.shp')
    ruta400 = ruta2 + "1.shp"
    ruta405 = ruta2 + "11.shp"
    ruta500 = ruta2 + "100.shp"
    ruta510 = ruta2 + "poly.shp"
    ruta520 = ruta2 + "point.shp"  #df00
    ruta521 = ruta2 + "point1.shp"  #df11
    ruta522 = ruta2 + "point2.shp"  #df22
    ruta523 = ruta2 + "point3.shp"  #df33
    ruta530 = ruta2 + "krig.shp"
    ruta540 = ruta2 + "contours.png"
    ruta600 = ruta2 + "200.shp"
    
    import os
    
    # Lista de archivos para verificar y limpiar
    archivos_limpiar = [ruta520, ruta521, ruta522, ruta523, ruta510]
    
    # Itera sobre cada archivo en la lista
    for archivo in archivos_limpiar:
        if os.path.exists(archivo):
            os.remove(archivo)
    
    from shapely.geometry import shape, Point        
    import geopandas as gpd

    gdf0 = gpd.read_file(ruta400)
    gdf = gdf0[gdf0.IDTK != 255]
    gdf = gdf[gdf.IDTK != 0]
    gdf00 = gdf.to_crs(3116)#3116/3857
    gdf000 = gdf00.to_crs(4326)#4326
    gdf.to_file(ruta500)
    
    
    def funcion_principal_df00():
        global df00
        df11a = df00
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
                                             
            df498 = df00.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
                   
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))

            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

            df3 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'level', 'area', 'geometry'])
                # inter['people'] = inter['level']
                inter = inter[inter['sensitive'] != 0]
                inter['Environmen'] = 1
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Defines the base path
            
            ruta22 = ruta50.removesuffix('.shp')
            ruta4 = ruta50 
            
            poligonos = gpd.read_file(ruta400)

            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union

            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})

            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')

            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])

            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})

            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))

            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta50)
            shp1 = shp1[shp1['sensitive'] != 1]
            shp2 = gpd.read_file(ruta405)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.overlay(shp1, shp2, how='union')
            fusion['sensitive'] = fusion['sensitive'].fillna(0)
            fusion = fusion.drop(['Id_1', 'Clase', 'Este', 'Norte', 'IDTK', 'Id_2', 'Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500

            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit Enlc']).div(1)
            df503 = pd.DataFrame(df501)
            #df503['Probit Enlc'] = 0.001

            #probabilities value
            pr = df503['Probit Enlc'][0]/1
            pr1 = df503['Probit Enlc'][1]/1
            pr2 = df503['Probit Enlc'][2]/1
            pr3 = df503['Probit Enlc'][3]/1
            pr4 = df503['Probit Enlc'][4]/1
            pr5 = df503['Probit Enlc'][5]/1
            pr6 = df503['Probit Enlc'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['Environmen'] *= pr1
            gdf1['Environmen'] *= pr2
            gdf2['Environmen'] *= pr3
            gdf3['Environmen'] *= pr4
            gdf4['Environmen'] *= pr5
            gdf5['Environmen'] *= pr6
            gdf6['Environmen'] *= pr
            
            
            # Función para aplicar la condición a cada gdf
            import geopandas as gpd

            try:
                valor_frec = float(frec.get())
            
                def apply_condition(df1, coef):
                    if not df1.empty:
                        if df1['sensitive'].iloc[0] == 3:
                            df1['Environmen'] *= valor_frec * coef
                        elif df1['sensitive'].iloc[0] == 2:
                            df1['Environmen'] *= valor_frec
                    return df1
            
                # Lista de GeoDataFrames y sus coeficientes correspondientes
                gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
                coefs = [1, 1, 1, 0.76744186, 0, 0, 0]
            
                # Verificar si todos los GeoDataFrames están vacíos
                if all(df1.empty for df1 in gdfs):
                    print("No hay datos para procesar.")
                    pass
                else:
                    # Aplicar condiciones a cada gdf si no están vacíos
                    for i, (df1, coef) in enumerate(zip(gdfs, coefs)):
                        gdfs[i] = apply_condition(df1, coef)
            
                    # Actualizar los GeoDataFrames individuales después del procesamiento
                    gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6 = gdfs
            
                # Procesar resultados adicionales si es necesario
                # (el código para esto iría aquí)
            
            except Exception as e:
                print(f"Error: {e}")
                pass
                      
            
            try:
                gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
                gdf_merge = gdf_merge.explode('geometry')
                gdf_merge = gdf_merge[gdf_merge['Environmen'] != 0]
                gdf_merge['areapar'] = gdf_merge['geometry'].apply(lambda x: x.area)
                sumatoria = (gdf_merge['areapar'].sum())/10000
                
                # Aplicar la multiplicación por 'Env_risk' y manejar multiplicaciones por cero
                try:
                    gdf_merge['Env_risk'] = (gdf_merge['Environmen'] * (gdf_merge['areapar']/10000) * gdf_merge['sensitive'])/sumatoria
                except ZeroDivisionError:
                    pass
            
                gdf_merge = gdf_merge.reset_index(drop=True)
                
                # Exportar el GeoDataFrame solo si no está vacío
                if not gdf_merge.empty:
                    gdf_merge.to_file(ruta510)
                else:
                    pass
            
            except Exception as e:
                pass
            
            
            if not gdf_merge.empty:
                gdf_merge.to_file(ruta510)
            else:
                pass
            
            # Verificar si el archivo en ruta510 está vacío y si es así, pasar
            if not os.path.exists(ruta510) or os.path.getsize(ruta510) == 0:
                pass
            else:
                centroids = [] # Lista vacía para centroids
            
                with fiona.open(ruta510) as f:
                    # Iterar sobre todas las entradas en el shapefile
                    for feature in f:
                        # Obtener la geometría del polígono
                        polygon = shape(feature['geometry'])
                        # Calcular el centroide del polígono
                        centroid = polygon.centroid
                        # Almacenar las coordenadas del centroide en la lista
                        centroids.append((centroid.x, centroid.y))
            
                # DataFrame de centroids
                df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
                df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
                points = df_concatenado.copy()
                # Cambiar geometría 
                points['geometry'] = points['geometry'].centroid
            
                points2 = zone.copy()
                points2['Env_risk'] = points2['people']
                points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
                filas = points2['geometry']
            
                df_puntos1 = points2.explode("geometry").reset_index(drop=True)
                df_puntos = df_puntos1
            
                points3 = points.append(df_puntos, ignore_index=True)
                points3['Env_risk'] = points3['Env_risk'].fillna(0)
                points5 = points3[points3.Env_risk != 0]
                points6 = points5[points5.Env_risk != 1.0000000000000002e-06]
                min_value = points6['Env_risk'].min()
                points7 = points6[points6.Env_risk != min_value]
                points7 = points7.drop(['FID', 'area', 'people'], axis=1)
                # points7.to_file(ruta520)
                if not points6.empty:
                    points6.to_file(ruta522)
                else:
                    pass
        else:
            pass
    funcion_principal_df00()
    ############################
    def funcion_principal_df11():
        global df11
        df11a = df11
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
                                             
            df498 = df11.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
                   
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))

            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

            df3 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'level', 'area', 'geometry'])
                # inter['people'] = inter['level']
                inter = inter[inter['sensitive'] != 0]
                inter['Environmen'] = 1
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Defines the base path
            
            ruta22 = ruta50.removesuffix('.shp')
            ruta4 = ruta50 
            
            poligonos = gpd.read_file(ruta400)

            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union

            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})

            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')

            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])

            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})

            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))

            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta50)
            shp1 = shp1[shp1['sensitive'] != 1]
            shp2 = gpd.read_file(ruta405)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.overlay(shp1, shp2, how='union')
            fusion['sensitive'] = fusion['sensitive'].fillna(0)
            fusion = fusion.drop(['Id_1', 'Clase', 'Este', 'Norte', 'IDTK', 'Id_2', 'Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500

            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit Enlc']).div(1)
            df503 = pd.DataFrame(df501)
            #df503['Probit Enlc'] = 0.001

            #probabilities value
            pr = df503['Probit Enlc'][0]/1
            pr1 = df503['Probit Enlc'][1]/1
            pr2 = df503['Probit Enlc'][2]/1
            pr3 = df503['Probit Enlc'][3]/1
            pr4 = df503['Probit Enlc'][4]/1
            pr5 = df503['Probit Enlc'][5]/1
            pr6 = df503['Probit Enlc'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['Environmen'] *= pr1
            gdf1['Environmen'] *= pr2
            gdf2['Environmen'] *= pr3
            gdf3['Environmen'] *= pr4
            gdf4['Environmen'] *= pr5
            gdf5['Environmen'] *= pr6
            gdf6['Environmen'] *= pr
            
            
            # Función para aplicar la condición a cada gdf
            import geopandas as gpd

            try:
                valor_frec = float(frec.get())
            
                def apply_condition(df1, coef):
                    if not df1.empty:
                        if df1['sensitive'].iloc[0] == 3:
                            df1['Environmen'] *= valor_frec * coef
                        elif df1['sensitive'].iloc[0] == 2:
                            df1['Environmen'] *= valor_frec
                    return df1
            
                # Lista de GeoDataFrames y sus coeficientes correspondientes
                gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
                coefs = [1, 1, 1, 0.76744186, 0, 0, 0]
            
                # Verificar si todos los GeoDataFrames están vacíos
                if all(df1.empty for df1 in gdfs):
                    print("No hay datos para procesar.")
                    pass
                else:
                    # Aplicar condiciones a cada gdf si no están vacíos
                    for i, (df1, coef) in enumerate(zip(gdfs, coefs)):
                        gdfs[i] = apply_condition(df1, coef)
            
                    # Actualizar los GeoDataFrames individuales después del procesamiento
                    gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6 = gdfs
            
                # Procesar resultados adicionales si es necesario
                # (el código para esto iría aquí)
            
            except Exception as e:
                print(f"Error: {e}")
                pass
                      
            
            try:
                gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
                gdf_merge = gdf_merge.explode('geometry')
                gdf_merge = gdf_merge[gdf_merge['Environmen'] != 0]
                gdf_merge['areapar'] = gdf_merge['geometry'].apply(lambda x: x.area)
                sumatoria = (gdf_merge['areapar'].sum())/10000
                
                # Aplicar la multiplicación por 'Env_risk' y manejar multiplicaciones por cero
                try:
                    gdf_merge['Env_risk'] = (gdf_merge['Environmen'] * (gdf_merge['areapar']/10000) * gdf_merge['sensitive'])/sumatoria
                except ZeroDivisionError:
                    pass
            
                gdf_merge = gdf_merge.reset_index(drop=True)
                
                # Exportar el GeoDataFrame solo si no está vacío
                if not gdf_merge.empty:
                    gdf_merge.to_file(ruta510)
                else:
                    pass
            
            except Exception as e:
                pass
            
            
            if not gdf_merge.empty:
                gdf_merge.to_file(ruta510)
            else:
                pass
            
            # Verificar si el archivo en ruta510 está vacío y si es así, pasar
            if not os.path.exists(ruta510) or os.path.getsize(ruta510) == 0:
                pass
            else:
                centroids = [] # Lista vacía para centroids
            
                with fiona.open(ruta510) as f:
                    # Iterar sobre todas las entradas en el shapefile
                    for feature in f:
                        # Obtener la geometría del polígono
                        polygon = shape(feature['geometry'])
                        # Calcular el centroide del polígono
                        centroid = polygon.centroid
                        # Almacenar las coordenadas del centroide en la lista
                        centroids.append((centroid.x, centroid.y))
            
                # DataFrame de centroids
                df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
                df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
                points = df_concatenado.copy()
                # Cambiar geometría 
                points['geometry'] = points['geometry'].centroid
            
                points2 = zone.copy()
                points2['Env_risk'] = points2['people']
                points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
                filas = points2['geometry']
            
                df_puntos1 = points2.explode("geometry").reset_index(drop=True)
                df_puntos = df_puntos1
            
                points3 = points.append(df_puntos, ignore_index=True)
                points3['Env_risk'] = points3['Env_risk'].fillna(0)
                points5 = points3[points3.Env_risk != 0]
                points6 = points5[points5.Env_risk != 1.0000000000000002e-06]
                min_value = points6['Env_risk'].min()
                points7 = points6[points6.Env_risk != min_value]
                points7 = points7.drop(['FID', 'area', 'people'], axis=1)
                # points7.to_file(ruta520)
                if not points6.empty:
                    points6.to_file(ruta522)
                else:
                    pass
        else:
            pass
    funcion_principal_df11()
    ############################
    def funcion_principal_df22():
        global df22
        df11a = df22
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
                                             
            df498 = df22.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
                   
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))

            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

            df3 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'level', 'area', 'geometry'])
                # inter['people'] = inter['level']
                inter = inter[inter['sensitive'] != 0]
                inter['Environmen'] = 1
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Defines the base path
            
            ruta22 = ruta50.removesuffix('.shp')
            ruta4 = ruta50 
            
            poligonos = gpd.read_file(ruta400)

            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union

            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})

            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')

            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])

            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})

            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))

            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta50)
            shp1 = shp1[shp1['sensitive'] != 1]
            shp2 = gpd.read_file(ruta405)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.overlay(shp1, shp2, how='union')
            fusion['sensitive'] = fusion['sensitive'].fillna(0)
            fusion = fusion.drop(['Id_1', 'Clase', 'Este', 'Norte', 'IDTK', 'Id_2', 'Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500

            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit Enlc']).div(1)
            df503 = pd.DataFrame(df501)
            #df503['Probit Enlc'] = 0.001

            #probabilities value
            pr = df503['Probit Enlc'][0]/1
            pr1 = df503['Probit Enlc'][1]/1
            pr2 = df503['Probit Enlc'][2]/1
            pr3 = df503['Probit Enlc'][3]/1
            pr4 = df503['Probit Enlc'][4]/1
            pr5 = df503['Probit Enlc'][5]/1
            pr6 = df503['Probit Enlc'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['Environmen'] *= pr1
            gdf1['Environmen'] *= pr2
            gdf2['Environmen'] *= pr3
            gdf3['Environmen'] *= pr4
            gdf4['Environmen'] *= pr5
            gdf5['Environmen'] *= pr6
            gdf6['Environmen'] *= pr
            
            
            # Función para aplicar la condición a cada gdf
            import geopandas as gpd

            try:
                valor_frec = float(frec.get())
            
                def apply_condition(df1, coef):
                    if not df1.empty:
                        if df1['sensitive'].iloc[0] == 3:
                            df1['Environmen'] *= valor_frec * coef
                        elif df1['sensitive'].iloc[0] == 2:
                            df1['Environmen'] *= valor_frec
                    return df1
            
                # Lista de GeoDataFrames y sus coeficientes correspondientes
                gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
                coefs = [1, 1, 1, 0.76744186, 0, 0, 0]
            
                # Verificar si todos los GeoDataFrames están vacíos
                if all(df1.empty for df1 in gdfs):
                    print("No hay datos para procesar.")
                    pass
                else:
                    # Aplicar condiciones a cada gdf si no están vacíos
                    for i, (df1, coef) in enumerate(zip(gdfs, coefs)):
                        gdfs[i] = apply_condition(df1, coef)
            
                    # Actualizar los GeoDataFrames individuales después del procesamiento
                    gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6 = gdfs
            
                # Procesar resultados adicionales si es necesario
                # (el código para esto iría aquí)
            
            except Exception as e:
                print(f"Error: {e}")
                pass
                      
            
            try:
                gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
                gdf_merge = gdf_merge.explode('geometry')
                gdf_merge = gdf_merge[gdf_merge['Environmen'] != 0]
                gdf_merge['areapar'] = gdf_merge['geometry'].apply(lambda x: x.area)
                sumatoria = (gdf_merge['areapar'].sum())/10000
                
                # Aplicar la multiplicación por 'Env_risk' y manejar multiplicaciones por cero
                try:
                    gdf_merge['Env_risk'] = (gdf_merge['Environmen'] * (gdf_merge['areapar']/10000) * gdf_merge['sensitive'])/sumatoria
                except ZeroDivisionError:
                    pass
            
                gdf_merge = gdf_merge.reset_index(drop=True)
                
                # Exportar el GeoDataFrame solo si no está vacío
                if not gdf_merge.empty:
                    gdf_merge.to_file(ruta510)
                else:
                    pass
            
            except Exception as e:
                pass
            
            
            if not gdf_merge.empty:
                gdf_merge.to_file(ruta510)
            else:
                pass
            
            # Verificar si el archivo en ruta510 está vacío y si es así, pasar
            if not os.path.exists(ruta510) or os.path.getsize(ruta510) == 0:
                pass
            else:
                centroids = [] # Lista vacía para centroids
            
                with fiona.open(ruta510) as f:
                    # Iterar sobre todas las entradas en el shapefile
                    for feature in f:
                        # Obtener la geometría del polígono
                        polygon = shape(feature['geometry'])
                        # Calcular el centroide del polígono
                        centroid = polygon.centroid
                        # Almacenar las coordenadas del centroide en la lista
                        centroids.append((centroid.x, centroid.y))
            
                # DataFrame de centroids
                df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
                df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
                points = df_concatenado.copy()
                # Cambiar geometría 
                points['geometry'] = points['geometry'].centroid
            
                points2 = zone.copy()
                points2['Env_risk'] = points2['people']
                points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
                filas = points2['geometry']
            
                df_puntos1 = points2.explode("geometry").reset_index(drop=True)
                df_puntos = df_puntos1
            
                points3 = points.append(df_puntos, ignore_index=True)
                points3['Env_risk'] = points3['Env_risk'].fillna(0)
                points5 = points3[points3.Env_risk != 0]
                points6 = points5[points5.Env_risk != 1.0000000000000002e-06]
                min_value = points6['Env_risk'].min()
                points7 = points6[points6.Env_risk != min_value]
                points7 = points7.drop(['FID', 'area', 'people'], axis=1)
                # points7.to_file(ruta520)
                if not points6.empty:
                    points6.to_file(ruta522)
                else:
                    pass
                   
        else:
            pass
    funcion_principal_df22()
    ############################
    def funcion_principal_df33():
        global df33
        df11a = df33
        if df11a['Impact Radius'].sum() <= 1:
            # Reinicializar df11a para que solo contenga las columnas sin filas
            df11a = pd.DataFrame(columns=df11a.columns)
        else:
            pass
        
        if not df11a.empty:
            # DataFrame no está vacío, realiza las operaciones
            from shapely.geometry import shape, Point
            import geopandas as gpd
                                             
            df498 = df33.sort_values(by='Impact Radius')
            df498 = df498.reset_index(inplace=False, drop=True)
            files = len(df498)
            file = 7 - files
            for i in range(file):
                df498 = df498.append({'Class': 0, 'Impact Radius': 0, 'Probability': 0, 'Probit People': 0, 'Probit House': 0, 'Probit IAp': 0, 'Probit INps': 0, 'Probit LNps': 0, 'Probit INss': 0, 'Probit LNss': 0, 'Probit Ewater': 0, 'Probit Enlc': 0, 'Probit Eforest': 0}, ignore_index=True)
            
                    
            df499 = (df498['Impact Radius']).div(1) #35971.22302158273
            df500 = pd.DataFrame(df499)
            import os
            parent_dir = ruta2.rsplit('/', 1)[0]
            os.chdir(parent_dir)
            
            from shapely.geometry import shape, Point
            centroids = [] #Empy
                   
            with fiona.open(ruta500) as f:
                # Iterates over all the entries in the shapefile
                for feature in f:
                    # Gets the geometry of the polygon
                    polygon = shape(feature['geometry'])
                    # Calculate the centroid of the polygon
                    centroid = polygon.centroid
                    # Stores the coordinates of the centroid in the list
                    centroids.append((centroid.x, centroid.y))

            # DataFrame centroids
            df5 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
            shapefile = gpd.read_file(ruta500)
            df_concatenado = pd.concat([shapefile, df5], axis=1)
                
            shapefile01 = df_concatenado.sort_values(by='IDTK', ascending=True)
            shapefile02 = shapefile01.reset_index(drop=True)
            shapefile03 = shapefile02[shapefile02.IDTK != 255]
            shapefile03['IDTK'] = [i for i, row in enumerate(shapefile03.index)]
            shapefile03['IDTK'] = shapefile03['IDTK']  + 1
            
            df600 = shapefile03.loc[shapefile03['IDTK'] == (float(s.get()))]

            df3 = df600
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            #generate the rip buffer
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df3{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
                
                shapefile1 = gpd.read_file(filename)
                if index < len(df) - 1:
                    shapefile2 = gpd.read_file(f"df3{index+2}.shp")
                else:
                    # si es el último shapefile generado, no hacemos nada más
                    continue
                geometry1 = shapefile1["geometry"]
                geometry2 = shapefile2["geometry"]
                
                # convertimos las columnas a GeoSeries
                geoseries1 = gpd.GeoSeries(geometry1)
                geoseries2 = gpd.GeoSeries(geometry2)
                
                # realizamos el clip usando la función difference de geopandas
                clipped_shapefile = geoseries2.difference(geoseries1, align=True)
                
                # guardamos el resultado en un archivo .shp nuevo
                clipped_shapefile.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            for index, row in df500.iterrows():
                r = row[0]
                buffer = df3.buffer(r)
                filename = f"df55{index+1}.shp"
                buffer.to_file(filename, driver="ESRI Shapefile", encoding="utf-8")
            
            import os
            import glob
            # Obtener la lista de todos los archivos .shp en el directorio '/path/to/folder'
            shp_files = glob.glob(parent_dir + '/*.shp')
            for i, shp_file in enumerate(shp_files):
                shp_files[i] = shp_file.replace("\\", "/")
                    
            # Obtener solo los nombres de archivo sin la ruta
            #shp_filenames = [os.path.basename(f) for f in shp_files]
            #print(shp_filenames)  # ['shapefile1.shp', 'shapefile2.shp', 'shapefile3.shp']
            work = [w for w in shp_files if w.find('df3') != -1]
            work1 = [w for w in shp_files if w.find('df551') != -1]
            work = pd.DataFrame(work)
            work1 = pd.DataFrame(work1)
            
            def intersect_and_save(shp1, shp2, output_shp):
                # Realiza la intersección entre los shapefiles
                g1 = gpd.GeoDataFrame.from_file(shp1)
                g2 = gpd.GeoDataFrame.from_file(shp2)
                inter = gpd.overlay(g1, g2, how='intersection')
                # Calcula el área de cada polígono de la intersección
                inter['area'] = inter['geometry'].apply(lambda x: x.area)
                # Reorganiza las columnas del GeoDataFrame
                #inter = inter.reindex(columns=['index', 'level', 'area', 'geometry'])
                # inter['people'] = inter['level']
                inter = inter[inter['sensitive'] != 0]
                inter['Environmen'] = 1
                # Guarda el resultado en un archivo shapefile
                inter.to_file(output_shp)
            
            # Defines the base path
            
            ruta22 = ruta50.removesuffix('.shp')
            ruta4 = ruta50 
            
            poligonos = gpd.read_file(ruta400)

            # Unite all polygons into a single geometric object
            union_geometria = poligonos.unary_union

            # Create a new polygon that covers all the space within the shapefile
            xmin, ymin, xmax, ymax = union_geometria.bounds
            nuevo_poligono = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

            # Create a new layer for the filled polygons
            poligonos_rellenos = gpd.GeoDataFrame({'Id': [1000], 'geometry': [nuevo_poligono]})

            # Merge the original and filled polygon layers
            poligonos_completos = gpd.overlay(poligonos, poligonos_rellenos, how='union')

            # Create a new polygon that covers all the space inside the shapefile, but twice the size
            xmin, ymin, xmax, ymax = nuevo_poligono.bounds
            doble_poligono = Polygon([(xmin - (xmax - xmin), ymin - (ymax - ymin)), (xmin - (xmax - xmin), ymax + (ymax - ymin)), 
                                      (xmax + (xmax - xmin), ymax + (ymax - ymin)), (xmax + (xmax - xmin), ymin - (ymax - ymin))])

            # Create a new layer for the double polygon
            doble_capa = gpd.GeoDataFrame({'Id': [1000], 'geometry': [doble_poligono]})

            # Merge the full polygon layers and the double polygon into a single layer
            fusion_capas = gpd.GeoDataFrame(pd.concat([poligonos_completos, doble_capa], ignore_index=True))

            # Save the entire layer as a new shapefile
            fusion_capas.to_file(ruta405, driver='ESRI Shapefile')
            # load the two shapefiles to be merged
            shp1 = gpd.read_file(ruta50)
            shp1 = shp1[shp1['sensitive'] != 1]
            shp2 = gpd.read_file(ruta405)
            
            # Perform spatial merge of the two shapefiles
            fusion = gpd.overlay(shp1, shp2, how='union')
            fusion['sensitive'] = fusion['sensitive'].fillna(0)
            fusion = fusion.drop(['Id_1', 'Clase', 'Este', 'Norte', 'IDTK', 'Id_2', 'Id'], axis=1)
            # Save the entire layer as a new shapefile
            fusion.to_file(ruta500, driver='ESRI Shapefile')
            ruta4 = ruta500

            # export.to_file(ruta530)
            # Define la lista de sufijos
            sufijos = ['100', '70', '71', '72', '73', '74', '75', '76']
            # Genera las rutas de los archivos shapefile
            rutas = [ruta22 + sufijo + ".shp" for sufijo in sufijos]
            ruta7 = rutas[0]  
            ruta70 = rutas[1]
            ruta71 = rutas[2]
            ruta72 = rutas[3]
            ruta73 = rutas[4]
            ruta74 = rutas[5]
            ruta75 = rutas[6]
            ruta76 = rutas[7]
               
            shp = work[0][0]
            shp1 = work[0][1]
            shp2 = work[0][2]
            shp3 = work[0][3]
            shp4 = work[0][4]
            shp5 = work[0][5]
            shp6 = work1[0][0]
            
            z0 = gpd.read_file(shp)
            z1 = gpd.read_file(shp1)
            z2 = gpd.read_file(shp2)
            z3 = gpd.read_file(shp3)
            z4 = gpd.read_file(shp4)
            z5 = gpd.read_file(shp5)
            z6 = gpd.read_file(shp6)
            zone = pd.concat([z6, z0, z1, z2, z3, z4, z5])
            
            intersect_and_save(shp, ruta4, ruta70)
            intersect_and_save(shp1, ruta4, ruta71)
            intersect_and_save(shp2, ruta4, ruta72)
            intersect_and_save(shp3, ruta4, ruta73)
            intersect_and_save(shp4, ruta4, ruta74)
            intersect_and_save(shp5, ruta4, ruta75)
            intersect_and_save(shp6, ruta4, ruta76)
            
            df501 = (df498['Probit Enlc']).div(1)
            df503 = pd.DataFrame(df501)
            #df503['Probit Enlc'] = 0.001

            #probabilities value
            pr = df503['Probit Enlc'][0]/1
            pr1 = df503['Probit Enlc'][1]/1
            pr2 = df503['Probit Enlc'][2]/1
            pr3 = df503['Probit Enlc'][3]/1
            pr4 = df503['Probit Enlc'][4]/1
            pr5 = df503['Probit Enlc'][5]/1
            pr6 = df503['Probit Enlc'][6]/1
            
            v0 = 0 + (float(frec.get()))
            v1 = pr1 * ( (float(frec.get()) ) + 0)
            v2 = pr2 * ( (float(frec.get()) ) + 0)
            v3 = pr3 * ( (float(frec.get()) ) + 0)
            v4 = pr4 * ( (float(frec.get()) ) + 0)
            v5 = pr5 * ( (float(frec.get()) ) + 0)
            v6 = 0.01 * ( (float(frec.get()) ) + 0) #0.0001
            
            v00 = []
            v00.append (v0)
            v00.append (v1)
            v00.append (v2)
            v00.append (v3)
            v00.append (v4)
            v00.append (v5)
            v00.append (v6)
            
            zone = zone.assign(people=v00)    
            
            gdf = gpd.read_file(ruta70)
            gdf1 = gpd.read_file(ruta71)
            gdf2 = gpd.read_file(ruta72)
            gdf3 = gpd.read_file(ruta73)
            gdf4 = gpd.read_file(ruta74)
            gdf5 = gpd.read_file(ruta75)
            gdf6 = gpd.read_file(ruta76)
            
            gdf['Environmen'] *= pr1
            gdf1['Environmen'] *= pr2
            gdf2['Environmen'] *= pr3
            gdf3['Environmen'] *= pr4
            gdf4['Environmen'] *= pr5
            gdf5['Environmen'] *= pr6
            gdf6['Environmen'] *= pr
            
            
            # Función para aplicar la condición a cada gdf
            import geopandas as gpd

            try:
                valor_frec = float(frec.get())
            
                def apply_condition(df1, coef):
                    if not df1.empty:
                        if df1['sensitive'].iloc[0] == 3:
                            df1['Environmen'] *= valor_frec * coef
                        elif df1['sensitive'].iloc[0] == 2:
                            df1['Environmen'] *= valor_frec
                    return df1
            
                # Lista de GeoDataFrames y sus coeficientes correspondientes
                gdfs = [gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6]
                coefs = [1, 1, 1, 0.76744186, 0, 0, 0]
            
                # Verificar si todos los GeoDataFrames están vacíos
                if all(df1.empty for df1 in gdfs):
                    print("No hay datos para procesar.")
                    pass
                else:
                    # Aplicar condiciones a cada gdf si no están vacíos
                    for i, (df1, coef) in enumerate(zip(gdfs, coefs)):
                        gdfs[i] = apply_condition(df1, coef)
            
                    # Actualizar los GeoDataFrames individuales después del procesamiento
                    gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6 = gdfs
            
                # Procesar resultados adicionales si es necesario
                # (el código para esto iría aquí)
            
            except Exception as e:
                print(f"Error: {e}")
                pass
                      
            
            try:
                gdf_merge = pd.concat([gdf, gdf1, gdf2, gdf3, gdf4, gdf5, gdf6])
                gdf_merge = gdf_merge.explode('geometry')
                gdf_merge = gdf_merge[gdf_merge['Environmen'] != 0]
                gdf_merge['areapar'] = gdf_merge['geometry'].apply(lambda x: x.area)
                sumatoria = (gdf_merge['areapar'].sum())/10000
                
                # Aplicar la multiplicación por 'Env_risk' y manejar multiplicaciones por cero
                try:
                    gdf_merge['Env_risk'] = (gdf_merge['Environmen'] * (gdf_merge['areapar']/10000) * gdf_merge['sensitive'])/sumatoria
                except ZeroDivisionError:
                    pass
            
                gdf_merge = gdf_merge.reset_index(drop=True)
                
                # Exportar el GeoDataFrame solo si no está vacío
                if not gdf_merge.empty:
                    gdf_merge.to_file(ruta510)
                else:
                    pass
            
            except Exception as e:
                pass
            
            
            if not gdf_merge.empty:
                gdf_merge.to_file(ruta510)
            else:
                pass
            
            # Verificar si el archivo en ruta510 está vacío y si es así, pasar
            if not os.path.exists(ruta510) or os.path.getsize(ruta510) == 0:
                pass
            else:
                centroids = [] # Lista vacía para centroids
            
                with fiona.open(ruta510) as f:
                    # Iterar sobre todas las entradas en el shapefile
                    for feature in f:
                        # Obtener la geometría del polígono
                        polygon = shape(feature['geometry'])
                        # Calcular el centroide del polígono
                        centroid = polygon.centroid
                        # Almacenar las coordenadas del centroide en la lista
                        centroids.append((centroid.x, centroid.y))
            
                # DataFrame de centroids
                df6 = pd.DataFrame(centroids, columns=['Este', 'Norte'])
                df_concatenado = pd.concat([gdf_merge, df6], axis=1)
            
                points = df_concatenado.copy()
                # Cambiar geometría 
                points['geometry'] = points['geometry'].centroid
            
                points2 = zone.copy()
                points2['Env_risk'] = points2['people']
                points2.geometry = points2.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
                filas = points2['geometry']
            
                df_puntos1 = points2.explode("geometry").reset_index(drop=True)
                df_puntos = df_puntos1
            
                points3 = points.append(df_puntos, ignore_index=True)
                points3['Env_risk'] = points3['Env_risk'].fillna(0)
                points5 = points3[points3.Env_risk != 0]
                points6 = points5[points5.Env_risk != 1.0000000000000002e-06]
                min_value = points6['Env_risk'].min()
                points7 = points6[points6.Env_risk != min_value]
                points7 = points7.drop(['FID', 'area', 'people'], axis=1)
                # points7.to_file(ruta520)
                if not points6.empty:
                    points6.to_file(ruta522)
                else:
                    pass
        else:
            pass
    funcion_principal_df33()
    ############################

    rutas_shapefiles = [ruta520, ruta521, ruta522, ruta523]
    def load_geodataframe(ruta):
        if os.path.exists(ruta):
            return gpd.read_file(ruta)
        else:
            print(f"Advertencia: {ruta} no existe.")
            return gpd.GeoDataFrame()

    def process_data(ruta520, ruta521, ruta522, ruta523):
        gdf1 = load_geodataframe(ruta520)
        gdf2 = load_geodataframe(ruta521)
        gdf3 = load_geodataframe(ruta522)
        gdf4 = load_geodataframe(ruta523)
        
        points_gdf = pd.concat([gdf1, gdf2, gdf3, gdf4], ignore_index=True)
        
        if isinstance(points_gdf, gpd.GeoDataFrame):
            points_gdf = points_gdf.dropna(subset=['geometry'])
        
        return points_gdf
    
    try:
        rutas_shapefiles = [ruta520, ruta521, ruta522, ruta523]
        points_gdf = process_data(*rutas_shapefiles)
    
        if points_gdf.empty:
            messagebox.showinfo("No hay afectación ambiental")
        else:
            fig, ax = plt.subplots()
    
            # Carga y muestra de datos TIFF como imagen de fondo usando GDAL
            ds = gdal.Open(ruta120)
            data = ds.ReadAsArray()
            gt = ds.GetGeoTransform()
            extent = [gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3]]
            ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
    
            min_value = points_gdf['Env_risk'].dropna().min() * 0.001
            points_gdf.loc[(points_gdf['Este'].isnull()) | (points_gdf['Este'] == 0), 'Env_risk'] = min_value
    
            x = points_gdf['geometry'].x
            y = points_gdf['geometry'].y
            z = points_gdf['Env_risk']
            min_value = z.dropna().min() * 0.001
            z_filled = np.where(z > 0, z, min_value)
    
            levels = np.linspace(z_filled.min(), z_filled.max(), 35)
            cmap = ListedColormap([
                "white", "peru", "salmon", "darkgray", "gray",
                "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
                "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
                "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
                "gold", "orange", "darkorange", "orangered", "red", "darkred"
            ])
            contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
            plt.colorbar(contourf, ax=ax, label='Risk level Environmental Risk')
    
            plt.show()
    
            canvas = tk.Canvas(ventana)
            canvas.pack()
    
            graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
            graph_canvas.draw()
            graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
    
            mplcursors.cursor()
            toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
            toolbar.update()
            graph_canvas.get_tk_widget()
            canvas.place(x=715, y=160, width=780, height=530)
    
            table = points_gdf
            table1 = table[table.Env_risk != 0]
            table1 = table1[table1.Este != 0]
            table2 = table1[['sensitive', 'Env_risk', 'Este', 'Norte']]
            table2 = table2.dropna(subset=['Este'])
            root5 = table2
    
            Label(text="Table Environmental Risk", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
            pt.show()
    
        ventana.mainloop()
    except Exception as e:
        points_gdf = gpd.read_file(ruta520)
        x = points_gdf['geometry'].x
        y = points_gdf['geometry'].y
        z = points_gdf['Env_risk']
        min_value = z.dropna().min() * 0.001
        z_filled = np.where(z > 0, z, min_value)
    
        fig, ax = plt.subplots()
    
        ds = gdal.Open(ruta120)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]
        ax.imshow(np.moveaxis(data, 0, -1), extent=extent, origin='upper')
    
        levels = np.linspace(z_filled.min(), z_filled.max(), 35)
        cmap = ListedColormap([
            "white", "peru", "salmon", "darkgray", "gray",
            "midnightblue", "navy", "darkblue", "mediumblue", "blue", "dodgerblue",
            "deepskyblue", "lightseagreen", "turquoise", "mediumspringgreen",
            "springgreen", "limegreen", "green", "chartreuse", "yellowgreen",
            "gold", "orange", "darkorange", "orangered", "red", "darkred"
        ])
        contourf = ax.tricontourf(x, y, z_filled, levels=levels, cmap=cmap, alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Risk level Environmental Risk')
    
        plt.show()
    
        canvas = tk.Canvas(ventana)
        canvas.pack()
    
        graph_canvas = FigureCanvasTkAgg(fig, master=canvas)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
    
        mplcursors.cursor()
        toolbar = NavigationToolbar2Tk(graph_canvas, canvas)
        toolbar.update()
        graph_canvas.get_tk_widget()
        canvas.place(x=715, y=160, width=780, height=530)
    
        table = points_gdf
        table1 = table[table.Env_risk != 0]
        table1 = table1[table1.Este != 0]
        table2 = table1[['sensitive', 'Env_risk', 'Este', 'Norte']]
        table2 = table2.dropna(subset=['Este'])
        root5 = table2
    
        Label(text="Table Environmental Risk", fg='black', font=("Times New Roman", 10)).place(x=24, y=144)
        frame = tk.Frame(ventana)
        frame.pack(fill='both', expand=True)
        frame.place(x=20, y=170, width=650, height=560)
        pt = Table(frame, dataframe=root5, showtoolbar=True, showstatusbar=True)
        pt.show()
    
        ventana.mainloop()
    
        
        
############################################################################################
def limpiar():
    canvas.delete("all")
############################################################################################
"""Creacion de Ventana"""
ventana = tk.Tk()
ventana.title("ISSER")
w = 1500
h = 850
# extraW=ventana.winfo_screenwidth() - w
# extraH=ventana.winfo_screenheight() - h
# ventana.geometry("%dx%d%+d%+d" % (w, h, extraW / 2, extraH / 2))
ventana.geometry("{0}x{1}+0+0".format(ventana.winfo_screenwidth(), ventana.winfo_screenheight()))
canvas = Canvas(ventana, width=500, height=400)
canvas.pack()

# Mensaje de bienvenida
messagebox.showinfo("Bienvenido A ISSER", "Para habilitar las funcionalidades cargue la Imagen de la zona y la cobertura de Tanques en el menú Upload Spatial Data")



#####################################################################################
from tkinter import *

#Variables input
N = 1000                                                                       
d = DoubleVar()
r = DoubleVar()
radius = DoubleVar()
height1 = DoubleVar()
s = DoubleVar()
frec = DoubleVar()
popfac = DoubleVar()
ifac = DoubleVar()
pfac = DoubleVar()
rfac = DoubleVar()
ffp = DoubleVar()
jfp = DoubleVar()
pfp = DoubleVar()
opp = DoubleVar()


#################################################################################
radioframe = tk.LabelFrame(ventana, text='Modeling data', width=650, height=120)
radioframe.place(x=15, y=10)

radioframe = tk.LabelFrame(ventana, text='Consequence Process', width=790, height=120)
radioframe.place(x=710, y=10)

Label(text="Diameter [m]", font= ("Times New Roman",8)).place(x=20, y=25)
Entry(ventana, width=10, justify="center", textvariable=d).place(x=20, y=45)

Label(text="Select Number Tank", font= ("Times New Roman",8)).place(x=140, y=25)
Entry(ventana, width=10, justify="center", textvariable=s).place(x=140, y=45)

Label(text="Frecuence [n/year]", font= ("Times New Roman",8)).place(x=260, y=25)
Entry(ventana, width=10, justify="center", textvariable=frec).place(x=260, y=45)

Label(text="Population Factor %", font= ("Times New Roman",8)).place(x=720, y=30)
Entry(ventana, width=10, justify="center", textvariable=popfac).place(x=720, y=50)

Label(text="Infrastructure Factor %", font= ("Times New Roman",8)).place(x=1010, y=30)
Entry(ventana, width=10, justify="center", textvariable=ifac).place(x=1010, y=50)

Label(text="Productive Factor %", font= ("Times New Roman",8)).place(x=1190, y=30)
Entry(ventana, width=10, justify="center", textvariable=pfac).place(x=1190, y=50)

#Label(text="FF PIR", font= ("Times New Roman",8)).place(x=160, y=70)
Entry(ventana, width=8, justify="center", textvariable=ffp).place(x=150, y=70)
Entry(ventana, width=8, justify="center", textvariable=jfp).place(x=150, y=100)
Entry(ventana, width=8, justify="center", textvariable=pfp).place(x=340, y=70)
Entry(ventana, width=8, justify="center", textvariable=opp).place(x=340, y=100)

################################################################################

Button(ventana, text="Thermal flash fire PIR", bg="#00FFFF", command=table).place(x=20, y=70)
Button(ventana, text="Thermal jet fire PIR    ", bg="#00FFFF", command=table11).place(x=20, y=100)
Button(ventana, text="Thermal pool fire PIR", bg="#00FFFF", command=table22).place(x=210, y=70)
Button(ventana, text="Over pressure PIR      ", bg="#00FFFF", command=table33).place(x=210, y=100)
Button(ventana, text="Modeled tank", command=graf_RIP).place(x=400, y=65)
Button(ventana, text="Individual Risk", command=proc_people).place(x=550, y=30)
Button(ventana, text="Societal Risk", command=proc_societal).place(x=550, y=65)
Button(ventana, text="Population Index", command=proc_socioeconomicpop).place(x=720, y=100)
Button(ventana, text="Households Set", command=proc_socioeconomicpop1).place(x=860, y=65)
Button(ventana, text="Public Services Set", command=proc_socioeconomicpublic).place(x=860, y=100)
Button(ventana, text="Social Services Set", command=proc_socioeconomicsocial).place(x=860, y=30)
Button(ventana, text="Infrastructure Index", command=proc_infrastructure).place(x=1010, y=100)
Button(ventana, text="Productive Activities Index", command=proc_activities).place(x=1190, y=100)
Button(ventana, text="Socioeconomic Risk", command=proc_socioeconomic).place(x=1370, y=65)
Button(ventana, text="Enviromental Risk", command=proc_enviromental).place(x=550, y=100)
#Button(ventana, text="Add Layer", command=add_shapefile).place(x=400, y=30)

##############################################################################################
# Create a function to display the selected dataframe in a table
def show_df(df_name):
    global df44
    if df_name == "Flash fire":
        df44 = df00
    elif df_name == "Jet fire":
        df44 = df11
    elif df_name == "Pool fire":
        df44 = df22
    elif df_name == "Over pressure":
        df44 = df33
    
            
    for name, df in options:
        if df_name == name:
            
            Label(text = "RIP to process                  ", fg = 'black', font= ("Times New Roman",10)).place(x=24, y=144)
            frame = tk.Frame(ventana)
            frame.pack(fill='both', expand=True)
            frame.place(x=20, y=170, width=650, height=560)
            pt = Table(frame, dataframe=df44, showtoolbar=True, showstatusbar=True)
            pt.show()
            break
    ventana.mainloop()
    

# Create a list of options for the dropdown menu
options = [("Flash fire", df00),
           ("Jet fire", df11),
           ("Pool fire", df22),
           ("Over pressure", df33)]

# Create a variable to store the current selection
selected_df = tk.StringVar(ventana)
selected_df.set("Select end event") # Default value

# Create a dropdown menu

dropdown = tk.OptionMenu(ventana, selected_df, *[opt[0] for opt in options], command=lambda x: show_df(selected_df.get()))
dropdown.pack()
dropdown.place(x=400, y=95)
##########################################################################################
#Spatial Graph
radioframe = tk.LabelFrame(ventana, text='Spatial Analysis', bg="white", width=790, height=560)
radioframe.place(x=710, y=140)

radioframe = tk.LabelFrame(ventana, text='Data Table report', bg="white", width=650, height=560)
radioframe.place(x=20, y=140)
############################################################################################
Label(text = "Produccioón:", fg = 'black', font= ("Times New Roman",8)).place(x=10, y=745)
Label(text = "Versión Beta", fg = 'black', font =("Times New Roman",8)).place(x=60, y=745)
############################################################################################
"""Creacion De Los Menus"""
barraMenu = Menu(ventana)
mnuOpciones = Menu(barraMenu)
mnuUnidad1 = Menu(barraMenu, tearoff=0)
mnuUnidad2 = Menu(barraMenu, tearoff=0)
mnuUnidad3 = Menu(barraMenu, tearoff=0)
mnuUnidad4 = Menu(barraMenu, tearoff=0)
mnuUnidad5 = Menu(barraMenu, tearoff=0)
############################################################################################
"""Menu Upload data coverage"""
mnuUnidad1.add_command(label = "Load Image", command = upload_image)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load tank coverage", command = tanks)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load people coverage", command = people)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load enviromental coverage", command = enviromental)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load buildings coverage", command = buildings)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load households coverage", command = households)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load productive coverage", command = productive)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load public goods coverage", command = public)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load lineal public goods coverage", command = linearpublic)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load social infrastructure coverage", command = social)
mnuUnidad1.add_separator()
mnuUnidad1.add_command(label = "Load lineal social elements coverage", command = linealsocial)
mnuUnidad1.add_separator()

############################################################################################
"""Menu Opciones"""

mnuOpciones.add_command(label = "Exit", command = ventana.destroy)
############################################################################################
barraMenu.add_cascade(label = "Upload spatial data", menu = mnuUnidad1)

barraMenu.add_cascade(label = "Exit IWH", menu = mnuOpciones)

ventana.config(menu = barraMenu)
ico_path = 'D:/Trabajos/Doctorado/Ejecucion/Isser_logo.ico'
if os.path.exists(ico_path):
    ventana.iconbitmap(ico_path)
else:
    print(f"Advertencia: {ico_path} no existe. Se ejecutará la aplicación sin el icono.")

# Continuar con el resto de la configuración de la ventana
ventana.title("Isser")

ventana.mainloop()
