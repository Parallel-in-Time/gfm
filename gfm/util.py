#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:28:59 2021

@author: telu
"""
import os
import matplotlib.pyplot as plt

def setFig(xLabel, yLabel, title=None, figSize=(6.4, 4.8), grid=True,
           fileName=None):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.grid(grid)
    plt.gcf().set_size_inches(*figSize)
    plt.tight_layout()
    if fileName is not None:
        plt.savefig(fileName)
        os.system(f"pdfcrop {fileName} {fileName}")

def getLastPlotCol():
    return plt.gca().get_lines()[-1].get_color()
