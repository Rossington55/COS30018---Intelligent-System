# import search
from math import floor
from re import search
from tkinter import *
import tkintermapview
import numpy as np
import search
from progress.bar import IncrementalBar as bar
from data.data import get_scats_coords

bg = "#0b132b"
fg = "#5bc0be"
w = "#ffffff"


def handle_from(e):
    handle_target(e, "start")


def handle_to(e):
    handle_target(e, "to")


def handle_target(e, source):
    # Get lat long
    site = []
    for scat in scats:
        if scat[0] == e:
            site = scat
            break

    # Clear markers and add current one
    for marker in markers:
        if marker.source != None and marker.source != source:
            continue
        marker.delete()
    marker = mapWidg.set_marker(site[1], site[2], text=site[0])
    marker.source = source
    markers.append(marker)


def handle_submit():
    # Basic ass validation
    if fromVal.get() == "" or toVal.get() == "" or timeVal.get() == "":
        print("Selection fields are empty")
        return

    search.harrisonsMethod(int(fromVal.get()), int(toVal.get()), timeVal.get())


times = []
for i in range(96):
    times.append(f"{floor(i/4):02d}:{(int(i)%4*15):02d}")

# Init root window
root = Tk()
root.configure(background=bg)
root.title('Goggle map')
root.geometry("1080x720")


# Get all the scat numbers
print("Initializing sites...")
scats = get_scats_coords("data/data1.xls")
# Scats are in form [[no, lat, long]]
scat_numbers = []
for scat in scats:
    scat_numbers.append(scat[0])

fields = LabelFrame(root, padx=40, bg=bg, borderwidth=0)
fields.grid(row=0, column=0)

# Init entry fields
# Start/From
Label(fields, text="Start", bg=bg, fg=w).grid(row=0, column=0)
fromVal = StringVar()
fromEntry = OptionMenu(fields, fromVal, *scat_numbers,
                       command=handle_from)
fromEntry.config(bg=fg, fg=w, width=10, borderwidth=0, highlightthickness=0)
fromEntry.grid(row=1, column=0)

# To
Label(fields, bg=bg).grid(row=3, column=0)
Label(fields, text="To", bg=bg, fg=w).grid(row=4, column=0)
toVal = StringVar()
toEntry = OptionMenu(fields, toVal, *scat_numbers,
                     command=handle_to)
toEntry.config(bg=fg, fg=w, width=10, borderwidth=0, highlightthickness=0)
toEntry.grid(row=5, column=0)

# Time
Label(fields, bg=bg).grid(row=6, column=0)
Label(fields, text="Time", bg=bg, fg=w).grid(row=7, column=0)
timeVal = StringVar()
timeEntry = OptionMenu(fields, timeVal, *times)
timeEntry.config(bg=fg, fg=w, width=10, borderwidth=0, highlightthickness=0)
timeEntry.grid(row=8, column=0)

# Submit
Label(fields, bg=bg).grid(row=9, column=0)
Label(fields, bg=bg).grid(row=10, column=0)
submit = Button(fields, text="SUBMIT", command=handle_submit)
submit.config(bg=fg, fg=w, width=10, borderwidth=0)
submit.grid(row=11, column=0)


# Init map
mapLabel = LabelFrame(root, pady=40, bg=bg, borderwidth=0)
mapLabel.grid(row=0, column=1)

mapWidg = tkintermapview.TkinterMapView(
    mapLabel, width=800, height=600, corner_radius=10)
mapWidg.grid(row=0, column=0)

# Position + Zoom
mapWidg.set_position(-37.822076, 145.037794)  # EN Building
mapWidg.set_zoom(13)

# Init markers
markers = []
for site in scats:
    # Scats = [[no, lat, long]]
    marker = mapWidg.set_marker(site[1], site[2], text=site[0])
    marker.source = None
    markers.append(marker)


root.mainloop()

# search.printRoutes(search.harrisonsMethod(4051, 2825, '2006-1-10 13:00'))
