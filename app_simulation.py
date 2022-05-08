#!/usr/bin/env python
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk  
import numpy as np
import time
from utils.helper import fft

class Application(tk.Frame):
    def __init__(self, root=None):
        tk.Frame.__init__(self,root)
        self.root = root
        self.xdata = []
        self.ydata = []
        self.br = 0
        self.hr = 0
        self.strVarBr = tk.StringVar()
        self.strVarHr = tk.StringVar()
        self._reset_hr_br()
        self.createWidgets()

    def _reset_hr_br(self) :
        self.strVarBr.set(f"Breathing rate: {self.br}bpm")
        self.strVarHr.set(f"Breathing rate: {self.hr}bpm")
        self.root.update()
    
    def createWidgets(self):
        fig,axs=plt.subplots(3,1,figsize=(3,3))
        fig.tight_layout()
        canvas=FigureCanvasTkAgg(fig,master=self.root)
        canvas.get_tk_widget().grid(row=0,column=1)
        canvas.draw()
        self.thread = threading.Thread(target=self._process, args=(canvas,axs))
        self.thread.setDaemon(True)
        self.thread.start()
        
        self.plotbutton=tk.Button(master=self.root, text="Quit", command=self.root.quit)
        self.plotbutton.grid(row=0,column=0)

        lbl = tk.Label(root, textvariable= self.strVarBr)
        lbl.grid()

        lbl = tk.Label(root, textvariable= self.strVarHr)
        lbl.grid()
        
        
    def _process(self, canvas, ax):
        x = 0
        while True:
            self.xdata.append(x)
            self.ydata.append(np.sin(x*np.pi))
            fftx,ffty = fft(np.array(self.ydata))
            self._plot(canvas,ax, fftx,ffty)
            time.sleep(0.1)
            self.br = self.br + 1
            self.hr = self.hr + 2
            self._reset_hr_br()
            x=x+0.1

    def _plot(self, canvas,axs,fftx,ffty):
        for i, ax in enumerate(axs):
            ax.clear()         # clear axes from previous plot
            if i == 1:
                ax.plot(fftx,ffty)
            else:
                ax.plot(self.xdata,self.ydata)
        
        canvas.draw()
    
if "__main__" == __name__:
    root=tk.Tk()
    root.title("Amazing Radar App")
    root.geometry('750x650')
    app=Application(root=root)
    app.mainloop()