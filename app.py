import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk  
import numpy as np
import time
from utils.helper import fft

xdata = []
ydata = []

def process(canvas,ax):
    x = 0
    while True:
        xdata.append(x)
        ydata.append(np.sin(x*np.pi))
        fftx,ffty = fft(np.array(ydata))
        plot(canvas,ax,xdata,ydata,fftx,ffty)
        time.sleep(0.1)
        x=x+0.1

def plot(canvas,axs,xdata,ydata,fftx,ffty):
    for i, ax in enumerate(axs):
        ax.clear()         # clear axes from previous plot
        if i == 1:
            ax.plot(fftx,ffty)
        else:
            ax.plot(xdata,ydata)
    canvas.draw()
        
class Application(tk.Frame):
    def __init__(self, root=None):
        tk.Frame.__init__(self,root)
        self.root = root
        self.createWidgets()

    def createWidgets(self):
        fig,axs=plt.subplots(3,1,figsize=(3,3))
        fig.tight_layout()
        canvas=FigureCanvasTkAgg(fig,master=self.root)
        canvas.get_tk_widget().grid(row=0,column=1)
        canvas.draw()
        self.thread = threading.Thread(target=process, args=(canvas,axs))
        self.thread.setDaemon(True)
        self.thread.start()
        
        self.plotbutton=tk.Button(master=self.root, text="Quit", command=self.root.quit)
        self.plotbutton.grid(row=0,column=0)
        
        br = 15
        lbl = tk.Label(root, text = f"Breathing rate: {br}bpm")
        lbl.grid()
        
        hr = 50
        lbl = tk.Label(root, text = f"Heart rate: {hr}hbpm")
        lbl.grid()
    
if "__main__" == __name__:
    root=tk.Tk()
    root.title("Welcome to GeekForGeeks")
    root.geometry('750x650')
    app=Application(root=root)
    app.mainloop()