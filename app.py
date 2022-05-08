import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk  
import numpy as np
import time

def process(canvas,ax):
    while True:
        plot(canvas,ax)
        time.sleep(1)

def plot(canvas,ax):
    c = ['r','b','g']  # plot marker colors
    ax.clear()         # clear axes from previous plot
    for i in range(3):
        theta = np.random.uniform(0,360,10)
        r = np.random.uniform(0,1,10)
        ax.plot(theta,r,linestyle="None",marker='o', color=c[i])
        canvas.draw()
        
class Application(tk.Frame):
    def __init__(self, root=None):
        tk.Frame.__init__(self,root)
        self.root = root
        self.createWidgets()

    def createWidgets(self):
        fig=plt.figure(figsize=(2,2))
        ax=fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
        canvas=FigureCanvasTkAgg(fig,master=self.root)
        canvas.get_tk_widget().grid(row=0,column=1)
        canvas.draw()
        self.thread = threading.Thread(target=process, args=(canvas,ax))
        self.thread.setDaemon(True)
        self.thread.start()
        self.plotbutton=tk.Button(master=self.root, text="Quit", command=self._quit)
        self.plotbutton.grid(row=0,column=0)
    
    def _quit(self):
        self.root.quit()


root=tk.Tk()
root.title("Welcome to GeekForGeeks")
root.geometry('550x450')
app=Application(root=root)
app.mainloop()