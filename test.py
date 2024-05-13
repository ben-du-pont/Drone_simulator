self.cidkey = self.point.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        if event.key == 'ctrl+c':
            plt.close(self.point.figure)  # Close the plot