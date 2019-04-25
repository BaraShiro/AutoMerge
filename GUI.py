from tkinter import *
from tkinter import filedialog
from AutoMerge import find_matching_frames


class App:

    def __init__(self, master):

        frame = Frame(master)
        frame.pack()

        self.first_file = ""
        self.second_file = ""

        self.button = Button(frame, text="QUIT", fg="red", command=frame.quit)
        self.button.pack(side=LEFT)

        self.bfst = Button(frame, text="First", command=self.browse_fst)
        self.bfst.pack(side=LEFT)

        self.bsnd = Button(frame, text="Second", command=self.browse_snd)
        self.bsnd.pack(side=LEFT)

        self.go = Button(frame, text="GO!", command=self.go)
        self.go.pack(side=LEFT)

    def browse_fst(self):
        print(self.first_file)
        self.first_file = filedialog.askopenfilename(title="Select file",
                                                   filetypes=(("Video files", "*.avi *.mp4 *.mkv"), ("all files", "*.*")))
        print(self.first_file)


    def browse_snd(self):
        print(self.second_file)
        self.second_file = filedialog.askopenfilename(title="Select file",
                                                   filetypes=(("Video files", "*.avi *.mp4 *.mkv"), ("all files", "*.*")))
        print(self.second_file)

    def go(self):
        if self.first_file == "" or self.second_file == "":
            print("Select two valid files!")
            return
        else:
            result = find_matching_frames(self.first_file, [self.second_file], seconds=2, multichannel=False, method='mse')
            print(result)
            return

root = Tk()

app = App(root)

root.mainloop()
#root.destroy() # optional; see description below


