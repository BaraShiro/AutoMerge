from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from AutoMerge import find_matching_frames


class App:

    def __init__(self, master):

        self.colour = BooleanVar()
        self.mode = StringVar()
        self.seconds = IntVar()

        Label(master, text="Leading:").grid(row=0, column=0, sticky=W)

        self.lead_entry = Entry(master, width=100, state='readonly')
        self.lead_entry.grid(row=0, column=1)

        self.lead_add_button = Button(master, text="+", width=3, command=self.browse_lead)
        self.lead_add_button.grid(row=0, column=3, padx=5, pady=5, sticky=NW)

        Label(master, text="Following:").grid(row=1, column=0, sticky=NW)

        self.following_scrollbar = Scrollbar(master, orient=VERTICAL)
        self.following_scrollbar.grid(row=1, column=2, rowspan=3, sticky=N+S)

        self.following_listbox = Listbox(master, selectmode=SINGLE, width=100, yscrollcommand=self.following_scrollbar.set)
        self.following_listbox.grid(row=1, column=1, rowspan=3)

        self.following_scrollbar.config(command=self.following_listbox.yview)

        self.following_add_button = Button(master, text="+", width=3, command=self.browse_following)
        self.following_add_button.grid(row=1, column=3, padx=5, pady=5, sticky=NW)

        self.following_sub_button = Button(master, text="-", width=3,
                                           command=lambda sub=self.following_listbox: self.following_listbox.delete(ANCHOR))
        self.following_sub_button.grid(row=2, column=3, padx=5, pady=5, sticky=NW)

        self.go = Button(master, text="Go!", command=self.go)
        self.go.grid(row=4, columnspan=2, sticky=E+W, padx=5, pady=5)

        self.test_button = Button(master, text="Test", command=self.test)
        self.test_button.grid(row=4, column=3, sticky=E+W, padx=5, pady=5)

        self.colour_check = Checkbutton(master, text="Colour", variable=self.colour)
        self.colour_check.grid(row=0, column=4)

        self.mse_radio_button = Radiobutton(master, text="MSE", variable=self.mode, value="mse")
        self.mse_radio_button.grid(row=1, column=4,sticky=NW)
        self.mse_radio_button.select()

        self.nrmse_radio_button = Radiobutton(master, text="NRMSE", variable=self.mode, value="nrmse")
        self.nrmse_radio_button.grid(row=1, column=5, sticky=NW)

        self.psnr_radio_button = Radiobutton(master, text="PSNR", variable=self.mode, value="psnr")
        self.psnr_radio_button.grid(row=2, column=4, sticky=NW)

        self.ssim_radio_button = Radiobutton(master, text="SSIM", variable=self.mode, value="ssim")
        self.ssim_radio_button.grid(row=2, column=5, sticky=NW)

        self.seconds_spinbox = Spinbox(master, from_=1, to=10, textvariable=self.seconds, state='readonly')
        self.seconds_spinbox.grid(row=3, column=4, columnspan=2, sticky=NW)
        self.seconds.set(3)


    def browse_lead(self):
        file_path = filedialog.askopenfilename(title="Select file",
                                                   filetypes=(("Video files", "*.avi *.mp4 *.mkv"), ("all files", "*.*")))
        self.lead_entry.configure(state='normal')
        self.lead_entry.delete(0, END)
        self.lead_entry.insert(0, file_path)
        self.lead_entry.configure(state='readonly')

    def browse_following(self):
        file_path = filedialog.askopenfilename(title="Select file",
                                                   filetypes=(("Video files", "*.avi *.mp4 *.mkv"), ("all files", "*.*")))
        self.following_listbox.insert(END, file_path)

    def test(self):
        print(type(self.seconds.get()))

    def go(self):
        leading_vid = self.lead_entry.get()
        following_vids = list(self.following_listbox.get(0, END))

        if leading_vid == "":
            messagebox.showwarning(
                "Bad input",
                "Add a leading video."
            )
            return

        if following_vids == []:
            messagebox.showwarning(
                "Bad input",
                "Add at least one following video."
            )
            return

        # TODO: Put in try block to catch exceptions (wrong filetype, non existing file, etc.)
        result = find_matching_frames(leading_vid, following_vids, seconds=self.seconds.get(),
                                      multichannel=self.colour.get(), method=self.mode.get())
        print(result)  # TODO: present result in a better way. Maybe write to file?
        return


root = Tk()  # Create main window
root.title("AutoMerge")  # Give the main window a name

app = App(root)  # Create the App object

root.mainloop()  # Start
# root.destroy()  # Maybe needed on Linux for graceful exit, throws an exception on Windows


