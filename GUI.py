from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from AutoMerge import find_matching_frames


class App:

    def __init__(self, window):

        # Variables to be passed to find_matching_frames
        self.colour = BooleanVar()
        self.resize = BooleanVar()
        self.mode = StringVar()
        self.seconds = IntVar()

        # The main frame, mainly for the padding
        self.main = Frame(window, padx=10, pady=10)
        self.main.grid(row=0, column=0)

        # The leading file selection frame
        self.leading_file_frame = LabelFrame(self.main, text="Leading file", padx=5, pady=5)
        self.leading_file_frame.grid(row=0, column=0, sticky=NW+E)

        # Separator
        Frame(self.main, height=10, padx=10, pady=10).grid(row=1, column=1)

        # The following file selection frame
        self.following_files_frame = LabelFrame(self.main, text="Following files", padx=5, pady=5)
        self.following_files_frame.grid(row=2, column=0, sticky=NW)

        # Separator
        Frame(self.main, width=10, padx=10, pady=10).grid(row=0, column=1)

        # The options frame
        self.options_frame = LabelFrame(self.main, text="Options", padx=5, pady=5)
        self.options_frame.grid(row=0, column=2, rowspan=3, sticky=NW)

        # The go button
        self.go = Button(self.main, text="Go!", width=10, command=self.go)
        self.go.grid(row=3, column=2, sticky=NE)

        # Widgets that go into the leading file selection frame
        self.lead_entry = Entry(self.leading_file_frame, width=100, state='readonly')
        self.lead_entry.grid(row=0, column=0)

        self.lead_add_button = Button(self.leading_file_frame, text="...", width=3, command=self.browse_lead)
        self.lead_add_button.grid(row=0, column=1, padx=5, pady=5, sticky=NW)

        # Widgets that go into the following files selection frame
        self.following_scrollbar = Scrollbar(self.following_files_frame, orient=VERTICAL)
        self.following_scrollbar.grid(row=0, column=1, rowspan=3, sticky=N+S)

        self.following_listbox = Listbox(self.following_files_frame, selectmode=SINGLE, width=100, yscrollcommand=self.following_scrollbar.set)
        self.following_listbox.grid(row=0, column=0, rowspan=3)

        self.following_scrollbar.config(command=self.following_listbox.yview)  # Connect the scrollbar to the listbox

        self.following_add_button = Button(self.following_files_frame, text="+", width=3, command=self.browse_following)
        self.following_add_button.grid(row=0, column=2, padx=5, pady=5, sticky=NW)

        self.following_sub_button = Button(self.following_files_frame, text="-", width=3,
                                           command=lambda sub=self.following_listbox: self.following_listbox.delete(ANCHOR))
        self.following_sub_button.grid(row=1, column=2, padx=5, pady=5, sticky=NW)

        # Widgets that go into the options frame
        self.colour_check = Checkbutton(self.options_frame, text="Colour", variable=self.colour)
        self.colour_check.grid(row=0, column=0)

        self.resize_check = Checkbutton(self.options_frame, text="Downscale", variable=self.resize)
        self.resize_check.grid(row=0, column=1)

        Label(self.options_frame, text="Algorithm:").grid(row=1, column=0, sticky=NW)

        self.mse_radio_button = Radiobutton(self.options_frame, text="MSE", variable=self.mode, value="mse")
        self.mse_radio_button.grid(row=2, column=0,sticky=NW)
        self.mse_radio_button.select()

        self.nrmse_radio_button = Radiobutton(self.options_frame, text="NRMSE", variable=self.mode, value="nrmse")
        self.nrmse_radio_button.grid(row=2, column=1, sticky=NW)

        self.psnr_radio_button = Radiobutton(self.options_frame, text="PSNR", variable=self.mode, value="psnr")
        self.psnr_radio_button.grid(row=3, column=0, sticky=NW)

        self.ssim_radio_button = Radiobutton(self.options_frame, text="SSIM", variable=self.mode, value="ssim")
        self.ssim_radio_button.grid(row=3, column=1, sticky=NW)

        Label(self.options_frame, text="Seconds:").grid(row=4, column=0, sticky=NW)

        self.seconds_spinbox = Spinbox(self.options_frame, from_=1, to=10, textvariable=self.seconds, state='readonly')
        self.seconds_spinbox.grid(row=5, column=0, columnspan=2, sticky=NW)
        self.seconds.set(3)

    def browse_lead(self):
        file_path = filedialog.askopenfilename(title="Select file",
                                                   filetypes=(("Video files", "*.avi *.mp4 *m4v *.mkv *.mov"), ("All files", "*.*")))

        if file_path:  # file_path will be the empty string if the user cancels the dialog window
            self.lead_entry.configure(state='normal')       # Make entry writeable
            self.lead_entry.delete(0, END)                  # Delete entire string
            self.lead_entry.insert(0, file_path)            # Put file_path as the string
            self.lead_entry.configure(state='readonly')     # Make entry read only again

    def browse_following(self):
        file_path = filedialog.askopenfilename(title="Select file",
                                                   filetypes=(("Video files", "*.avi *.mp4 *m4v *.mkv *.mov"), ("All files", "*.*")))

        if file_path:  # file_path will be the empty string if the user cancels the dialog window
            self.following_listbox.insert(END, file_path)

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
                                      multichannel=self.colour.get(), downscale=self.resize.get(),
                                      method=self.mode.get(), verbose=3)
        print(result)  # TODO: present result in a better way. Maybe write to file?
        return


main_window = Tk()  # Create main window
main_window.title("AutoMerge")  # Give the main window a name

app = App(main_window)  # Create the App object

main_window.mainloop()  # Start
# main_window.destroy()  # Maybe needed on Linux for graceful exit, throws an exception on Windows
