from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from AutoMerge import find_matching_frames


class App:

    def __init__(self, master):

        Label(master, text="Leading:").grid(row=0, column=0, sticky=W)

        self.lead_entry = Entry(master, width=100, state='readonly')
        self.lead_entry.grid(row=0, column=1)

        self.lead_add_button = Button(master, text="+", width=3, command=self.browse_lead)
        self.lead_add_button.grid(row=0, column=2, padx=5, pady=5, sticky=NW)

        Label(master, text="Following:").grid(row=1, column=0, sticky=NW)

        self.following_listbox = Listbox(master, selectmode=SINGLE, width=100)
        self.following_listbox.grid(row=1, column=1, rowspan=2)

        self.following_add_button = Button(master, text="+", width=3, command=self.browse_following)
        self.following_add_button.grid(row=1, column=2, padx=5, pady=5, sticky=NW)

        self.following_sub_button = Button(master, text="-", width=3,
                                           command=lambda sub=self.following_listbox: self.following_listbox.delete(ANCHOR))
        self.following_sub_button.grid(row=2, column=2, padx=5, pady=5, sticky=NW)

        self.go = Button(master, text="GO!", command=self.go)
        self.go.grid(row=3, columnspan=3, sticky=E+W, padx=5, pady=5)

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
        l = list(self.following_listbox.get(0, END))
        lead = self.lead_entry.get()
        if lead == "":
            print("najj!")
        else:
            print(lead)

    def go(self):
        leading_vid = self.lead_entry.get()
        following_vids = list(self.following_listbox.get(0, END))

        if following_vids == []:
            messagebox.showwarning(
                "Bad input",
                "Add at least one following video."
            )
            return

        if leading_vid == "":
            messagebox.showwarning(
                "Bad input",
                "Add a leading video."
            )
            return

        result = find_matching_frames(leading_vid, following_vids, seconds=2, multichannel=False, method='mse')
        print(result)
        return

root = Tk()

app = App(root)

root.mainloop()
#root.destroy() # optional; see description below


