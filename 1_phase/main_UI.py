import PySimpleGUI as sg
import os.path
import io
from PIL import Image
from main_func import classify


file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Text(size=(40, 1), key="-STATE-")],
    [sg.Image(key="-IMAGE-")],  # only png,gif
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Malaria Detection", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))

        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            image = Image.open(filename)
            pred = classify(filename)
            image.thumbnail((256, 256))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            
            
            if pred == 1:
                state = 'Infected'
            else:
                state = 'Uninfected'

            window["-TOUT-"].update(["-FILE LIST-"][0])
            window["-STATE-"].update(state)
            window["-IMAGE-"].update(data=bio.getvalue())
        except:
            pass
window.refresh()
window.close()
