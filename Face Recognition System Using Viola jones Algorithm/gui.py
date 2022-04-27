import os
import ast
import cv2
import numpy
import sqlite3
import datetime
import face_recognition
from tkinter import *

def insert(persons):
    connection = sqlite3.connect("Face_Recognition.db")
    cursor = connection.cursor()
    txt = "INSERT INTO Information VALUES ('"+str(persons)+"','"+str(datetime.datetime.now())+"','"+"Appeared')"
    cursor.execute(txt)
    connection.commit()
    connection.close()

def view():
    view = Tk()
    view.geometry('330x700')
    view.title("Database")
    connection = sqlite3.connect("Face_Recognition.db")
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM Information")
    rows = cursor.fetchall()
    connection.close()
    label = Label(view, text='ID                    Date & Time                 Event')
    label.pack()
    scrollbar = Scrollbar(view)
    scrollbar.pack(side=RIGHT, fill=Y)
    mylist = Listbox(view, yscrollcommand=scrollbar.set,width = 42,height = 38)
    mylist.insert(END, '----------------------------------------------------------')
    for items in rows:
        items = '    ' + str(items[0])+ '    ' + str(items[1]) + '    ' + str(items[2])
        mylist.insert(END, items)
        mylist.insert(END, '-----------------------------------------------------------')

    mylist.pack()
    scrollbar.config(command=mylist.yview)

    view.resizable(False, False)
    view.mainloop()

def face_recogition():
    directory_name = path.get()
    if directory_name == '':
        return

    imgtest = face_recognition.load_image_file(directory_name)
    imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)
    encodetest = face_recognition.face_encodings(imgtest)[0]
    person = 0
    for files in os.listdir('D:/Face Recognition/'):
        if files.endswith('txt'):
            person = person + 1

    for persons in range(person):
        File_object = open('d:/Face Recognition/' + str(persons) + '.txt', 'r')
        result = ast.literal_eval(File_object.read())
        File_object.close()

        result = numpy.asarray(result)

        results = face_recognition.compare_faces([result], encodetest, 0.4)[0]

        if (results):
            print(f'The the person is of ID =  {persons}.')
            insert(persons)
            view()
            break

    if (~results):
        print('Person not found.')
        pass

def import_pic():
    directory_name = path.get()
    if directory_name == '':
        return
    directory_name = path.get() + "\ "
    directory_name = directory_name[:len(directory_name)-1]

    persons = 0
    for files in os.listdir('D:/Face Recognition/'):
        if files.endswith('txt'):
           persons = persons + 1

    result = []
    for j in range(128):
        result.append(0.0)

    ext = ('jpg','png')
    pic_count = 0
    for files in os.listdir(directory_name):
        if files.endswith(ext):
            pic_count = pic_count + 1
            img = face_recognition.load_image_file(directory_name + files)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            encode = face_recognition.face_encodings(img)[0].tolist()

            for j in range(128):
                result[j] = result[j] + encode[j]

    if 1 < pic_count:
         for j in range(128):
            result[j] = result[j] / pic_count

    File_object = open('d:/Face Recognition/' + str(persons) + '.txt', 'w')
    File_object.write(str(result))
    File_object.close()

if not os.path.exists('D:\Face Recognition'):
    os.makedirs('D:\Face Recognition')

persons = 0
connection = sqlite3.connect("Face_Recognition.db")
cursor = connection.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS Information (ID TEXT, DateTime TEXT, Event TEXT)")
connection.commit()
connection.close()

window = Tk()

window.title("Face Recognition")

window.geometry('400x95')

path = Entry(window, width = 51, bg="silver", borderwidth=3, font = ('bold'), fg="dark green")
path.pack()

Import = Button(window, text="Import", font=(None,12), bg="LightGreen", fg="RoyalBlue", command = import_pic, padx=5, pady=10)
Import.place(x=3, y=37)

Face_Recogition = Button(window, text="Face Recognition", font=(None,12), bg="LightPink", fg="Brown", command = face_recogition, padx=50, pady=10)
Face_Recogition.place(x=79, y=37)

Export = Button(window, text="View", font=(None,12), bg="NavajoWhite", fg="RoyalBlue", command = view, padx=5, pady=10)
Export.place(x=337, y=37)
window.resizable(False, False)
window.mainloop()

