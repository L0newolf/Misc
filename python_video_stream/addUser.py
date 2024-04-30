#!/usr/bin/python

from tkinter import *
import sqlite3

root = Tk()
root.title("Add User to Database")
width = 400
height = 280
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
root.resizable(0, 0)

# ==============================METHODS========================================


def Database():
    global conn, cursor
    conn = sqlite3.connect("userDB.db")
    cursor = conn.cursor()


def AddUser(event=None):
    Database()

    if USERNAME.get() == "" or PASSWORD.get() == "":
        lbl_text.config(text="Please complete the required fields!", fg="red")
    else:
    	cursor.execute("SELECT * FROM `userDB` WHERE `ID` = ?", (USERNAME.get()))
        if cursor.fetchone() is not None:
    		lbl_text.config(text="User already exists", fg="red")
            USERNAME.set("")
            PASSWORD.set("")
            EMAIL.set("")
            USERID.set("")
            ACCESSLEVEL.set("")
        else:
            cursor.execute("INSERT INTO `userDB` (ID, email, password,userName,accessLevel) VALUES (?, ?, ?,?,?)", (USERID.get(), EMAIL.get(), PASSWORD.get(), USERNAME.get(), ACCESSLEVEL.get()))
            cursor.execute("SELECT * FROM `userDB` WHERE `ID` = ?", (USERNAME.get()))
            if cursor.fetchone() is not None:
            	HomeWindow()
            	USERNAME.set("")
            	PASSWORD.set("")
            	EMAIL.set("")
            	USERID.set("")
            	ACCESSLEVEL.set("")
            	lbl_text.config(text="")
            else:
            	lbl_text.config(text="Failed to add user, Please try again", fg="red")
            	USERNAME.set("")
            	PASSWORD.set("")
            	EMAIL.set("")
            	USERID.set("")
            	ACCESSLEVEL.set("")

def HomeWindow():
    global Home
    root.withdraw()
    Home = Toplevel()
    Home.title("Python: Simple Login Application")
    width = 600
    height = 500
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.resizable(0, 0)
    Home.geometry("%dx%d+%d+%d" % (width, height, x, y))
    lbl_home = Label(Home, text="Successfully Login!", font=('times new roman', 20)).pack()
    btn_back = Button(Home, text='Back', command=Back).pack(pady=20, fill=X)

def Back():
    Home.destroy()
    root.deiconify()

# ==============================VARIABLES======================================
USERNAME = StringVar()
PASSWORD = StringVar()
EMAIL = StringVar()
USERID = StringVar()
ACCESSLEVEL = StringVar()

# ==============================FRAMES=========================================
Top = Frame(root, bd=2,  relief=RIDGE)
Top.pack(side=TOP, fill=X)
Form = Frame(root, height=200)
Form.pack(side=TOP, pady=20)

# ==============================LABELS=========================================
lbl_title = Label(Top, text = "Add user Application", font=('arial', 15))
lbl_title.pack(fill=X)

lbl_username = Label(Form, text = "Username:", font=('arial', 14), bd=15)
lbl_username.grid(row=0, sticky="e")
lbl_password = Label(Form, text = "Password:", font=('arial', 14), bd=15)
lbl_password.grid(row=1, sticky="e")
lbl_email = Label(Form, text = "Email:", font=('arial', 14), bd=15)
lbl_email.grid(row=1, sticky="e")
lbl_userid = Label(Form, text = "User Id:", font=('arial', 14), bd=15)
lbl_userid.grid(row=1, sticky="e")
lbl_accesslevel = Label(Form, text = "Access Level:", font=('arial', 14), bd=15)
lbl_accesslevel.grid(row=1, sticky="e")

lbl_text = Label(Form)
lbl_text.grid(row=2, columnspan=2)

# ==============================ENTRY WIDGETS==================================
username = Entry(Form, textvariable=USERNAME, font=(14))
username.grid(row=0, column=1)
password = Entry(Form, textvariable=PASSWORD, show="*", font=(14))
password.grid(row=1, column=1)
email = Entry(Form, textvariable=EMAIL, show="*", font=(14))
email.grid(row=1, column=1)
userid = Entry(Form, textvariable=USERID, show="*", font=(14))
userid.grid(row=1, column=1)
accesslevel = Entry(Form, textvariable=ACCESSLEVEL, show="*", font=(14))
accesslevel.grid(row=1, column=1)

# ==============================BUTTON WIDGETS=================================
btn_login = Button(Form, text="Add User", width=45, command=AddUser)
btn_login.grid(pady=25, row=3, columnspan=2)
btn_login.bind('<Return>', AddUser)

# ==============================INITIALIATION==================================
if __name__ == '__main__':
    root.mainloop()                	
