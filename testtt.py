import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import *
from PIL import Image, ImageFilter, ImageTk
import os


import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model

model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    global result
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatres(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

PATH = os.path.dirname(os.path.realpath(__file__))


class App(customtkinter.CTk):
    APP_NAME = "LOGIN"
    WIDTH = 800
    HEIGHT = 520

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title(App.APP_NAME)
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.minsize(App.WIDTH, App.HEIGHT)
        self.maxsize(App.WIDTH, App.HEIGHT)
        self.resizable(False, False)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # load image with PIL and convert to PhotoImage
        self.image1 = Image.open(PATH + "/test_images/galaxy.jpg").resize((self.WIDTH + 200, self.HEIGHT + 130))
        self.image = self.image1.filter(ImageFilter.BLUR)
        self.bg_image = ImageTk.PhotoImage(self.image)

        self.image_label = tkinter.Label(master=self, image=self.bg_image)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=300,
                                            height=500,
                                            corner_radius=10, bg_color=("black", "#190F49"))
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.image2 = Image.open(PATH + "/test_images/blank-profile-picture-973460__340.jpg").resize(
            (self.WIDTH - 700, self.HEIGHT - 420))
        self.logo_image = ImageTk.PhotoImage(self.image2)
        self.image_label1 = tkinter.Label(master=self.frame, image=self.logo_image)
        self.image_label1.place(relx=0.5, rely=0.3, anchor=tkinter.CENTER)

        self.label_1 = customtkinter.CTkLabel(master=self.frame, width=200, height=60,
                                              text_font=("Comic Sans MS", 10, "bold"),
                                              text="WELCOME TO\nSTRESS MANAGEMENT\nCHATBOT!!")
        self.label_1.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)

        self.entry_1 = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=200,
                                              placeholder_text="Username/email", fg_color="#BFE8D4",
                                              text_color="black")

        self.entry_1.place(relx=0.5, rely=0.52, anchor=tkinter.CENTER)

        self.entry_2 = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=200, show="*",
                                              placeholder_text="Password", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_2.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER)

        self.button_2 = customtkinter.CTkButton(master=self.frame, text="Login",
                                                corner_radius=10, command=self.login_portal, width=200, height=40)
        self.button_2.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)

        self.button_3 = customtkinter.CTkButton(master=self.frame, text="Forgot password",
                                                corner_radius=10, command=self.callback1, width=160,
                                                fg_color="#589C9C")
        self.button_3.place(relx=0.5, rely=0.85, anchor=tkinter.N)

        self.button_4 = customtkinter.CTkButton(master=self.frame, text="Create account",
                                                corner_radius=10, command=self.callback, width=160,
                                                fg_color="#589C9C")
        self.button_4.place(relx=0.5, rely=0.92, anchor=tkinter.N)

        self.button_3 = customtkinter.CTkButton(master=self, text="ABOUT",
                                                corner_radius=10, command=self.callback11, width=80, height=20,
                                                fg_color="#6A34DD")
        self.button_3.place(relx=0.88, rely=0.94)

    def button_event(self):
        self.image1 = Image.open(PATH + "/test_images/galaxy.jpg").resize((self.WIDTH + 200, self.HEIGHT + 130))
        self.image = self.image1.filter(ImageFilter.BLUR)
        self.bg_image = ImageTk.PhotoImage(self.image)

        self.image_label = tkinter.Label(master=self, image=self.bg_image)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=300,
                                            height=500,
                                            corner_radius=10, bg_color=("black", "#190F49"))
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.image2 = Image.open(PATH + "/test_images/blank-profile-picture-973460__340.jpg").resize(
            (self.WIDTH - 700, self.HEIGHT - 420))
        self.logo_image = ImageTk.PhotoImage(self.image2)
        self.image_label1 = tkinter.Label(master=self.frame, image=self.logo_image)
        self.image_label1.place(relx=0.5, rely=0.3, anchor=tkinter.CENTER)

        self.label_1 = customtkinter.CTkLabel(master=self.frame, width=200, height=60,
                                              text_font=("Comic Sans MS", 10, "bold"),
                                              text="WELCOME TO\nSTRESS MANAGEMENT\nCHATBOT!!")
        self.label_1.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)

        self.entry_1 = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=200,
                                              placeholder_text="Username/email", fg_color="#BFE8D4",
                                              text_color="black")

        self.entry_1.place(relx=0.5, rely=0.52, anchor=tkinter.CENTER)

        self.entry_2 = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=200, show="*",
                                              placeholder_text="Password", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_2.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER)

        self.button_2 = customtkinter.CTkButton(master=self.frame, text="Login",
                                                corner_radius=10, command=self.login_portal, width=200, height=40)
        self.button_2.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)

        self.button_3 = customtkinter.CTkButton(master=self.frame, text="Forgot password",
                                                corner_radius=10, command=self.callback1, width=160,
                                                fg_color="#589C9C")
        self.button_3.place(relx=0.5, rely=0.85, anchor=tkinter.N)

        self.button_4 = customtkinter.CTkButton(master=self.frame, text="Create account",
                                                corner_radius=10, command=self.callback, width=160,
                                                fg_color="#589C9C")
        self.button_4.place(relx=0.5, rely=0.92, anchor=tkinter.N)

        self.button_3 = customtkinter.CTkButton(master=self, text="ABOUT",
                                                corner_radius=10, command=self.callback11, width=80, height=20,
                                                fg_color="#6A34DD")
        self.button_3.place(relx=0.88, rely=0.94)

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()

    def callback(self):
        self.image1 = Image.open(PATH + "/test_images/galaxy.jpg").resize((self.WIDTH + 200, self.HEIGHT + 130))
        self.image = self.image1.filter(ImageFilter.BLUR)
        self.bg_image = ImageTk.PhotoImage(self.image)

        self.image_label = tkinter.Label(master=self, image=self.bg_image)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.label_1 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("Comic Sans MS", 10, "bold"),
                                              text="CREATE YOUR PROFILE!!", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_1.place(relx=0.5, rely=0.025, anchor=tkinter.CENTER)

        self.label_u = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="WHAT IS YOUR USERNAME", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_u.place(relx=0.015, rely=0.08)
        self.entry_1 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="Username", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_1.place(relx=0.28, rely=0.08)

        self.label_2 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="TYPE YOUR EMAIL", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_2.place(relx=0.015, rely=0.2)

        self.entry_1 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="Email", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_1.place(relx=0.28, rely=0.2)

        self.label_3 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="YOUR FIRST NAME", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_3.place(relx=0.015, rely=0.32)

        self.entry_2 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="First name", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_2.place(relx=0.28, rely=0.32)

        self.label_4 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="YOUR LAST NAME", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_4.place(relx=0.015, rely=0.44)

        self.entry_3 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="Last name", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_3.place(relx=0.28, rely=0.44)

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self, values=["Male", "Female", "Others"],
                                                        fg_color="#232A69", bg_color=("black", "#190F49")
                                                        )
        self.optionmenu_1.grid(pady=10, padx=20, sticky="w")
        self.optionmenu_1.place(relx=0.28, rely=0.56)

        self.label_5 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="WHAT IS YOUR GENDER", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_5.place(relx=0.015, rely=0.56)

        self.entry_5 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="Password", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_5.place(relx=0.28, rely=0.68)

        self.label_6 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="SET YOUR PASSWORD", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_6.place(relx=0.015, rely=0.68)

        self.button_1 = customtkinter.CTkButton(master=self, text="CREATE ACCOUNT",
                                                corner_radius=10, command=self.button_event, width=200, height=40)
        self.button_1.place(relx=0.28, rely=0.8)

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=300,
                                            height=400,
                                            corner_radius=10, bg_color=("black", "#190F49"))
        self.frame.place(relx=0.6, rely=0.1)
        self.label_c = customtkinter.CTkLabel(master=self.frame, width=200, height=20,
                                              text_font=("Arial", 10, "normal"),
                                              text="I'm not a robot", text_color="black", corner_radius=10,
                                              fg_color="white",
                                              )
        self.label_c.place(relx=0.165, rely=0.05)

        self.frames = customtkinter.CTkFrame(master=self.frame,
                                             width=200,
                                             height=200,
                                             corner_radius=10)
        self.frames.place(relx=0.165, rely=0.2)
        self.entry_6 = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=150,
                                              placeholder_text="Enter the text in the above image", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_6.place(relx=0.165, rely=0.8)

        self.button_2 = customtkinter.CTkButton(master=self.frame, text="CLICK",
                                                corner_radius=10, command=self.button_event, width=50, height=50)
        self.button_2.place(relx=0.72, rely=0.78)
        # load image with PIL and convert to PhotoImage
        self.imagere = Image.open(PATH + "/test_images/recaptcha.jpg").resize((self.WIDTH - 700, self.HEIGHT - 420))
        self.bg_imagere = ImageTk.PhotoImage(self.imagere)
        self.image_labelre = tkinter.Label(master=self.button_2, image=self.bg_imagere)
        self.image_labelre.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        # load image with PIL and convert to PhotoImage
        def imagcc():
            import random
            list_images = ["smwm.jpg", "7364.jpg"]
            return random.choice(list_images)

        self.imagec = Image.open(PATH + "/test_images/" + imagcc()).resize((self.WIDTH - 590, self.HEIGHT - 300))
        self.bg_imagec = ImageTk.PhotoImage(self.imagec)
        self.image_labelc = tkinter.Label(master=self.frames, image=self.bg_imagec)
        self.image_labelc.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.button_3 = customtkinter.CTkButton(master=self, text="BACK",
                                                corner_radius=10, command=self.button_event, width=80, height=20,
                                                fg_color="#6A34DD")
        self.button_3.place(relx=0.01, rely=0.01)

    def callback1(self):
        self.image1 = Image.open(PATH + "/test_images/galaxy.jpg").resize((self.WIDTH + 200, self.HEIGHT + 130))
        self.image = self.image1.filter(ImageFilter.BLUR)
        self.bg_image = ImageTk.PhotoImage(self.image)

        self.image_label = tkinter.Label(master=self, image=self.bg_image)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.label_1 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("Comic Sans MS", 10, "bold"),
                                              text="RESET YOUR PASSWORD", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_1.place(relx=0.5, rely=0.025, anchor=tkinter.CENTER)

        self.label_u = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="USERNAME", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_u.place(relx=0.015, rely=0.08)
        self.entry_1 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="Username", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_1.place(relx=0.28, rely=0.08)

        self.label_2 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="EMAIL", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_2.place(relx=0.015, rely=0.2)

        self.entry_1 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="Email", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_1.place(relx=0.28, rely=0.2)

        self.label_3 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="FIRST NAME", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_3.place(relx=0.015, rely=0.32)

        self.entry_2 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="First name", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_2.place(relx=0.28, rely=0.32)

        self.label_4 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="LAST NAME", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_4.place(relx=0.015, rely=0.44)

        self.entry_3 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="Last name", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_3.place(relx=0.28, rely=0.44)

        self.entry_5 = customtkinter.CTkEntry(master=self, corner_radius=6, width=200,
                                              placeholder_text="Password", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_5.place(relx=0.28, rely=0.56)

        self.label_6 = customtkinter.CTkLabel(master=self, width=200, height=20,
                                              text_font=("VERDANA", 10, "italic"),
                                              text="YOUR NEW PASSWORD", corner_radius=10, fg_color="#212A74",
                                              bg_color=("black", "#190F49"))
        self.label_6.place(relx=0.015, rely=0.56)

        self.button_1 = customtkinter.CTkButton(master=self, text="RESET PASSWORD",
                                                corner_radius=10, command=self.button_event, width=200, height=40)
        self.button_1.place(relx=0.28, rely=0.68)

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=300,
                                            height=400,
                                            corner_radius=10, bg_color=("black", "#190F49"))
        self.frame.place(relx=0.6, rely=0.1)
        self.label_c = customtkinter.CTkLabel(master=self.frame, width=200, height=20,
                                              text_font=("Arial", 10, "normal"),
                                              text="I'm not a robot", text_color="black", corner_radius=10,
                                              fg_color="white",
                                              )
        self.label_c.place(relx=0.165, rely=0.05)

        self.frames = customtkinter.CTkFrame(master=self.frame,
                                             width=200,
                                             height=200,
                                             corner_radius=10)
        self.frames.place(relx=0.165, rely=0.2)
        self.entry_6 = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=200,
                                              placeholder_text="Enter the text in the above image", fg_color="#BFE8D4",
                                              text_color="black")
        self.entry_6.place(relx=0.165, rely=0.8)

        # load image with PIL and convert to PhotoImage
        def imagcc():
            import random
            list_images = ["smwm.jpg", "7364.jpg", "avoid-captcha.jpg"]
            return random.choice(list_images)

        self.imagec = Image.open(PATH + "/test_images/" + imagcc()).resize((self.WIDTH - 590, self.HEIGHT - 300))
        self.bg_imagec = ImageTk.PhotoImage(self.imagec)
        self.image_labelc = tkinter.Label(master=self.frames, image=self.bg_imagec)
        self.image_labelc.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.button_3 = customtkinter.CTkButton(master=self, text="BACK",
                                                corner_radius=10, command=self.button_event, width=80, height=20,
                                                fg_color="#6A34DD")
        self.button_3.place(relx=0.01, rely=0.01)

    def callback11(self):
        self.image1 = Image.open(PATH + "/test_images/galaxy.jpg").resize((self.WIDTH + 200, self.HEIGHT + 130))
        self.image = self.image1.filter(ImageFilter.BLUR)
        self.bg_image = ImageTk.PhotoImage(self.image)

        self.image_label = tkinter.Label(master=self, image=self.bg_image)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=550,
                                            height=800,
                                            corner_radius=10, bg_color=("black", "#190F49"))
        self.frame.place(relx=0.5, rely=0.2, anchor=tkinter.CENTER)
        self.label_2 = customtkinter.CTkLabel(master=self.frame, width=200,
                                              text_font=("Comic Sans MS", 20, "bold"), text_color="#44B6BD",
                                              text="ABOUT\n")
        self.label_2.place(relx=0.5, rely=0.42, anchor=tkinter.CENTER)
        self.label_1 = customtkinter.CTkLabel(master=self.frame, width=200,
                                              text_font=("Comic Sans MS", 10, "bold"),
                                              text="People of today's era are invariably subjected to immense\n"
                                                   "amounts of stress, and there are plenty of factors contributing.\n"
                                                   "Many are unable to cope with the challenges and stressful environment\n"
                                                   "and fail to receive help in the right way, thus leading to persistent\n"
                                                   "damage to their lifestyles. This is sometimes followed by performance\n"
                                                   "degradation of the individual such as a student and not being counted\n"
                                                   "as an asset. The solution we present is a simple chatbot that can\n"
                                                   "communicate with the user and through a series of questions presented\n"
                                                   "to the user the chatbot can predict whether the user is stressed, the\n"
                                                   "level of stress, and ways to cope with stress. Based on the range of \n"
                                                   "the level and the probabilistic parameters of stress, each individual\n"
                                                   "is given feedback and an advisable solution. The chatbot is to be made\n"
                                                   "using python & will consist of advanced technologies that enable it to\n"
                                                   "predict with higher accuracies. The individual may adapt the solution \n"
                                                   "and make way for his or her mental peace thereby reducing stress levels.\n")
        self.label_1.place(relx=0.5, rely=0.64, anchor=tkinter.CENTER)

        self.label_3 = customtkinter.CTkLabel(master=self.frame, width=200,
                                              text_font=("Comic Sans MS", 12, "bold"), text_color="#44B6BD",
                                              text="OUR TEAM MEMBERS\n ")
        self.label_3.place(relx=0.8, rely=0.858, anchor=tkinter.CENTER)
        self.label_4 = customtkinter.CTkLabel(master=self.frame, width=200,
                                              text_font=("Comic Sans MS", 12, "bold"), text_color="#6A34DD",
                                              text="Aditya shaji \n"
                                                   "Athulkrishna Prakash\n"
                                                   "Balasurya K B\n"
                                                   "Sreeraj M J")
        self.label_4.place(relx=0.8, rely=0.92, anchor=tkinter.CENTER)

        self.button_3 = customtkinter.CTkButton(master=self, text="BACK",
                                                corner_radius=10, command=self.button_event, width=80, height=20,
                                                fg_color="#6A34DD")
        self.button_3.place(relx=0.01, rely=0.01)

    def login_portal(self):
        # TITLE
        self.title("CHATBOT")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # CONFIGURE
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # SET THE LEFT FRAME
        self.frame_left = customtkinter.CTkFrame(master=self, width=180, corner_radius=10)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        # SET THE RIGHT FRAME
        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        # CHATBOT LABEL
        self.label_1 = customtkinter.CTkLabel(master=self.frame_left, text="CHATBOT", text_font=("Roboto Medium", -16))
        self.label_1.grid(row=0, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left, text="Test your stress",
                                                command=self.button_event1)
        self.button_1.grid(row=3, column=0, pady=10, padx=20)

        # SEND BUTTON
        self.button_2 = customtkinter.CTkButton(master=self.frame_right, text="SEND", command=self.send_function)
        self.button_2.grid(row=4, column=2, pady=10, padx=20)

        # WELCOME
        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="WELCOME <USERNAME>!!")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        # OPTION MENU
        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left, values=["Dark", "Light", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # CHATBOT
        self.frame_info = customtkinter.CTkFrame(master=self.frame_right, height=440, width=350)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=10, padx=10, sticky="nsew")

        # USER INPUT
        self.entry2 = customtkinter.CTkEntry(master=self.frame_right, width=200, placeholder_text="TYPE YOUR MESSAGE!!")
        self.entry2.grid(row=4, column=0, columnspan=2, pady=2, padx=10, sticky="we")

        # CHAT LOG
        self.ChatLog = Text(master=self.frame_info, bd=0, bg="#A8B9E5", height=26, width=45)
        self.ChatLog.grid(row=0, column=0, columnspan=2, rowspan=4, pady=10, padx=10, sticky="nsew")
        self.ChatLog.config(state=DISABLED)

        # SCROLL BAR
        self.scroll_bar = customtkinter.CTkScrollbar(master=self.frame_right, command=self.ChatLog.yview,
                                                     cursor="heart")
        self.ChatLog['yscrollcommand'] = self.scroll_bar.set
        self.scroll_bar.place(x=383, y=10, height=542)

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def send_function(self):
        msg = self.entry2.get().strip()
        if msg != '':
            self.ChatLog.config(state=NORMAL)
            self.ChatLog.insert(END, "You: " + msg + '\n\n')
            self.ChatLog.config(foreground="#442265", background="#B0B0B0")
            res = chatres(msg)
            self.ChatLog.insert(END, "Bot: " + res + '\n\n')
            self.ChatLog.config(state=DISABLED)
            self.ChatLog.yview(END)


    def button_event1(self):
        # load image with PIL and convert to PhotoImage
        self.image1 = Image.open(PATH + "/test_images/galaxy.jpg").resize((self.WIDTH + 200, self.HEIGHT + 130))
        self.image = self.image1.filter(ImageFilter.BLUR)
        self.bg_image = ImageTk.PhotoImage(self.image)

        self.image_label = tkinter.Label(master=self, image=self.bg_image)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=400,
                                            height=500,
                                            corner_radius=10, bg_color=("black", "#190F49"))
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.Cht = Text(master=self.frame, bd=0, bg="#A8B9E5", height=30, width=50)
        self.Cht.place(relx=0.005, rely=0.005)

        self.label_1 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="1.How often are you able to stay focused on the present moment?",
                                              fg_color="black")
        self.label_1.place(relx=0.04, rely=0.005)
        self.entry_11 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                               placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                               text_color="black")

        self.entry_11.place(relx=0.28, rely=0.14, anchor=tkinter.CENTER)

        self.label_2 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="2.Do you fall asleep easily at night?",
                                              fg_color="black")
        self.label_2.place(relx=0.04, rely=0.2)
        self.entry_2 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                              placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                              text_color="black")

        self.entry_2.place(relx=0.28, rely=0.34, anchor=tkinter.CENTER)

        self.label_3 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="3.On average, do you get 7-8 hours of sleep?",
                                              fg_color="black")
        self.label_3.place(relx=0.04, rely=0.4)
        self.entry_3 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                              placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                              text_color="black")

        self.entry_3.place(relx=0.28, rely=0.54, anchor=tkinter.CENTER)

        self.label_4 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="4.Do you experience headaches or muscle tension?",
                                              fg_color="black")
        self.label_4.place(relx=0.04, rely=0.6)
        self.entry_4 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                              placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                              text_color="black")

        self.entry_4.place(relx=0.28, rely=0.74, anchor=tkinter.CENTER)

        self.label_5 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="5.Do you have a hard time staying focused?",
                                              fg_color="black")
        self.label_5.place(relx=0.04, rely=0.8)
        self.entry_5 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                              placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                              text_color="black")

        self.entry_5.place(relx=0.28, rely=0.94, anchor=tkinter.CENTER)

        self.button_3 = customtkinter.CTkButton(master=self, text="NEXT",
                                                corner_radius=10, command=self.next, width=80, height=20,
                                                fg_color="#6A34DD")
        self.button_3.place(relx=0.88, rely=0.94)

    def next(self):
        # load image with PIL and convert to PhotoImage
        self.image1 = Image.open(PATH + "/test_images/galaxy.jpg").resize((self.WIDTH + 200, self.HEIGHT + 130))
        self.image = self.image1.filter(ImageFilter.BLUR)
        self.bg_image = ImageTk.PhotoImage(self.image)

        self.image_label = tkinter.Label(master=self, image=self.bg_image)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=400,
                                            height=500,
                                            corner_radius=10, bg_color=("black", "#190F49"))
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.Cht = Text(master=self.frame, bd=0, bg="#A8B9E5", height=30, width=50)
        self.Cht.place(relx=0.005, rely=0.005)

        self.label_1 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="6.Do you feel like withdrawing, isolating yourself?",
                                              fg_color="black")
        self.label_1.place(relx=0.04, rely=0.005)
        self.entry_6 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                              placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                              text_color="black")

        self.entry_6.place(relx=0.28, rely=0.14, anchor=tkinter.CENTER)

        self.label_2 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="7.Has there been a change in your daily habits such",
                                              fg_color="black")
        self.label_2.place(relx=0.04, rely=0.2)
        self.entry_7 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                              placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                              text_color="black")

        self.entry_7.place(relx=0.28, rely=0.34, anchor=tkinter.CENTER)

        self.label_3 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="8.Do you get irritated very often?",
                                              fg_color="black")
        self.label_3.place(relx=0.04, rely=0.4)
        self.entry_8 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                              placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                              text_color="black")

        self.entry_8.place(relx=0.28, rely=0.54, anchor=tkinter.CENTER)

        self.label_4 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="9.Do you undergo breathing difficulty?",
                                              fg_color="black")
        self.label_4.place(relx=0.04, rely=0.6)
        self.entry_9 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                              placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                              text_color="black")

        self.entry_9.place(relx=0.28, rely=0.74, anchor=tkinter.CENTER)

        self.label_5 = customtkinter.CTkLabel(master=self.Cht, width=200, height=40,
                                              text="10.Do you often overreact?",
                                              fg_color="black")
        self.label_5.place(relx=0.04, rely=0.8)
        self.entry_10 = customtkinter.CTkEntry(master=self.Cht, corner_radius=6, width=200,
                                               placeholder_text="1 2 3", fg_color="#BFE8D4", bg_color="#A8B9E5",
                                               text_color="black")

        self.entry_10.place(relx=0.28, rely=0.94, anchor=tkinter.CENTER)

        self.button_3 = customtkinter.CTkButton(master=self, text="DONE",
                                                corner_radius=10, command=self.test, width=80, height=20,
                                                fg_color="#6A34DD")
        self.button_3.place(relx=0.88, rely=0.94)

    def test(self):
        self.login_portal()
        a = self.entry_11.get()
        b = self.entry_2.get()
        c = self.entry_3.get()
        d = self.entry_4.get()
        e = self.entry_5.get()
        f = self.entry_6.get()
        g = self.entry_7.get()
        h = self.entry_8.get()
        i = self.entry_9.get()
        j = self.entry_10.get()
        print(a,b,c,d,e,f,g,h,i,j)


if __name__ == "__main__":
    app = App()
    app.start()