#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animals Chatbot
"""
    
def existsInKB(object):
    # This function checks if the parameter is known to the chatbot
    mammal =        Expression.fromstring("mammal("+object+")")
    fish =          Expression.fromstring("fish("+object+")")
    amphibian =     Expression.fromstring("amphibian("+object+")")
    inverterbrate = Expression.fromstring("inverterbrate("+object+")")
    bird =          Expression.fromstring("bird("+object+")")

    if mammal in kb: return True
    elif fish in kb: return True
    elif amphibian in kb: return True
    elif inverterbrate in kb: return True
    elif bird in kb: return True
        
    else: return False
    

#######################################################
#          Task A: Initialise AIML agent
#######################################################
import aiml
# Create an aiml Kernel object.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="animalchatbot.xml")

#######################################################
#          Task A: Load Q/A csv
#######################################################
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
questions=[]
answers=[]
data = pandas.read_csv('animalsQA.csv', header=None)
[questions.append(row) for row in data[0]]
[answers.append(row) for row in data[1]]

#######################################################
#          Task B: Initialise Knowledgebase. 
#######################################################
import re
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring
kb=[]
data = pandas.read_csv('animal-kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# check for kb contradiction on startup
if(ResolutionProver().prove("", kb)): 
    print("There is a contradiciton in the knowledgebase. Exiting program")
    raise SystemExit

#######################################################
#          Task C: Initialise CNN Model
#######################################################
# sys for loading images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import keras
import keras.utils



#######################################################
#                   Welcome user
#######################################################
print("Welcome to the animal facts chatbot! Please ask an animal related question.")
#######################################################
#                    Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Goodbye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        match cmd:
            case 0:
                print(params[1])
                break
            #######################################################
            #          Task A: Similariry Matching
            #######################################################
            case 1:
                similarites = []

                # find the similarity values for each question
                for question in questions:
                    corpus = [params[1], question]

                    vectoriser = TfidfVectorizer()
                    trsfm = vectoriser.fit_transform(corpus)

                    similarites.append(cosine_similarity(trsfm[0:1], trsfm)[0,1])
                    
                # get the index for the most similar question and return the corresponding answer
                maxIndex = max(enumerate(similarites),key=lambda value: value[1])[0]
                print(answers[maxIndex])
                
            #######################################################
            #          Task B: Logical Processing 
            #######################################################
            
            case 20: # Add to knowledgbase
                word = re.findall(r'\S+', params[1])
                subject = word[2]; object = word[0]
                expression=read_expr(subject + '(' + object + ')')
                kb.append(expression)
                if(ResolutionProver().prove("", kb)): 
                    print("Sorry, this statement contradicts my knowledgebase.")
                    kb.remove(expression)
                else: 
                    if(subject == "amphibian" or subject == "inverterbrate"):
                        print('OK, I will remember that',object,'is an', subject)
                    else:
                        print('OK, I will remember that',object,'is a', subject)
                
            case 21: # Check Knowledgebase
                word = re.findall(r'\S+', params[1])
                subject = word[2]; object = word[0]
                expression=read_expr(subject + '(' + object + ')')
                answer=ResolutionProver().prove(expression, kb)
                if answer:
                    if(subject == "amphibian" or subject == "inverterbrate"):
                        print(str(answer) + ", the " + object + " is an " + subject)
                    else:
                        print(str(answer) + ", the " + object + " is a " + subject)
                   
                else:
                    # Branch where answer is false
                    if (existsInKB(object)):
                        # This branch deals with the output if the bot knows about the 
                        # subject but the answer is fase
                        if(subject == "amphibian" or subject == "inverterbrate"):
                            print(str(answer) + ", the " + object + " is not an " + subject)
                        else:
                            print(str(answer) + ", the " + object + " is not a " + subject)
                    else:
                        # Branch where subject isn't in knowledgebase
                        print("Sorry, I don't know what the "+object+" is.")
            #######################################################
            #          Task C: CNN Model - Load Image 
            #######################################################
            case 30:
                try: 
                    src = input("Enter the image name: ")
                    catimage = mpimg.imread(src)
                    imgplot = plt.imshow(catimage); plt.show()
                    
                    model = keras.models.load_model("CNN-Model.h5")
                    model.compile()
                    
                    img = keras.utils.load_img(
                        src, target_size=(180, 180)
                    )
                    img_array = keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    
                    predictions = model.predict(img_array)
                    print(predictions[0][0]*100)
                    score = (predictions[0][0])*100
                    print(f"This image is {100 - score:.2f}% cat and {score:.2f}% dog.")
                
                        
                    
                    
                except (FileNotFoundError):
                    print("File not found.")
                
                    

            case 99: # Fail case
                print("I did not get that, please try again.")
    else: print(answer)
    

    
    