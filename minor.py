from __future__ import print_function, unicode_literals
import CalculateEmotion
import os
import pyfiglet 
from PyInquirer import style_from_dict, Token, prompt, Separator
from pprint import pprint
from pyfiglet import Figlet
import test
import real_time_video
f = Figlet(font='slant')
print (f.renderText('               Emotion             Recognition'))
print('\n')
print("Group Members: \nLakshya Khera\nGaurav Parihar\nShubham Tak\nShubhangi Agarwal")

style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})

questions = [
    {
        'type':'checkbox',
        'message':'Should we start analysing your emotion :D?',
        'name':'choice',
        'choices':[
            {
                'name':'Yes'
            },
            {
                'name':'No'
            }
        ]
    }
]
answers = prompt(questions, style=style)
if(answers['choice'][0]=='Yes'):
    import os
    SpeechResult , SpeechProbability = test.Speech()
    #print("Speech Result =>" , SpeechResult, SpeechProbability)
    print('Audio Analysis is done!')
    print('Analysing Facial Features')
    FacialResult, FacialProbability = real_time_video.Facial()
    #print('Facial Result =>',FacialResult,FacialProbability)
    # Runs facial features and calculate
    FinalEmotion,Accuracy = CalculateEmotion.Calculate_Emotion(SpeechResult,FacialResult)
    print(f.renderText('You are  '+ FinalEmotion))
    print("Accuracy is => ",Accuracy)

print (f.renderText('Thankyou'))
