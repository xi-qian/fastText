import time
import fasttext
from fasttext import FTModel
s = 'okay - more clips , but less dialogue/music please , my kids , 3 and 5 , enjoy a lot of the silly clips that they see on afv and i purchased this set for them . surprising to me , they have not shown great interest in these dvds . i think one of the issues is that the dvds include much of the dialogue by the hosts , including the afv theme music as they go to and come back from commercial breaks . the music and dialogue combined contribute to the majority of time in these videos . in short , if you were to actually calculate the amount of time devoted to clips , it really doesn \' t amount to a whole lot more than you would catch in a typical episode on tv . i guess i/we expected to see a greater compilation of clips given the longevity of the show .'
fasttext.load_lib('../libfasttext.so')
model = FTModel('../data/model.bin');
ls, ps = model.predict(s, 1);
print ls[0], ps[0]
