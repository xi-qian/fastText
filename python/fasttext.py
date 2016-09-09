from ctypes import *

def load_lib(path):
  FTModel.lib = cdll.LoadLibrary(path)

class FTModel:
  lib = None;
  def __init__(self, path):
    self.model = FTModel.lib.load_model(path)
  def predict(self, s, k):
    lables = (c_int*k)()
    probs = (c_float*k)()
    FTModel.lib.predict(self.model, c_char_p(s), lables, probs, c_int(k))
    return lables, probs
