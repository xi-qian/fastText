/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "simplefasttext.h"

#include <fenv.h>
#include <math.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>

int SimpleFastText::loadParams(const std::string& filename) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    return -1;
  }
  args_ = std::make_shared<Args>();
  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  args_->load(ifs);
  dict_->load(ifs);
  input_->load(ifs);
  output_->load(ifs);
  ifs.close();
  return 0;
}

int SimpleFastText::singlePredict(char * input_string, int * lables, float * probs, int k) {
  std::shared_ptr<Model> model = std::make_shared<Model>(input_, output_, args_, 0);
  if (args_->model == model_name::sup) {
    model->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model->setTargetCounts(dict_->getCounts(entry_type::word));
  }

  std::vector<int32_t> line, labels;
  std::istringstream ifs(input_string);
  dict_->getLine(ifs, line, labels, model->rng);
  dict_->addNgrams(line, args_->wordNgrams);
  if (line.empty()) {
    return -1;
  }
  std::vector<std::pair<real, int32_t>> predictions;
  model->predict(line, k, predictions);
  int i=0;
  for (auto it = predictions.cbegin(); it != predictions.cend() && i<k; it++, i++) {
    lables[i]=it->second;
    probs[i]=exp(it->first);
  }
  return 0;
}

extern "C" {
 static void con() __attribute__((constructor));

  void con() { 
    utils::initTables();
  }
  void* load_model(char * model_path){
    SimpleFastText * p = new SimpleFastText();
    p->loadParams(std::string(model_path));
    return p;
  }
  int predict(void* model, char * input_string, int * lables, float * probs, int k){
    return ((SimpleFastText *)model)->singlePredict(input_string, lables, probs, k);
  }
}
