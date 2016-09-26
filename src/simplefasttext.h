/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_SIMPLEFASTTEXT_H
#define FASTTEXT_SIMPLEFASTTEXT_H

#include <time.h>

#include <atomic>
#include <memory>

#include "matrix.h"
#include "vector.h"
#include "dictionary.h"
#include "model.h"
#include "utils.h"
#include "real.h"
#include "args.h"

class SimpleFastText {
  private:
    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;
    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;
    std::shared_ptr<Model> model_;
  public:
    int loadParams(const std::string&);
    int singlePredict(char * input_string, char ** lables, float * probs, int k);
};

#endif
